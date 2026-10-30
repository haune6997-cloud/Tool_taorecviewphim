"""Viết kịch bản lời bình SAU KHI đã cắt cảnh, bám đúng những gì có trong từng đoạn video.

* Mỗi đoạn được đưa cho AI cùng: mô tả hình (do Vision AI nhìn ảnh), thoại gốc có mốc thời gian, vị trí trong phim,
  giới hạn số từ theo thời lượng, và tóm tắt cốt truyện TOÀN phim (tóm tắt theo từng đoạn rồi gộp, có cache).
* Kết quả được KIỂM TRA: chữ Hán/ngoại ngữ lẫn vào, cụm máy móc ("cảnh hành động dồn dập"…), câu/mở đầu trùng nhau,
  giống nhau ≥ 60% cụm 3 từ → xin AI viết lại đúng các đoạn lỗi; vẫn lỗi thì dùng bản dự phòng cho đoạn đó.
* Kịch bản dự phòng (không AI) ghép từ kho câu xoay vòng + thoại + mô tả hình thật, nhớ mọi câu đã dùng nên không lặp.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import deque
from pathlib import Path
from typing import Optional

from .ai import AIClient, AIError, AIUnavailable, parse_json
from .config import BANNED_PHRASES, SOFTEN, Settings
from .media import Job, fmt_time
from .select import Block
from .transcribe import Line


# ---------------------------------------------------------------- độ dài / làm nhẹ từ ngữ
def words_per_sec(st: Settings) -> float:
    """~3.7 từ/giây ở tốc độ chuẩn của giọng edge-tts tiếng Việt, nhân theo rate đã cộng."""
    from .tts import resolve_prosody
    rate, _ = resolve_prosody(st)
    return 3.7 * (1.0 + rate / 100.0)


def max_words(block: Block, st: Settings) -> int:
    return max(4, int(block.dur * words_per_sec(st) * 0.92))


_SENT = re.compile(r"(?<=[.!?…])\s+")


def fit_words(text: str, limit: int) -> str:
    """Cắt theo câu cho vừa số từ; nếu 1 câu đã dài quá thì cắt tại từ và kết thúc bằng dấu chấm."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text.split()) <= limit:
        return text
    out, n = [], 0
    for s in _SENT.split(text):
        w = len(s.split())
        if n + w > limit:
            break
        out.append(s)
        n += w
    if out:
        return " ".join(out)
    return " ".join(text.split()[:limit]).rstrip(",;:") + "."


def soften(text: str) -> str:
    """Khiên vàng: thay từ ngữ bạo lực/nhạy cảm bằng cách nói nhẹ hơn."""
    for k in sorted(SOFTEN, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(k)}(?!\w)", SOFTEN[k], text, flags=re.I)
    return text


# ---------------------------------------------------------------- kiểm tra chất lượng
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_VI_MARK = re.compile(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", re.I)


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


def looks_vietnamese(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 6:
        return False
    return len(_VI_MARK.findall(text)) / len(letters) >= 0.12


def norm_sentence(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", strip_accents(s)).strip()


def sentences(text: str) -> list[str]:
    return [x.strip() for x in _SENT.split(re.sub(r"\s+", " ", text).strip()) if len(x.strip()) > 3]


def trigrams(text: str) -> set[tuple]:
    w = norm_sentence(text).split()
    return {tuple(w[i:i + 3]) for i in range(max(0, len(w) - 2))}


def banned_hit(text: str) -> Optional[str]:
    low = text.lower()
    return next((b for b in BANNED_PHRASES if b in low), None)


def validate_text(text: str, limit: int) -> Optional[str]:
    """Trả về lý do nếu lời bình không đạt, None nếu ổn."""
    t = text.strip()
    if len(t.split()) < 3:
        return "quá ngắn/rỗng"
    if _CJK.search(t):
        return "lẫn chữ Hán/Nhật/Hàn"
    if len(t.split()) >= 8 and not looks_vietnamese(t):
        return "không phải tiếng Việt có dấu"
    b = banned_hit(t)
    if b:
        return f"có cụm máy móc “{b}”"
    if re.search(r"\b(json|id\s*[:=]|max_words)\b", t, re.I):
        return "lọt cấu trúc dữ liệu vào lời bình"
    if len(t.split()) > limit * 1.6:
        return "dài hơn nhiều so với thời lượng"
    return None


def repetition_issues(texts: dict[int, str], order: Optional[list[int]] = None) -> dict[int, str]:
    """Tìm các đoạn lặp: trùng câu, cùng cách mở đầu ≥3 lần trong 5 đoạn gần nhất, giống ≥60% cụm 3 từ với đoạn trước."""
    ids = order or sorted(texts)
    bad: dict[int, str] = {}
    seen: dict[str, int] = {}
    openers: deque[tuple[int, str]] = deque(maxlen=5)
    prev_tri: list[tuple[int, set]] = []
    for i in ids:
        t = texts.get(i, "")
        if not t:
            continue
        for s in sentences(t):
            k = norm_sentence(s)
            if len(k.split()) >= 4 and k in seen and seen[k] != i:
                bad[i] = "trùng nguyên câu với đoạn khác"
            seen.setdefault(k, i)
        op = " ".join(norm_sentence(t).split()[:3])
        if op and sum(1 for _, o in openers if o == op) >= 2:
            bad.setdefault(i, f"mở đầu “{op}…” lặp lại")
        openers.append((i, op))
        tri = trigrams(t)
        for j, pt in prev_tri[-4:]:
            if len(tri) >= 8 and len(pt) >= 8 and len(tri & pt) / min(len(tri), len(pt)) >= 0.6:
                bad.setdefault(i, f"giống ≥60% đoạn {j + 1}")
        prev_tri.append((i, tri))
    return bad


# ---------------------------------------------------------------- tóm tắt cốt truyện toàn phim
def _dialogue_lines(blocks: list[Block], lines: Optional[list[Line]]) -> list[str]:
    out, seen = [], set()
    if lines:
        for l in sorted(lines, key=lambda x: x.start):
            t = l.text.strip()
            if len(t.split()) >= 2 or l.plot >= 0.7:
                out.append(f"[{fmt_time(l.start)}] {t}" + (" ★" if l.plot >= 0.7 else ""))
    else:
        for b in blocks:
            for d in b.dialogue:
                t = d.get("text", "").strip()
                if t and t not in seen:
                    seen.add(t)
                    out.append(f"[{fmt_time(d.get('t', 0))}] {t}" + (" ★" if d.get("plot", 0) >= 0.7 else ""))
    return out


def _chunk_lines(lines: list[str], chars: int, max_chunks: int = 10) -> list[str]:
    total = sum(len(x) + 1 for x in lines)
    chars = max(chars, total // max_chunks + 200)
    chunks, cur, n = [], [], 0
    for x in lines:
        if n + len(x) > chars and cur:
            chunks.append("\n".join(cur))
            cur, n = [], 0
        cur.append(x)
        n += len(x) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _ask_text(ai: AIClient, prompt: str, max_out: int = 500) -> str:
    out = ai.generate([{"text": prompt}], json_mode=False, temperature=0.4, max_out=max_out).strip()
    out = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", out).strip(" \"'\n")
    if _CJK.search(out):
        raise AIError("empty", "tóm tắt lẫn chữ Hán")
    return out


def summarize_plot(blocks: list[Block], lines: Optional[list[Line]], st: Settings, ai: AIClient, job: Job,
                   cache_dir: Optional[str] = None) -> str:
    """Tóm tắt cốt truyện TOÀN phim: chia thoại thành các đoạn theo thời gian (đủ đầu→cuối), tóm từng đoạn rồi gộp.
    Kết quả cache theo nội dung thoại nên chạy lại bước 2 không tốn thêm hạn mức."""
    dl = _dialogue_lines(blocks, lines)
    if len(dl) < 3:
        return ""
    sig = hashlib.sha1(("|".join(dl) + st.title + st.genre).encode()).hexdigest()[:16]
    cf = Path(cache_dir) / "summary.json" if cache_dir else None
    if cf and cf.exists():
        try:
            d = json.loads(cf.read_text(encoding="utf-8"))
            if d.get("sig") == sig and d.get("summary"):
                job.log("   ↳ Dùng lại tóm tắt cốt truyện đã lưu.")
                return d["summary"]
        except ValueError:
            pass
    small = not getattr(ai, "has_cloud", True) or getattr(ai, "mode", None) == "ollama"
    chunks = _chunk_lines(dl, 3200 if small else 5200)
    head = f"Phim: '{st.title or 'không rõ tên'}' (thể loại {st.genre}). Dòng có ★ là câu thoại then chốt. "
    job.log(f"   ↳ Đọc {len(dl)} câu thoại, tóm tắt cốt truyện theo {len(chunks)} phần…")
    parts: list[str] = []
    try:
        for k, ch in enumerate(chunks):
            job.check()
            job.progress(0.02 + 0.10 * (k + 1) / len(chunks), f"Tóm tắt cốt truyện {k + 1}/{len(chunks)}")
            parts.append(_ask_text(ai, head + f"Đây là phần {k + 1}/{len(chunks)} của lời thoại (có mốc thời gian, có thể là bản dịch thô "
                                   "hoặc ngoại ngữ). Viết 2–4 câu tiếng Việt tóm tắt CHÍNH XÁC chuyện gì xảy ra trong phần này: ai, làm gì, "
                                   "mâu thuẫn/tiết lộ nào. Chỉ dựa vào thoại, không bịa. Chỉ trả đoạn văn, không chào hỏi.\n\n" + ch, 350))
        if len(parts) == 1:
            summary = parts[0]
        else:
            joined = "\n".join(f"Phần {i + 1}: {p}" for i, p in enumerate(parts))
            summary = _ask_text(ai, head + "Dưới đây là tóm tắt từng phần theo thứ tự thời gian. Gộp thành MỘT đoạn tóm tắt cốt truyện toàn phim "
                                "khoảng 120–160 từ: nhân vật chính, mâu thuẫn trung tâm, các bước ngoặt theo đúng thứ tự, cao trào và kết cục. "
                                "Chỉ dùng thông tin có sẵn bên dưới. Chỉ trả đoạn văn tiếng Việt.\n\n" + joined, 500)
    except AIUnavailable:
        job.log("   ⚠ AI tắt cho bước này → bỏ qua tóm tắt tổng thể.")
        return ""
    except (AIError, ValueError) as ex:
        job.log(f"   ⚠ Bỏ qua tóm tắt tổng thể ({str(ex).splitlines()[0]}).")
        return ""
    job.log(f"   ★ Tóm tắt cốt truyện: {summary[:140]}…")
    if cf:
        cf.write_text(json.dumps({"sig": sig, "summary": summary}, ensure_ascii=False), encoding="utf-8")
    return summary


# ---------------------------------------------------------------- prompt
def _phase(frac: float) -> str:
    return "mở đầu" if frac < 0.2 else "giữa phim" if frac < 0.7 else "cao trào" if frac < 0.92 else "kết phim"


def _item(b: Block, st: Settings, film_len: float) -> dict:
    t0 = b.shots[0].start
    shots = []
    for s in b.shots:
        d: dict = {"phút": fmt_time(s.start)}
        d["hình"] = s.caption if s.caption_ai else f"(không rõ chi tiết; loại cảnh: {s.kind or 'thường'})"
        shots.append(d)
    thoai = [f"[{fmt_time(d['t'])}] {d['text']}" + (" ★" if d["plot"] >= 0.7 else "") for d in b.dialogue][:8]
    return {"id": b.id, "vị_trí": f"{t0 / max(film_len, 1):.0%} phim ({_phase(t0 / max(film_len, 1))})",
            "giây": round(b.dur, 1), "tối_đa_từ": max_words(b, st), "các_cảnh": shots,
            "thoại": thoai or ["(không có thoại trong đoạn này – kể nối mạch từ đoạn trước và các cảnh)"]}


def build_prompt(blocks: list[Block], st: Settings, prev: list[str], first: bool, last: bool, summary: str,
                 film_len: float, avoid: Optional[dict[int, str]] = None, used_openers: Optional[list[str]] = None) -> str:
    g = st.genre_cfg()
    items = [_item(b, st, film_len) for b in blocks]
    rules = [
        "Mỗi đoạn 1–3 câu tiếng Việt tự nhiên như người kể chuyện review phim; số từ KHÔNG vượt 'tối_đa_từ'.",
        "Nội dung đoạn nào phải KHỚP ĐÚNG đoạn đó: dựa vào 'các_cảnh' (điều đang nhìn thấy) và 'thoại' (điều đang được nói) của chính đoạn. "
        "Lồng hành động nhìn thấy vào lời kể như đang kể chuyện (vd 'Hắn lao vào…', 'Cô ấy chùn bước…'), KHÔNG tả máy quay/kỹ thuật.",
        "Chỉ dùng thông tin có trong TÓM TẮT, 'các_cảnh', 'thoại'. KHÔNG bịa tên riêng, sự kiện, địa điểm không xuất hiện ở đó. "
        "Không biết tên thì gọi theo đặc điểm (người đàn ông áo đen, cô gái trẻ…).",
        "Thoại có thể là ngoại ngữ/dịch thô: hãy chuyển ý sang tiếng Việt, đừng chép nguyên; câu có ★ là then chốt, nên được nhấn.",
        "Các đoạn nối nhau thành một câu chuyện liền mạch NHƯNG mỗi đoạn mở đầu bằng cách khác nhau; không lặp lại câu/cụm từ đã dùng.",
        "Chỉ viết tiếng Việt có dấu (tên riêng nước ngoài được giữ nguyên). Tuyệt đối không chữ Hán/Hàn/Nhật.",
        "CẤM các cụm: " + "; ".join(f"“{b}”" for b in BANNED_PHRASES[:7]) + ", và không nhắc 'shot', 'đoạn', 'mô tả hình', 'thoại gốc'.",
    ]
    if first:
        rules.append("Đoạn đầu tiên mở bằng một câu MÓC (hook) gây tò mò, gợi đúng xung đột chính của phim.")
    if last:
        rules.append("Đoạn cuối cùng chốt lại kết cục/cao trào và để lại dư vị hoặc một câu hỏi gợi mở.")
    ctx = ""
    if summary:
        ctx += f"TÓM TẮT CỐT TRUYỆN TOÀN PHIM (để hiểu bức tranh lớn):\n{summary}\n\n"
    if prev:
        ctx += f"Lời bình đoạn ngay trước đó (để nối mạch, KHÔNG lặp lại): {json.dumps(prev[-2:], ensure_ascii=False)}\n"
    if used_openers:
        ctx += f"Các kiểu mở đầu đã dùng, hãy mở khác đi: {json.dumps(used_openers[-6:], ensure_ascii=False)}\n"
    if avoid:
        ctx += "LẦN TRƯỚC CÁC ĐOẠN SAU BỊ LỖI, hãy viết lại khác hẳn: " + "; ".join(f"id {i}: {r}" for i, r in avoid.items()) + "\n"
    return ("Bạn là biên kịch kênh REVIEW PHIM tiếng Việt. Video đã được CẮT SẴN; viết lời bình cho từng đoạn sao cho khớp đúng "
            "hình và thoại của đoạn đó và đủ đọc trong thời lượng của nó.\n"
            f"Phim: {st.title or '(không rõ tên – chỉ dựa vào dữ liệu bên dưới)'}\nThể loại: {st.genre}\n"
            f"Phong cách giọng kể: {g.get('style', 'Kịch tính, cuốn hút')}\n\n{ctx}\n"
            "QUY TẮC:\n" + "\n".join(f"- {r}" for r in rules) + "\n\nDỮ LIỆU CÁC ĐOẠN:\n" + json.dumps(items, ensure_ascii=False) +
            '\n\nTRẢ VỀ DUY NHẤT một JSON object: {"items": [{"id": <id>, "text": "<lời bình>"}]} đủ mọi id, JSON hợp lệ.')


# ---------------------------------------------------------------- kịch bản dự phòng (không cần AI)
OPEN_HOOK = ["Có những câu chuyện chỉ cần vài phút đầu đã khiến người xem không rời mắt được.",
             "Mọi thứ bắt đầu từ một khoảnh khắc tưởng chừng bình thường, nhưng nó sẽ kéo theo cả chuỗi biến cố.",
             "Hãy nhớ kỹ những gì sắp diễn ra, vì từng chi tiết nhỏ ở đây đều có lý do.",
             "Câu chuyện này mở ra bằng một nút thắt mà chẳng ai lường trước được lối thoát."]
CLOSERS = ["Mọi sợi dây cuối cùng cũng được kéo căng, và kết cục để lại một dư vị rất lâu mới tan.",
           "Đến đây, mọi lựa chọn đều đã có cái giá của nó.",
           "Và khi màn hình tắt đi, câu hỏi ấy vẫn còn đọng lại: nếu là bạn, bạn sẽ chọn thế nào?",
           "Không ồn ào, nhưng cái kết ấy đủ để người xem lặng đi một lúc."]
ANCHORS = ["Ngay lúc này", "Không lâu sau đó", "Đến đây", "Chỉ ít phút sau", "Rồi bất ngờ", "Thế nhưng", "Trong lúc đó",
           "Sau những gì vừa xảy ra", "Càng về sau", "Tại thời điểm ấy", "Mà chưa kịp thở", "Lúc này"]
PHASE_CLAUSE = {
    "mở đầu": ["những mảnh ghép đầu tiên dần hiện ra", "thế giới của nhân vật được phác ra từng chút một",
                "mọi chuyện mới chỉ bắt đầu nhưng không khí đã khác thường", "chưa ai biết điều gì đang chờ phía trước",
                "các mối quan hệ lần lượt lộ diện", "một điều gì đó đã lệch khỏi quỹ đạo quen thuộc"],
    "giữa phim": ["tình thế mỗi lúc một rối hơn", "mâu thuẫn bắt đầu lộ rõ và không ai còn đường lùi",
                  "mỗi bước đi đều kéo theo hệ quả", "sự thật đang bị bóc ra từng lớp", "ranh giới giữa bạn và thù mờ đi",
                  "các lựa chọn bắt đầu có giá của nó"],
    "cao trào": ["mọi thứ dồn về một điểm bùng nổ", "chỉ một sai lầm nhỏ cũng phải trả giá đắt",
                 "thời gian gần như không còn cho ai nữa", "những gì chôn giấu bấy lâu đều trồi lên mặt đất",
                 "không còn chỗ nào để lẩn trốn", "mọi sợi dây đều căng đến cực hạn"],
    "kết phim": ["bụi lắng xuống nhưng vết hằn thì còn đó", "mọi chuyện đi dần về hồi kết",
                 "cái giá của từng quyết định mới thật sự rõ ràng", "câu trả lời gần như đã hiện ra trước mắt",
                 "những gì còn lại là hệ quả của mọi lựa chọn trước đó", "mọi thứ chậm lại để người ta kịp nhìn lại tất cả"],
}
KIND_CLAUSE = {
    "hành động": ["nhịp phim dồn dập hẳn lên", "mọi thứ bùng nổ chỉ trong vài giây", "không còn chỗ cho do dự",
                  "mọi người buộc phải hành động ngay"],
    "nội tâm / đối thoại cận mặt": ["phim chậm lại để nhìn thẳng vào điều nhân vật không nói ra", "một ánh mắt đã nói thay cả lời thoại",
                                    "khoảnh khắc lặng đi nặng hơn mọi lời nói"],
    "tối / căng thẳng": ["bầu không khí nặng nề hơn bao giờ hết", "càng đi sâu, mọi thứ càng tối và khó đoán",
                         "sự bất an len vào từng khoảng lặng"],
    "đối thoại": ["một cuộc trò chuyện tưởng nhẹ nhàng nhưng ẩn bên dưới là cả toan tính", "lời qua tiếng lại hé lộ thêm quan hệ giữa các bên",
                  "từng câu nói đều mang một ý đồ"],
    "thường": ["câu chuyện lặng lẽ xoay chuyển", "một đoạn nối nhưng không hề thừa", "mọi thứ tiếp tục tiến lên theo cách không ai ngờ"],
}
TAG_LINE = {
    "tiết lộ": ["Một sự thật được hé lộ ngay tại đây.", "Và rồi một bí mật cuối cùng cũng lộ diện.", "Hóa ra mọi chuyện không như người ta tưởng."],
    "phản bội": ["Niềm tin bị phá vỡ ngay trước mắt.", "Một sự phản bội lộ ra, khiến mọi thứ đảo chiều.", "Kẻ đứng sau cuối cùng cũng bị vạch mặt."],
    "kích hoạt": ["Một thứ đã được kích hoạt, và không còn đường quay lại.", "Đồng hồ bắt đầu đếm ngược.", "Một cái nút được nhấn, mọi thứ chuyển động."],
    "sinh tử": ["Ranh giới giữa sống và chết mong manh đến lạ.", "Cái giá lúc này là mạng sống.", "Một giây chậm trễ cũng có thể là quá muộn."],
    "cảm xúc": ["Một lời nói chạm đúng nơi mềm yếu nhất.", "Cảm xúc dâng lên, không ai giấu nổi nữa.", "Có những lời xin lỗi đến quá muộn."],
    "bước ngoặt": ["Một bước ngoặt bất ngờ xảy ra.", "Tình thế đảo ngược chỉ trong một câu nói.", "Đến đây thì mọi dự đoán đều sụp đổ."],
}
QUOTE_LEAD = ["Câu nói vang lên: “{q}”", "Và rồi nhân vật thốt ra: “{q}”", "Lời thoại then chốt đây: “{q}”", "Chỉ một câu: “{q}”"]
SEE_LEAD = ["{c}", "Trước mắt người xem: {c}", "Lúc này, {c}", "Hình ảnh hiện lên: {c}"]


class OfflineWriter:
    """Viết lời bình không cần AI. Nhớ mọi câu đã dùng + vài mở đầu gần nhất → không bao giờ lặp."""

    def __init__(self, seed: int = 11):
        self.rng = random.Random(seed)
        self.used: set[str] = set()
        self.recent: deque[str] = deque(maxlen=4)        # amorce vừa dùng → không mở đầu giống nhau liên tiếp

    def pick(self, pool: list[str]) -> str:
        fresh = [p for p in pool if norm_sentence(p) not in self.used]
        c = self.rng.choice(fresh or pool)
        self.used.add(norm_sentence(c))
        return c

    def varied(self, pool: list[str]) -> str:
        """Chọn câu chưa dùng; hết thì thêm amorce thời gian để thành câu mới."""
        fresh = [p for p in pool if norm_sentence(p) not in self.used]
        if fresh:
            return self.pick(fresh)
        for _ in range(30):
            a, c = self.rng.choice(ANCHORS), self.rng.choice(pool)
            cand = f"{a}, {c[0].lower() + c[1:]}"
            if norm_sentence(cand) not in self.used:
                self.used.add(norm_sentence(cand))
                return cand
        return self.rng.choice(pool)

    def bridge(self, kind: str, phase: str) -> str:
        clauses = KIND_CLAUSE.get(kind, KIND_CLAUSE["thường"]) + PHASE_CLAUSE[phase]
        cand = ""
        for _ in range(60):
            anchor = self.rng.choice([a for a in ANCHORS if a not in self.recent] or ANCHORS)
            cand = f"{anchor}, {self.rng.choice(clauses)}."
            if norm_sentence(cand) not in self.used:
                break
        self.used.add(norm_sentence(cand))
        self.recent.append(cand.split(",")[0])
        return cand

    def write(self, b: Block, k: int, n: int, film_len: float) -> str:
        phase = _phase(b.shots[0].start / max(film_len, 1.0))
        parts: list[str] = []
        if k == 0:
            parts.append(self.varied(OPEN_HOOK))
        caps = [s.caption.strip().rstrip(".") for s in b.shots if s.caption_ai and s.caption.strip()]
        if caps:                                              # chỉ dùng mô tả do AI NHÌN ẢNH, không dùng mô tả suy luận
            c = self.rng.choice(caps)
            cap, low = c[0].upper() + c[1:], c[0].lower() + c[1:]
            parts.append(self.pick([x.format(c=cap if x.startswith("{c}") else low) for x in SEE_LEAD]).rstrip(".") + ".")
        key = max(b.dialogue, key=lambda d: d["plot"], default=None)
        if key and key["plot"] >= 0.7:
            q = key["text"].strip().rstrip(".!?… ")
            if looks_vietnamese(q) and len(q.split()) <= 18:
                parts.append(self.pick([x.format(q=q) for x in QUOTE_LEAD]))
            else:                                             # thoại ngoại ngữ: không chép, dùng câu theo nhãn nút thắt
                tag = next((t for t in ("phản bội", "tiết lộ", "kích hoạt", "sinh tử", "bước ngoặt", "cảm xúc")
                            if any(t in (s.tags or []) for s in b.shots)), "bước ngoặt")
                parts.append(self.varied(TAG_LINE[tag]))
        elif key and looks_vietnamese(key["text"]) and len(key["text"].split()) <= 14 and self.rng.random() < 0.6:
            q = key["text"].strip().rstrip(".!?… ")
            parts.append(self.pick([f"Có người nói: “{q}”", f"Trong khung cảnh ấy vang lên câu: “{q}”"]))
        if len(parts) < 2:
            kind = b.shots[0].kind if b.shots and b.shots[0].kind in KIND_CLAUSE else "thường"
            parts.append(self.bridge(kind, phase))
        if k == n - 1:
            parts.append(self.varied(CLOSERS))
        return " ".join(parts)


def offline_text(block: Block, st: Settings, first: bool, last: bool, writer: Optional[OfflineWriter] = None,
                 film_len: float = 0.0) -> str:
    """Giữ chữ ký cũ để tương thích; dùng OfflineWriter dùng chung nếu truyền vào (để nhớ câu đã dùng)."""
    w = writer or OfflineWriter(block.id + 1)
    fl = film_len or max(s.end for s in block.shots)
    return w.write(block, 0 if first else 1, 1 if last else 3, fl)


# ---------------------------------------------------------------- điều phối viết kịch bản
def _batches(blocks: list[Block], st: Settings, film_len: float, budget: int, max_items: int) -> list[list[Block]]:
    out, cur, n = [], [], 0
    for b in blocks:
        sz = len(json.dumps(_item(b, st, film_len), ensure_ascii=False))
        if cur and (n + sz > budget or len(cur) >= max_items):
            out.append(cur)
            cur, n = [], 0
        cur.append(b)
        n += sz
    if cur:
        out.append(cur)
    return out


def _parse_items(raw: str) -> dict[int, str]:
    d = parse_json(raw)
    if isinstance(d, dict):
        d = d.get("items") or next((v for v in d.values() if isinstance(v, list)), [])
    out: dict[int, str] = {}
    for it in d if isinstance(d, list) else []:
        if isinstance(it, dict) and str(it.get("text", "")).strip():
            try:
                out[int(it["id"])] = re.sub(r"\s+", " ", str(it["text"])).strip()
            except (KeyError, ValueError, TypeError):
                continue
    return out


def write_script(blocks: list[Block], st: Settings, ai: Optional[AIClient], job: Job,
                 lines: Optional[list[Line]] = None, workdir: Optional[str] = None) -> list[Block]:
    narr = [b for b in blocks if b.role == "main"]
    film_len = max((s.end for b in narr for s in b.shots), default=1.0)
    order = [b.id for b in narr]
    by_id = {b.id: b for b in narr}
    done: dict[int, str] = {}
    stats = {"ai": 0, "retry": 0, "offline": 0}

    if ai is not None and narr:
        job.progress(0.02, "AI đọc & tóm tắt cốt truyện…")
        summary = summarize_plot(narr, lines, st, ai, job, workdir)
        small = not getattr(ai, "has_cloud", True) or getattr(ai, "mode", None) == "ollama"
        budget, max_items = (2600, 4) if small else (5200, 8)

        def run_batch(batch: list[Block], prev: list[str], idx: int, avoid: Optional[dict] = None, depth: int = 0) -> None:
            job.check()
            opener_hist = [" ".join(t.split()[:4]) for t in (done.get(i) for i in order) if t]
            prompt = build_prompt(batch, st, prev, batch[0].id == order[0], batch[-1].id == order[-1], summary, film_len,
                                  avoid, opener_hist)
            try:
                got = _parse_items(ai.generate([{"text": prompt}], json_mode=True, temperature=0.85 if avoid else 0.75,
                                               max_out=min(3500, 160 * len(batch) + 300)))
            except AIUnavailable:
                raise
            except AIError as ex:
                if ex.kind in ("too_large", "bad_request", "safety", "empty") and len(batch) > 1 and depth < 4:
                    job.log(f"   ↳ Lô {idx} bị từ chối ({ex.short}) → chia đôi lô rồi thử lại.")
                    h = len(batch) // 2
                    run_batch(batch[:h], prev, idx, avoid, depth + 1)
                    run_batch(batch[h:], [done[batch[h - 1].id]] if batch[h - 1].id in done else prev, idx, avoid, depth + 1)
                    return
                job.log(f"   ⚠ Lô {idx} lỗi: {str(ex).splitlines()[0]} → dùng dự phòng cho đoạn lỗi.")
                return
            except ValueError as ex:
                job.log(f"   ⚠ Lô {idx}: AI trả dữ liệu không đọc được ({ex}).")
                return
            for b in batch:
                t = got.get(b.id)
                if t:
                    done[b.id] = t

        prev: list[str] = []
        batches = _batches(narr, st, film_len, budget, max_items)
        try:
            for bi, part in enumerate(batches, 1):
                run_batch(part, prev, bi)
                prev = [done[b.id] for b in part if b.id in done] or prev
                job.progress(0.15 + 0.65 * bi / len(batches), f"AI viết kịch bản… lô {bi}/{len(batches)}")

            # --- kiểm tra chất lượng → viết lại đúng các đoạn lỗi (tối đa 2 vòng)
            for rnd in (1, 2):
                bad = {i: validate_text(done[i], max_words(by_id[i], st)) for i in done}
                bad = {i: r for i, r in bad.items() if r}
                for i, r in repetition_issues(done, order).items():
                    bad.setdefault(i, r)
                missing = {b.id: "AI không trả đoạn này" for b in narr if b.id not in done}
                todo = {**missing, **bad}
                if not todo:
                    break
                job.log(f"   ↻ Kiểm tra lần {rnd}: {len(todo)} đoạn chưa đạt (vd: {next(iter(todo.values()))}) → yêu cầu viết lại.")
                for i in bad:
                    done.pop(i, None)
                stats["retry"] += len(todo)
                redo = [by_id[i] for i in order if i in todo]
                for bi, part in enumerate(_batches(redo, st, film_len, budget, max_items), 1):
                    ctx = [done[o] for o in order if o in done and o < part[0].id][-1:]
                    run_batch(part, ctx, bi, {b.id: todo[b.id] for b in part})
        except AIUnavailable:
            job.log("   → AI tắt cho phần còn lại: các đoạn chưa có sẽ dùng kịch bản dự phòng.")
        stats["ai"] = len(done)
    else:
        job.log("Chưa cấu hình AI → dùng kịch bản dự phòng (nên chỉnh tay).")

    # --- hoàn thiện: đoạn chưa đạt → dự phòng (không lặp), rồi cắt độ dài + khiên vàng
    writer = OfflineWriter(len(narr))
    for k, b in enumerate(narr):
        lim = max_words(b, st)
        txt = done.get(b.id)
        if not txt or validate_text(txt, lim):
            txt = writer.write(b, k, len(narr), film_len)
            stats["offline"] += 1
        for s in sentences(txt):
            writer.used.add(norm_sentence(s))
        txt = fit_words(txt, lim)
        b.text = soften(txt) if st.safety_filter else txt
    for b in blocks:
        if b.role == "outro":
            b.text = ""
    job.log(f"   Kịch bản: {stats['ai']} đoạn do AI viết · {stats['retry']} lượt viết lại · {stats['offline']} đoạn dự phòng.")
    return blocks
