"""Viết kịch bản lời bình SAU KHI đã cắt cảnh, bám đúng những gì có trong từng đoạn video.

- Mỗi đoạn được đưa cho AI cùng: mô tả hình (do Vision AI nhìn ảnh), thoại gốc có mốc thời gian, vị trí trong phim,
  giới hạn số từ theo thời lượng, và tóm tắt cốt truyện TOÀN phim (tóm tắt theo từng đoạn rồi gộp, có cache).
- Kết quả được KIỂM TRA: chữ Hán/ngoại ngữ lẫn vào, cụm máy móc ("cảnh hành động dồn dập"…), câu/mở đầu trùng nhau,
  giống nhau ≥ 60% cụm 3 từ → xin AI viết lại đúng các đoạn lỗi; vẫn lỗi thì dùng bản dự phòng cho đoạn đó.
- Kịch bản dự phòng (không AI) ghép từ kho câu xoay vòng + thoại + mô tả hình thật, nhớ mọi câu đã dùng nên không lặp.
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
        "VĂN PHONG KỂ CHUYỆN (STORYTELLER): Bạn là người kể chuyện review phim kịch tính, cuốn hút; KHÔNG PHẢI người thuyết minh mô tả video.",
        "CẤM TUYỆT ĐỐI TẢ ẢNH MÁY MÓC: Nghiêm cấm xuất hiện các cụm từ: 'hình ảnh cho thấy', 'chúng ta thấy', 'trước mắt', 'khung cảnh', 'màn hình', 'trong cảnh này', 'người xem', 'video cho thấy'.",
        "BIẾN DỮ LIỆU THÀNH HÀNH ĐỘNG: Lấy dữ liệu từ 'các_cảnh' và 'thoại' để diễn giải thành diễn biến tâm lý, động cơ và xung đột thực tế của nhân vật (Ví dụ: KHÔNG VIẾT 'Hình ảnh hai robot đánh nhau sôi động', HÃY VIẾT 'Một đòn công phá cực hiểm lập tức được tung ra, dồn đối thủ vào thế ngàn cân treo sợi tóc!').",
        "KHỐNG CHẾ ĐỘ DÀI: Mỗi đoạn viết từ 1–3 câu súc tích, nhịp văn dứt khoát, số từ TUYỆT ĐỐI KHÔNG VƯỢT 'tối_đa_từ' để vừa vặn thời lượng đọc.",
        "BÁM SÁT CỐT TRUYỆN: Kết hợp 'TÓM TẮT' và 'thoại' để gọi đúng tên nhân vật hoặc bối cảnh. Nếu không rõ tên, hãy dùng đại từ cuốn hút (cậu thiếu niên, kẻ giấu mặt, tay súng bí ẩn). Tuyệt đối không bịa sự kiện ngoài luồng.",
        "MẠCH TRUYỆN LIỀN MẠCH & MỞ ĐẦU BIẾN HOÁ: Các đoạn nối nhau có tính nhân - quả, dồn dập cao trào; mở đầu mỗi đoạn phải khác biệt hoàn toàn, không lặp lại cấu trúc câu.",
        "ĐỘ AN TOÀN CHUỖI JSON: Nếu có trích dẫn lời nhân vật, bắt buộc dùng dấu nháy đơn '...' hoặc nháy góc “...”, CẤM dùng ngoặc kép thẳng (\") bên trong lời bình để tránh làm hỏng cấu trúc chuỗi JSON.",
    ]
    if first:
        rules.append("Đoạn đầu tiên: Phải là một câu MÓC (Hook) cực mạnh đánh thẳng vào xung đột gay cấn nhất để giữ chân người xem ngay giây đầu tiên.")
    if last:
        rules.append("Đoạn cuối cùng: Đưa câu chuyện đến đỉnh điểm cao trào/kết cục, để lại một câu hỏi mở hoặc nhận định sâu sắc khiến người xem suy ngẫm.")
    ctx = ""
    if summary:
        ctx += f"TÓM TẮT CỐT TRUYỆN TOÀN PHIM (Nền tảng để hiểu diễn biến):\n{summary}\n\n"
    if prev:
        ctx += f"Đoạn vừa kể trước đó (để nối tiếp mạch cảm xúc, không viết lùi lại): {json.dumps(prev[-2:], ensure_ascii=False)}\n"
    if used_openers:
        ctx += f"Các cách mở đầu vừa dùng (hãy chọn cách mở đầu khác hẳn): {json.dumps(used_openers[-6:], ensure_ascii=False)}\n"
    if avoid:
        ctx += "LẦN TRƯỚC CÁC ĐOẠN SAU BỊ LỖI VĂN PHONG, hãy đổi mới hoàn toàn: " + "; ".join(f"id {i}: {r}" for i, r in avoid.items()) + "\n"

    return ("Bạn là biên kịch kênh REVIEW PHIM hàng đầu. Video đã được dựng sẵn; nhiệm vụ của bạn là thổi hồn cốt truyện vào từng đoạn "
            "sao cho kịch tính, tự nhiên, khớp cảm xúc và vừa khít thời lượng phát sóng.\n"
            f"Tác phẩm: {st.title or '(chưa rõ tên - bám sát dữ liệu)'}\nThể loại: {st.genre}\n"
            f"Tông giọng yêu cầu: {g.get('style', 'Kịch tính, sắc bén, cuốn hút')}\n\n{ctx}\n"
            "NGUYÊN TẮC BẮT BUỘC:\n" + "\n".join(f"- {r}" for r in rules) + "\n\nDỮ LIỆU ĐOẠN PHIM:\n" + json.dumps(items, ensure_ascii=False) +
            '\n\nTRẢ VỀ DUY NHẤT một JSON hợp lệ: {"items": [{"id": <id>, "text": "<lời kể chuyện review hấp dẫn>"}]} đủ mọi id. KHÔNG viết lời chào hay bọc markdown.')


# ---------------------------------------------------------------- kịch bản dự phòng (không cần AI)
OPEN_HOOK = [
    "Một câu chuyện mà ngay từ những phút đầu tiên đã kéo người xem vào chuỗi bi kịch không lối thoát.",
    "Mọi biến cố kinh hoàng nhất thực chất chỉ bắt đầu từ một khoảnh khắc tưởng như vô hại.",
    "Hãy nhìn thật kỹ từng chi tiết ở đây, vì bất kỳ ai xuất hiện cũng có thể là kẻ nắm giữ chìa khóa sinh tử.",
    "Câu chuyện này mở màn bằng một nút thắt nghẹt thở mà chẳng ai lường trước được cái giá phải trả.",
    "Nếu rơi vào hoàn cảnh ngàn cân treo sợi tóc này, liệu có mấy ai dám đưa ra lựa chọn tàn nhẫn như vậy?",
]

CLOSERS = [
    "Mọi sợi dây định mệnh cuối cùng cũng kéo căng tới mức đứt đoạn, để lại cái kết đầy ám ảnh.",
    "Đến cuối cùng, cái giá phải trả cho từng tham vọng đều quá đắt đỏ.",
    "Màn hình khép lại nhưng câu hỏi ấy vẫn còn day dứt: nếu đứng ở vị trí đó, bạn sẽ chọn thế nào?",
    "Một cái kết không ồn ào nhưng đủ sức khiến bất kỳ ai theo dõi cũng phải lặng người hồi lâu.",
    "Tất cả đã ngã ngũ, nhưng vết hằn của sự thật này chắc chắn sẽ còn kéo dài mãi về sau.",
]

ANCHORS = [
    "Ngay khoảnh khắc này", "Chưa kịp định thần", "Bất ngờ thay", "Thế nhưng", "Tình thế lập tức đảo chiều",
    "Không một giây do dự", "Đúng lúc dầu sôi lửa bỏng", "Biến số ập đến khi", "Và rồi chuyện gì đến cũng phải đến",
    "Càng dấn sâu vào nguy hiểm", "Không còn đường lui", "Chỉ tích tắc sau đó",
]

PHASE_CLAUSE = {
    "mở đầu": [
        "những mắt xích đầu tiên bắt đầu xoay chuyển báo hiệu bão táp sắp ập tới",
        "thế giới tàn khốc của nhân vật dần lộ diện qua từng chi tiết bất thường",
        "bầu không khí bình yên giả tạo nhanh chóng bị xé toạc",
        "chưa một ai lường trước được cạm bẫy chết người đang giăng sẵn ngay trước mắt",
        "từng nhân vật lần lượt bước vào ván cờ sinh tử mà không hề hay biết",
    ],
    "giữa phim": [
        "vòng xoáy mâu thuẫn bắt đầu bùng lên dữ dội, cuốn phăng mọi toan tính ban đầu",
        "ranh giới giữa đồng minh và kẻ thù ngày càng trở nên mong manh",
        "mỗi bước đi sai lầm lúc này đều phải trả bằng cái giá cực kỳ đắt",
        "sự thật trần trụi dần bị bóc tách, phơi bày những âm mưu thâm hiểm nhất",
        "tình thế dồn ép khiến không một ai còn đường quay đầu lại",
    ],
    "cao trào": [
        "mọi uất ức và xung đột dồn nén bấy lâu nay chính thức phát nổ",
        "thời gian đang cạn kiệt từng giây, đẩy tất cả vào cuộc chiến một mất một còn",
        "bản năng sinh tồn trỗi dậy khi đối mặt với hiểm cảnh nghẹt thở",
        "không còn chỗ cho sự khoan nhượng, bất kỳ sai sót nào cũng đồng nghĩa với dấu chấm hết",
        "ván bài sinh tử đã lật ngửa, ranh giới sống chết chỉ còn tính bằng tích tắc",
    ],
    "kết phim": [
        "khói lửa dần lắng xuống, nhưng mất mát để lại thì không gì bù đắp nổi",
        "mọi toan tính và ân oán cuối cùng cũng đi đến hồi phân định rạch ròi",
        "cái kết đắng cay này chính là câu trả lời nghiệt ngã nhất cho mọi tham vọng",
        "sự thật cay đắng đã phơi bày trọn vẹn, không còn gì để che giấu",
        "hồi kết khép lại để người xem kịp nhìn nhận lại toàn bộ bi kịch vừa diễn ra",
    ],
}

KIND_CLAUSE = {
    "hành động": [
        "tiết tấu lập tức được đẩy lên mức nghẹt thở với những pha va chạm nảy lửa",
        "một đòn phản kích sấm sét được tung ra, xé toạc thế trận giằng co",
        "không một giây chần chừ, cuộc đối đầu bước vào giai đoạn đẫm máu",
        "tốc độ trận chiến bị đẩy lên cực hạn khiến đối phương hoàn toàn trở tay không kịp",
    ],
    "nội tâm / đối thoại cận mặt": [
        "khoảnh khắc tĩnh lặng này nặng nề hơn vạn lời nói",
        "chỉ một ánh mắt lạnh băng cũng đủ bộc lộ dã tâm và sự giằng xé bên trong",
        "nỗi sợ hãi và toan tính hằn rõ trên gương mặt nhân vật khi đối diện sự thật",
    ],
    "tối / căng thẳng": [
        "bầu không khí u ám bao trùm, áp lực đè nặng lên từng nhịp thở",
        "càng tiến sâu vào hiểm địa, hiểm họa rình rập càng trở nên khó lường",
        "nỗi bất an tột cùng xâm chiếm tâm trí khi bóng tối bao vây tứ phía",
    ],
    "đối thoại": [
        "cuộc chạm trán bằng lời nói này sắc bén và nguy hiểm không kém gì vũ khí",
        "từng câu chữ thốt ra đều ngầm cài cắm toan tính hiểm độc",
        "lời qua tiếng lại phơi bày rạn nứt không thể hàn gắn giữa hai bờ chiến tuyến",
    ],
    "thường": [
        "mạch truyện lặng lẽ chuyển hướng, chuẩn bị cho bước ngoặt kế tiếp",
        "một khoảng lặng then chốt làm tiền đề cho cơn bão ngầm sắp sửa bùng nổ",
        "tình thế tiếp tục tiến triển theo cách không ai có thể ngờ tới",
    ],
}

TAG_LINE = {
    "tiết lộ": ["Bức màn bí mật chấn động cuối cùng cũng bị xé toạc.", "Sự thật tàn nhẫn được phơi bày khiến mọi giả thuyết sụp đổ.", "Hóa ra tất cả chỉ là một màn kịch được dàn dựng tinh vi."],
    "phản bội": ["Nhát dao chí mạng lại đến từ chính người được tin tưởng nhất.", "Một cú lật mặt cay đắng biến niềm tin thành trò hề.", "Kẻ chủ mưu thực sự lộ diện ngay trong nội bộ."],
    "kích hoạt": ["Ngòi nổ sinh tử đã được kích hoạt, không một thế lực nào có thể ngăn lại.", "Đồng hồ định mệnh bắt đầu đếm ngược từng giây.", "Cái bẫy chính thức sập xuống, khóa chặt mọi lối thoát."],
    "sinh tử": ["Ranh giới giữa sự sống và cái chết chưa bao giờ mong manh đến thế.", "Chỉ cần chậm nửa bước, cái giá phải trả chính là mạng sống.", "Trong khoảnh khắc sinh tử này, chỉ có kẻ tàn nhẫn nhất mới sống sót."],
    "cảm xúc": ["Lời trăn trối chạm trúng vết thương lòng sâu kín nhất.", "Cảm xúc dồn nén bùng nổ, không một ai có thể giữ nổi sự bình tĩnh.", "Những ân hận muộn màng giờ đây đã trở nên vô nghĩa."],
    "bước ngoặt": ["Cục diện trận đấu bị đảo ngược hoàn toàn trong sự ngỡ ngàng.", "Một biến cố bất ngờ xảy ra, đạp đổ mọi tính toán ban đầu.", "Nước cờ táo bạo này đã làm thay đổi hoàn toàn cục diện."],
}

QUOTE_LEAD = [
    "Một tuyên bố đầy đanh thép vang lên: “{q}”",
    "Lời khẳng định lạnh lùng như gáo nước lạnh: “{q}”",
    "Câu nói then chốt làm thay đổi tất cả: “{q}”",
    "Chỉ một câu nói đủ khiến tất cả câm lặng: “{q}”",
]

SEE_LEAD = [
    "{c}",
    "Ngay lập tức, {c}",
    "Không để đối phương kịp thở, {c}",
    "Tận dụng thời cơ, {c}",
]


def _clean_caption(raw_cap: str) -> str:
    """Loại bỏ hoàn toàn các từ ngữ robot/miêu tả kỹ thuật của Vision AI."""
    c = raw_cap.strip().rstrip(".!?…")
    c = re.sub(r"^(hình ảnh (cho thấy|là)|chúng ta thấy|trong cảnh này|khung cảnh|màn hình xuất hiện|cảnh quay|video cho thấy)\s*", "", c, flags=re.I)
    c = re.sub(r"\b(sôi động|sắc nét|mờ nhạt|kỹ thuật số|máy quay)\b", "", c, flags=re.I)
    c = re.sub(r"\s+", " ", c).strip()
    return c


class OfflineWriter:
    """Viết lời bình kịch tính không cần AI. Bỏ hoàn toàn văn phong miêu tả hình ảnh."""

    def __init__(self, seed: int = 11):
        self.rng = random.Random(seed)
        self.used: set[str] = set()
        self.recent: deque[str] = deque(maxlen=4)

    def pick(self, pool: list[str]) -> str:
        fresh = [p for p in pool if norm_sentence(p) not in self.used]
        c = self.rng.choice(fresh or pool)
        self.used.add(norm_sentence(c))
        return c

    def varied(self, pool: list[str]) -> str:
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

        clean_caps = [_clean_caption(s.caption) for s in b.shots if s.caption_ai and _clean_caption(s.caption)]
        if clean_caps:
            c = self.rng.choice(clean_caps)
            if len(c.split()) >= 3:
                cap, low = c[0].upper() + c[1:], c[0].lower() + c[1:]
                lead_template = self.rng.choice(SEE_LEAD)
                built = lead_template.format(c=cap if lead_template.startswith("{c}") else low).rstrip(".") + "."
                parts.append(built)

        key = max(b.dialogue, key=lambda d: d["plot"], default=None)
        if key and key["plot"] >= 0.7:
            q = key["text"].strip().rstrip(".!?… ")
            if looks_vietnamese(q) and len(q.split()) <= 18:
                parts.append(self.pick([x.format(q=q) for x in QUOTE_LEAD]))
            else:
                tag = next((t for t in ("phản bội", "tiết lộ", "kích hoạt", "sinh tử", "bước ngoặt", "cảm xúc")
                            if any(t in (s.tags or []) for s in b.shots)), "bước ngoặt")
                parts.append(self.varied(TAG_LINE[tag]))
        elif key and looks_vietnamese(key["text"]) and len(key["text"].split()) <= 14 and self.rng.random() < 0.6:
            q = key["text"].strip().rstrip(".!?… ")
            parts.append(self.pick([f"Nhân vật thốt lên đầy cay đắng: “{q}”", f"Giữa tình thế hiểm nghèo, câu nói vang lên: “{q}”"]))

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


def _extract_items_regex(text: str) -> dict[int, str]:
    """Cứu cánh: Dùng Regex bóc tách trực tiếp từng cặp {"id": ..., "text": "..."} khi JSON bị vỡ cú pháp."""
    out = {}
    pattern = re.compile(r'\{[^{}]*?"id"\s*:\s*(\d+)[^{}]*?"text"\s*:\s*"(.*?)"[^{}]*?\}', re.DOTALL)
    for m in pattern.finditer(text):
        try:
            idx = int(m.group(1))
            val = m.group(2).encode().decode('unicode_escape', errors='ignore')
            out[idx] = re.sub(r"\s+", " ", val).strip()
        except Exception:
            continue
    return out


def _clean_json_string(s: str) -> str:
    """Xóa bỏ các khối bọc markdown và loại bỏ trailing comma."""
    s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.I)
    s = re.sub(r"\s*```$", "", s.strip())
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end + 1]
    s = re.sub(r",\s*([\]}])", r"\1", s)
    return s


def _parse_items(raw: str) -> dict[int, str]:
    if not raw or not raw.strip():
        return {}

    cleaned = _clean_json_string(raw)
    d = None
    try:
        d = json.loads(cleaned)
    except Exception:
        try:
            d = parse_json(raw)
        except Exception:
            pass

    if isinstance(d, dict):
        d = d.get("items") or next((v for v in d.values() if isinstance(v, list)), [])

    out: dict[int, str] = {}
    if isinstance(d, list):
        for it in d:
            if isinstance(it, dict) and str(it.get("text", "")).strip():
                try:
                    out[int(it["id"])] = re.sub(r"\s+", " ", str(it["text"])).strip()
                except (KeyError, ValueError, TypeError):
                    continue

    if not out:
        out = _extract_items_regex(raw)

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
                # Nếu Groq bị lỗi cụt chữ, tự động thử đổi sang Ollama nếu có
                if hasattr(ai, "mode") and ai.mode != "ollama" and getattr(ai, "can_ollama", False):
                    job.log(f"   ↳ Lô {idx} lỗi ({str(ex).splitlines()[0]}) → Tự chuyển sang Ollama...")
                    ai.mode = "ollama"
                    run_batch(batch, prev, idx, avoid, depth + 1)
                    return
                job.log(f"   ⚠ Lô {idx} lỗi: {str(ex).splitlines()[0]} → dùng dự phòng cho đoạn lỗi.")
                return
            except Exception as ex:
                job.log(f"   ⚠ Lô {idx}: Dùng cứu hộ JSON ({ex}).")
                got = _extract_items_regex(ai.generate([{"text": prompt}], json_mode=False, max_out=min(3500, 160 * len(batch) + 300)))

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


def generate_story_script(summary: str, st: Settings, ai: AIClient, job: Job) -> list[str]:
    """Cho AI viết một kịch bản review phim hoàn chỉnh theo cấu trúc 3 hồi trước khi nhặt cảnh."""
    g = st.genre_cfg()
    num_target_sentences = max(15, min(40, int(st.target_minutes * 6)))  # ~6 câu/phút

    prompt = f"""Bạn là biên kịch kênh REVIEW PHIM hàng đầu. Hãy viết kịch bản review cho bộ phim: '{st.title or 'Nội dung phim'}'.
Thể loại: {st.genre}
Tông giọng: {g.get('style', 'Kịch tính, cuốn hút, dồn dập')}

TÓM TẮT DIỄN BIẾN TOÀN BỘ PHIM:
{summary}

YÊU CẦU BẮT BUỘC:
1. Viết đúng {num_target_sentences} câu kể chuyện tiếng Việt liền mạch theo kết cấu 3 hồi:
   - Câu đầu: Móc câu (Hook) gây tò mò, giật gân, cuốn hút người xem ngay lập tức.
   - Các câu giữa: Dồn dập cao trào, đẩy mạnh xung đột, động cơ nhân vật và các nút thắt lớn.
   - Câu cuối: Đưa ra dư vị, câu hỏi mở hoặc bài học sâu sắc.
2. TUYỆT ĐỐI KHÔNG dùng các từ thuyết minh kỹ thuật hay tả ảnh máy móc như: "hình ảnh cho thấy", "chúng ta thấy", "trong cảnh này", "video này".
3. Hãy kể như một câu chuyện điện ảnh thật sự (Hắn lao vào..., Bí mật rợn người bị lật mở...).
4. Mỗi câu ngắn gọn, súc tích (12–25 từ/câu) để đọc vừa nhịp hình ảnh.

TRẢ VỀ DUY NHẤT một JSON hợp lệ:
{{"sentences": ["câu 1...", "câu 2...", "câu 3..."]}}
KHÔNG viết lời chào hỏi hay bọc văn bản thừa."""

    job.log(f"   ✍ Đang viết kịch bản dẫn đường ({num_target_sentences} câu)...")
    res = ai.generate([{"text": prompt}], json_mode=True, max_out=4000)

    cleaned = _clean_json_string(res)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "sentences" in data:
            return [s.strip() for s in data["sentences"] if s.strip()]
    except Exception:
        pass

    # Cứu hộ tách dòng nếu JSON bị lỗi
    lines = [re.sub(r'^\d+[\.\)]\s*', '', l).strip(' ",\'') for l in res.splitlines() if len(l.strip()) > 10]
    return lines[:num_target_sentences]