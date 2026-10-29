"""Viết kịch bản lời bình SAU KHI đã cắt cảnh: AI nhìn đúng các shot đã chọn nên lời bình bám chặt hình."""
from __future__ import annotations

import json
import re
from typing import Optional

from .config import SOFTEN, VOICES, Settings
from .gemini import Gemini, GeminiError, parse_json
from .media import Job
from .select import Block


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


def offline_text(block: Block, st: Settings, first: bool, last: bool) -> str:
    """Kịch bản dự phòng (không cần AI): ghép mô tả hình + thoại nút thắt. Chất lượng thấp hơn Gemini."""
    cap = next((s.caption for s in block.shots if s.caption), "")
    key = max(block.dialogue, key=lambda d: d["plot"], default=None)
    parts = []
    if first:
        parts.append("Câu chuyện bắt đầu từ một khoảnh khắc khiến ai cũng phải tò mò.")
    if cap:
        parts.append(cap.rstrip(".") + ".")
    if key and key["plot"] >= 0.7:
        parts.append(f"Và rồi một câu nói thay đổi tất cả: “{key['text'].strip()}”")
    if last:
        parts.append("Kết thúc ở đây, nhưng dư âm của nó thì còn ở lại rất lâu.")
    return " ".join(parts) or "Mọi chuyện tiếp tục diễn biến theo hướng không ai ngờ tới."


def _prompt(blocks: list[Block], st: Settings, prev: list[str], first: bool, last: bool) -> str:
    g = st.genre_cfg()
    items = []
    for b in blocks:
        items.append({
            "id": b.id, "seconds": round(b.dur, 1), "max_words": max_words(b, st),
            "loai_canh": sorted({s.kind for s in b.shots if s.kind}),
            "hinh_anh": [s.caption for s in b.shots if s.caption],
            "thoai_goc": [f"{d['text']}" + (" [NÚT THẮT]" if d["plot"] >= 0.7 else "") for d in b.dialogue][:8],
        })
    ctx = f"Lời bình đã viết ngay trước đó (để nối mạch): {json.dumps(prev[-2:], ensure_ascii=False)}\n" if prev else ""
    return (
        "Bạn là biên kịch kênh REVIEW PHIM tiếng Việt. Video đã được CẮT SẴN; hãy viết lời bình cho từng đoạn "
        "sao cho khớp đúng hình ảnh và thoại của đoạn đó.\n"
        f"Phim: {st.title or '(không rõ tên – chỉ dựa vào dữ liệu bên dưới)'}\n"
        f"Thể loại: {st.genre}\nPhong cách giọng kể: {g['style']}\n"
        f"{ctx}"
        "QUY TẮC:\n"
        "- Tiếng Việt tự nhiên như người kể, 1–3 câu mỗi đoạn, KHÔNG vượt max_words của đoạn.\n"
        "- Chỉ dùng thông tin có trong hinh_anh và thoai_goc; không bịa tình tiết, tên riêng hay kết cục.\n"
        "- Thoại gốc có thể là ngoại ngữ: hãy chuyển ý sang tiếng Việt; câu có [NÚT THẮT] nên được nhấn mạnh hoặc trích lại.\n"
        "- Các đoạn nối mạch với nhau thành một câu chuyện liền; tránh lặp từ mở đầu.\n"
        + ("- Đoạn đầu tiên phải mở bằng một câu MÓC (hook) gây tò mò.\n" if first else "")
        + ("- Đoạn cuối cùng chốt bằng cảm xúc/câu hỏi gợi mở, kêu gọi theo dõi.\n" if last else "")
        + "- Không nhắc đến 'shot', 'đoạn video', 'mô tả', 'thoại gốc'.\n\n"
        f"DỮ LIỆU: {json.dumps(items, ensure_ascii=False)}\n\n"
        'TRẢ VỀ JSON: mảng [{"id": <id>, "text": "<lời bình>"}] đủ mọi id.'
    )


def write_script(blocks: list[Block], st: Settings, gem: Optional[Gemini], job: Job) -> list[Block]:
    narr = [b for b in blocks if b.role == "main"]
    done: dict[int, str] = {}
    if gem:
        CH = 20
        prev: list[str] = []
        for i in range(0, len(narr), CH):
            job.check()
            part = narr[i:i + CH]
            try:
                arr = parse_json(gem.generate([{"text": _prompt(part, st, prev, i == 0, i + CH >= len(narr))}]))
                if isinstance(arr, dict):
                    arr = arr.get("items") or next(iter(arr.values()))
                for it in arr:
                    if isinstance(it, dict) and str(it.get("text", "")).strip():
                        done[int(it["id"])] = str(it["text"]).strip()
            except (GeminiError, ValueError, KeyError, TypeError) as ex:
                job.log(f"Gemini viết kịch bản lô {i // CH + 1} lỗi: {ex} → dùng kịch bản dự phòng.")
            prev = [done[b.id] for b in part if b.id in done] or prev
            job.progress(0.1 + 0.85 * min(1.0, (i + CH) / len(narr)), "AI viết kịch bản…")
    else:
        job.log("Chưa có Gemini key → dùng kịch bản dự phòng (chất lượng thấp, bạn nên chỉnh tay).")
    for k, b in enumerate(narr):
        txt = done.get(b.id) or offline_text(b, st, k == 0, k == len(narr) - 1)
        txt = fit_words(txt, max_words(b, st))
        b.text = soften(txt) if st.safety_filter else txt
    for b in blocks:
        if b.role == "outro":
            b.text = ""
    return blocks
