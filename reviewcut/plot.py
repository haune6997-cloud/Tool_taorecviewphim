"""Nhận diện câu thoại "nút thắt" (tiết lộ, phản bội, kích hoạt, sinh tử…) để biết giây nào cần giữ."""
from __future__ import annotations

import re
import unicodedata

from .transcribe import Line


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


# nhóm: (trọng số, [cụm từ]) – cụm tiếng Việt viết có dấu; so khớp cả bản không dấu để chịu lỗi nhận dạng
CUES: dict[str, tuple[float, list[str]]] = {
    "tiết lộ": (0.85, ["thật ra", "thực ra", "sự thật là", "hóa ra", "bí mật", "thực chất", "chính là",
                       "không ai biết", "đã biết từ lâu", "actually", "the truth", "turns out", "secret",
                       "真相", "其实", "비밀", "진실", "사실은"]),
    "phản bội": (1.0, ["phản bội", "kẻ phản", "lừa dối", "lừa tôi", "kẻ chủ mưu", "đứng sau", "betray", "traitor",
                       "liar", "you lied", "背叛", "叛徒", "배신"]),
    "kích hoạt": (0.9, ["kích hoạt", "khởi động", "đếm ngược", "tự hủy", "activate", "launch", "countdown",
                        "self-destruct", "detonate", "启动", "发动"]),
    "sinh tử": (0.85, ["giết", "chết", "đừng bắn", "chạy đi", "cứu tôi", "cứu với", "máu", "kill", "dead",
                       "die", "run", "help me", "don't shoot", "blood", "杀", "救命", "죽", "살려"]),
    "cảm xúc": (0.6, ["yêu em", "yêu anh", "xin lỗi", "tha thứ", "tạm biệt", "mãi mãi", "con của", "i love you",
                      "sorry", "forgive", "goodbye", "forever", "我爱你", "对不起", "사랑해", "미안"]),
    "bước ngoặt": (0.7, ["không thể nào", "không thể tin", "chuyện gì xảy ra", "quá muộn", "kết thúc rồi",
                         "bắt đầu rồi", "no way", "impossible", "too late", "it's over", "what have you done",
                         "不可能", "来不及", "말도 안"]),
}
_NORM = {k: (w, [strip_accents(c) for c in cues]) for k, (w, cues) in CUES.items()}


def score_line(text: str) -> tuple[float, tuple]:
    t = strip_accents(text)
    best, tags = 0.0, []
    extra = 0.0
    for tag, (w, cues) in _NORM.items():
        hits = sum(1 for c in cues if c in t)
        if hits:
            tags.append(tag)
            best = max(best, w)
            extra += 0.08 * (hits - 1)
    s = best + min(0.2, extra)
    s += 0.08 * min(2, text.count("!")) + 0.05 * min(2, text.count("?"))
    if len(t.split()) <= 3 and s > 0:      # câu ngắn gọn, dứt khoát
        s += 0.05
    if re.search(r"\.\.\.|…", text) and s > 0:
        s += 0.03
    return min(1.0, s), tuple(tags)


def annotate(lines: list[Line]) -> list[Line]:
    for ln in lines:
        ln.plot, ln.tags = score_line(ln.text)
    return lines


def key_lines(lines: list[Line], thr: float = 0.7) -> list[Line]:
    return [l for l in lines if l.plot >= thr]
