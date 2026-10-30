"""Mô tả thị giác (Vision AI) cho từng shot đã chọn + trích khung hình xem trước."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from .ai import AIClient, AIError, AIUnavailable, image_part, parse_json
from .media import Job, find_tool, run
from .select import Shot

# Mô tả suy luận từ chỉ số vật lý (KHÔNG phải sự thật nhìn thấy): chỉ để hiển thị trong bảng và gợi ý nhịp.
# Nhiều biến thể + xoay vòng để không bao giờ lặp cùng một câu liên tiếp.
KIND_TEXT = {
    "hành động": ["Nhịp cắt gấp, chuyển động mạnh trong khung", "Cảnh có nhiều động tác nhanh", "Khung hình rung chuyển, tốc độ cao",
                  "Một pha hành động liên tục", "Chuyển động dồn về phía trước khung hình"],
    "nội tâm / đối thoại cận mặt": ["Cận khuôn mặt, ít chuyển động", "Khuôn mặt chiếm gần hết khung, nhịp chậm lại",
                                    "Khoảnh khắc lặng, dồn vào biểu cảm", "Góc quay sát gương mặt"],
    "tối / căng thẳng": ["Khung hình tối, ánh sáng yếu", "Bóng tối bao trùm phần lớn khung", "Cảnh thiếu sáng, nhịp chậm và nặng nề"],
    "đối thoại": ["Có người trò chuyện trong khung", "Cảnh hội thoại giữa các nhân vật", "Nhân vật trao đổi, camera gần như đứng yên"],
    "thường": ["Cảnh chuyển tiếp", "Khung hình bình thường, nhịp vừa phải", "Đoạn nối giữa hai diễn biến", "Bối cảnh được giữ rộng hơn"],
}


def heuristic_caption(s: Shot, rng: Optional[random.Random] = None) -> str:
    pool = KIND_TEXT.get(s.kind, KIND_TEXT["thường"])
    rng = rng or random.Random(int(s.start * 100))
    return rng.choice(pool)


def grab_frame(video: str, t: float, out: str, job: Optional[Job] = None, width: int = 448, fmt: str = "jpg") -> str:
    """Trích 1 khung hình tại giây ``t`` (JPEG nhỏ cho AI, PNG cho xem trước)."""
    q = ["-q:v", "5"] if fmt == "jpg" else []
    run([find_tool("ffmpeg"), "-y", "-v", "error", "-ss", f"{max(0.0, t):.2f}", "-i", video, "-frames:v", "1",
         "-vf", f"scale={width}:-2"] + q + [out], job)
    return out


def caption_shots(video: str, shots: list[Shot], workdir: str, ai: Optional[AIClient], job: Job, batch: int = 5) -> None:
    """Điền shot.caption. AI nhìn ảnh → caption_ai=True (là sự thật nhìn thấy). Không AI / lỗi → mô tả suy luận (caption_ai=False)."""
    cache_f = Path(workdir) / "captions.json"
    try:
        cache = json.loads(cache_f.read_text(encoding="utf-8")) if cache_f.exists() else {}
    except ValueError:
        cache = {}
    key = lambda s: f"{s.start:.2f}-{s.end:.2f}"
    todo = [s for s in shots if s.role == "main" and key(s) not in cache]
    for s in shots:
        if key(s) in cache:
            s.caption, s.caption_ai = cache[key(s)], True
    if ai and todo:
        fdir = Path(workdir) / "frames"
        fdir.mkdir(exist_ok=True)
        i, nbatch, total = 0, 0, (len(todo) + batch - 1) // batch
        while i < len(todo):
            job.check()
            part = todo[i:i + batch]
            nbatch += 1
            paths = []
            for j, s in enumerate(part):
                p = str(fdir / f"f{i + j:04d}.jpg")
                if not Path(p).exists():
                    grab_frame(video, (s.start + s.end) / 2, p, job)
                paths.append(p)
            prompt = (f"Bạn là trợ lý mô tả cảnh phim. Có {len(part)} ảnh, đánh số 1..{len(part)} theo thứ tự. "
                      "Với MỖI ảnh viết 1 câu tiếng Việt ≤ 20 từ: ai (đặc điểm nhận dạng như tuổi, giới tính, trang phục), "
                      "đang làm gì, ở đâu, không khí. Chỉ tả điều NHÌN THẤY, không đoán tên diễn viên, không bịa. "
                      f'Trả JSON đúng dạng {{"captions": ["...", ...]}} gồm đúng {len(part)} chuỗi.')
            parts = [{"text": prompt}] + [image_part(p) for p in paths]
            try:
                d = parse_json(ai.generate(parts, json_mode=True, temperature=0.3, max_out=120 * len(part)))
                arr = d.get("captions") or next(iter(d.values())) if isinstance(d, dict) else d
                got = 0
                for s, c in zip(part, arr):
                    if isinstance(c, str) and len(c.strip()) > 3:
                        s.caption, s.caption_ai = c.strip(), True
                        cache[key(s)] = s.caption
                        got += 1
                if got < len(part):
                    job.log(f"Vision AI chỉ trả {got}/{len(part)} mô tả ở lô {nbatch}; phần thiếu dùng mô tả suy luận.")
            except AIUnavailable:
                job.log("Vision AI tắt cho phần còn lại của bước này → mô tả suy luận từ chỉ số motion/mặt/tối.")
                break
            except (AIError, ValueError, TypeError) as ex:
                job.log(f"Vision AI lỗi ở lô {nbatch}: {str(ex).splitlines()[0]} → lô này dùng mô tả suy luận.")
            i += batch
            job.progress(0.1 + 0.85 * min(1.0, i / len(todo)), f"Vision AI mô tả hình ảnh {min(nbatch, total)}/{total}")
    rng = random.Random(7)
    last = ""
    for s in shots:
        if not s.caption:
            for _ in range(4):                      # tránh trùng liền kề
                c = heuristic_caption(s, rng)
                if c != last:
                    break
            s.caption, s.caption_ai, last = c, False, c
    cache_f.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
