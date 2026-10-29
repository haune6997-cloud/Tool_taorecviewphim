"""Mô tả thị giác (Vision AI / Image Captioning) cho từng shot đã chọn."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .gemini import Gemini, GeminiError, parse_json
from .media import Job, find_tool, run
from .select import Shot

KIND_TEXT = {
    "hành động": "Cảnh hành động dồn dập, máy quay chuyển động mạnh",
    "nội tâm / đối thoại cận mặt": "Cận mặt nhân vật, ánh nhìn đầy suy tư",
    "tối / căng thẳng": "Không gian tối, bầu không khí căng thẳng",
    "đối thoại": "Hai nhân vật đối thoại với nhau",
    "thường": "Diễn biến tiếp nối của câu chuyện",
}


def heuristic_caption(s: Shot) -> str:
    return KIND_TEXT.get(s.kind, KIND_TEXT["thường"])


def grab_frame(video: str, t: float, out: str, job: Job) -> None:
    run([find_tool("ffmpeg"), "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1",
         "-vf", "scale=480:-2", "-q:v", "4", out], job)


def caption_shots(video: str, shots: list[Shot], workdir: str, gem: Optional[Gemini], job: Job,
                  batch: int = 8) -> None:
    """Điền shot.caption. Có key → Gemini nhìn ảnh; không có / lỗi → mô tả suy ra từ chỉ số motion/mặt/tối."""
    cache_f = Path(workdir) / "captions.json"
    cache = json.loads(cache_f.read_text(encoding="utf-8")) if cache_f.exists() else {}
    key = lambda s: f"{s.start:.2f}-{s.end:.2f}"
    todo = [s for s in shots if s.role == "main" and key(s) not in cache]
    for s in shots:
        if key(s) in cache:
            s.caption = cache[key(s)]
    if gem and todo:
        fdir = Path(workdir) / "frames"
        fdir.mkdir(exist_ok=True)
        for i in range(0, len(todo), batch):
            job.check()
            part = todo[i:i + batch]
            paths = []
            for j, s in enumerate(part):
                p = str(fdir / f"f{i + j:04d}.jpg")
                grab_frame(video, (s.start + s.end) / 2, p, job)
                paths.append(p)
            prompt = ("Bạn là trợ lý mô tả cảnh phim. Với MỖI ảnh dưới đây (theo thứ tự), viết 1 câu tiếng Việt ≤ 20 từ: "
                      "ai (đặc điểm nhận dạng), đang làm gì, bối cảnh, cảm xúc/không khí. Không đoán tên diễn viên. "
                      f"Trả về JSON là mảng đúng {len(part)} chuỗi.")
            parts = [{"text": prompt}]
            for p in paths:
                parts.append(Gemini.image_part(p))
            try:
                arr = parse_json(gem.generate(parts, temperature=0.4))
                if isinstance(arr, dict):
                    arr = next(iter(arr.values()))
                for s, c in zip(part, arr):
                    if isinstance(c, str) and c.strip():
                        s.caption = c.strip()
                        cache[key(s)] = s.caption
            except (GeminiError, ValueError) as ex:
                job.log(f"Vision AI lỗi ở lô {i // batch + 1}: {ex} → dùng mô tả suy luận.")
                if isinstance(ex, GeminiError) and "403" in str(ex) or "400" in str(ex):
                    break                                 # key/model sai: đừng lặp lại
            job.progress(0.1 + 0.85 * min(1.0, (i + batch) / len(todo)), "Vision AI mô tả hình ảnh…")
    for s in shots:
        if not s.caption:
            s.caption = heuristic_caption(s)
    cache_f.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
