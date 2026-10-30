"""Xem trước hình ảnh: khung đầu/giữa/cuối của shot, ảnh lưới cả khối, ảnh có hiệu ứng, ảnh đại diện video."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from .config import Settings
from .media import Job
from .select import Block, Shot
from .vision import grab_frame


def _dir(workdir: str) -> Path:
    d = Path(workdir) / "preview"
    d.mkdir(parents=True, exist_ok=True)
    return d


def shot_frames(video: str, shot: Shot, workdir: str, width: int = 360, job: Optional[Job] = None) -> list[str]:
    """3 khung PNG: đầu / giữa / cuối của shot (có cache)."""
    d, out = _dir(workdir), []
    for tag, t in (("a", shot.start + 0.1), ("b", (shot.start + shot.end) / 2), ("c", max(shot.start, shot.end - 0.15))):
        f = d / f"s_{hashlib.sha1(f'{video}|{t:.2f}|{width}'.encode()).hexdigest()[:10]}_{tag}.png"
        if not f.exists():
            grab_frame(video, t, str(f), job, width=width, fmt="png")
        out.append(str(f))
    return out


def block_frames(video: str, block: Block, workdir: str, width: int = 240, job: Optional[Job] = None) -> list[str]:
    """Mỗi shot của khối 1 khung giữa → dải phim (filmstrip)."""
    d, out = _dir(workdir), []
    for s in block.shots:
        t = (s.start + s.end) / 2
        f = d / f"m_{hashlib.sha1(f'{video}|{t:.2f}|{width}'.encode()).hexdigest()[:10]}.png"
        if not f.exists():
            grab_frame(video, t, str(f), job, width=width, fmt="png")
        out.append(str(f))
    return out


def effect_frame(video: str, shot: Shot, st: Settings, workdir: str, job: Optional[Job] = None) -> str:
    """Khung giữa shot qua đúng bộ lọc sẽ dùng khi dựng (khung hình, màu, cắt dải phụ đề…)."""
    from .render import render_preview_frame
    t = (shot.start + shot.end) / 2
    key = hashlib.sha1(f"{video}|{t:.2f}|{st.aspect}|{st.resolved_grade()}|{st.crop_bottom}|{st.vignette}|{st.grain}".encode()).hexdigest()[:10]
    f = _dir(workdir) / f"e_{key}.png"
    if not f.exists():
        render_preview_frame(video, t, st, str(f), job)
    return str(f)


def source_thumb(video: str, workdir: str, duration: float = 0.0, width: int = 360) -> str:
    """Ảnh đại diện của video gốc (ở ~10% thời lượng)."""
    f = _dir(workdir) / f"src_{hashlib.sha1(video.encode()).hexdigest()[:10]}.png"
    if not f.exists():
        grab_frame(video, max(1.0, duration * 0.1) if duration else 5.0, str(f), None, width=width, fmt="png")
    return str(f)
