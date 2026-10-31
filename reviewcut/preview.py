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


def shot_frames(video: str, shot: Shot, workdir: str, width: int = 360, job: Optional[Job] = None) -> list:
    """3 khung In-Memory: đầu / giữa / cuối của shot (không ghi đĩa)."""
    out = []
    for tag, t in (("a", shot.start + 0.1), ("b", (shot.start + shot.end) / 2), ("c", max(shot.start, shot.end - 0.15))):
        img = grab_frame(video, t, out=None, job=job, width=width, fmt="png")
        out.append(img)
    return out


def block_frames(video: str, block: Block, workdir: str, width: int = 240, job: Optional[Job] = None) -> list:
    """Mỗi shot của khối 1 khung giữa → dải phim In-Memory."""
    out = []
    for s in block.shots:
        t = (s.start + s.end) / 2
        img = grab_frame(video, t, out=None, job=job, width=width, fmt="png")
        out.append(img)
    return out


def effect_frame(video: str, shot: Shot, st: Settings, workdir: str, job: Optional[Job] = None):
    """Khung giữa shot qua đúng bộ lọc sẽ dùng khi dựng In-Memory."""
    from .render import render_preview_frame
    t = (shot.start + shot.end) / 2
    return render_preview_frame(video, t, st, out_png=None, job=job)


def source_thumb(video: str, workdir: str, duration: float = 0.0, width: int = 360):
    """Ảnh đại diện của video gốc (ở ~10% thời lượng) In-Memory."""
    t = max(1.0, duration * 0.1) if duration else 5.0
    return grab_frame(video, t, out=None, width=width, fmt="png")