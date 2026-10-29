"""Tiện ích ffmpeg / ffprobe."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Cancelled(Exception):
    """Người dùng bấm Dừng."""


class Job:
    """Cờ huỷ + callback tiến trình dùng chung cho mọi bước."""

    def __init__(self, log: Optional[Callable[[str], None]] = None,
                 progress: Optional[Callable[[float, str], None]] = None):
        self._log, self._progress = log, progress
        self.cancel_event = threading.Event()

    def log(self, msg: str) -> None:
        (self._log or (lambda m: print(m, flush=True)))(msg)

    def progress(self, frac: float, msg: str = "") -> None:
        if self._progress:
            self._progress(max(0.0, min(1.0, frac)), msg)

    def check(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()

    def cancel(self) -> None:
        self.cancel_event.set()


def find_tool(name: str) -> str:
    """Tìm ffmpeg/ffprobe: cạnh dự án (bỏ qua file con trỏ Git LFS nhỏ) rồi tới PATH."""
    exe = name + (".exe" if os.name == "nt" else "")
    here = Path(__file__).resolve().parent.parent
    for d in (here, here / "bin", Path.cwd()):
        p = d / exe
        if p.is_file() and p.stat().st_size > 1_000_000:
            return str(p)
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"Không tìm thấy {name}. Cài ffmpeg và thêm vào PATH (https://ffmpeg.org).")
    return found


def run(cmd: list[str], job: Optional[Job] = None, cwd: Optional[str] = None, timeout: Optional[float] = None) -> str:
    """Chạy lệnh, trả stderr; ném lỗi kèm đoạn cuối log ffmpeg nếu thất bại."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, creationflags=NO_WINDOW)
    try:
        while True:
            try:
                out, err = p.communicate(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                if job and job.cancel_event.is_set():
                    p.kill()
                    p.communicate()
                    raise Cancelled()
        if p.returncode != 0:
            tail = err.decode("utf-8", "replace")[-1500:]
            raise RuntimeError(f"Lệnh thất bại ({Path(cmd[0]).name}):\n{tail}")
        return err.decode("utf-8", "replace")
    finally:
        if p.poll() is None:
            p.kill()


def probe(path: str) -> dict:
    out = subprocess.run([find_tool("ffprobe"), "-v", "error", "-print_format", "json", "-show_format",
                          "-show_streams", path], capture_output=True, creationflags=NO_WINDOW)
    if out.returncode != 0:
        raise RuntimeError(f"Không đọc được video: {out.stderr.decode('utf-8', 'replace')[-300:]}")
    info = json.loads(out.stdout)
    v = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    if v is None:
        raise RuntimeError("File không có luồng video.")
    num, _, den = (v.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 25.0
    dur = float(info["format"].get("duration") or v.get("duration") or 0)
    return dict(duration=dur, fps=fps or 25.0, width=int(v["width"]), height=int(v["height"]),
                has_audio=any(s.get("codec_type") == "audio" for s in info["streams"]))


def duration_of(path: str) -> float:
    out = subprocess.run([find_tool("ffprobe"), "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", path], capture_output=True, creationflags=NO_WINDOW)
    try:
        return float(out.stdout.decode().strip())
    except ValueError:
        return 0.0


def has_filter(name: str) -> bool:
    out = subprocess.run([find_tool("ffmpeg"), "-hide_banner", "-filters"], capture_output=True,
                         creationflags=NO_WINDOW).stdout.decode("utf-8", "replace")
    return any(len(t) > 1 and t[1] == name for t in (ln.split() for ln in out.splitlines()))


def has_encoder(name: str) -> bool:
    out = subprocess.run([find_tool("ffmpeg"), "-hide_banner", "-encoders"], capture_output=True,
                         creationflags=NO_WINDOW).stdout.decode("utf-8", "replace")
    return any(name in line.split() for line in out.splitlines())


def extract_audio(video: str, wav: str, job: Optional[Job] = None) -> None:
    run([find_tool("ffmpeg"), "-y", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", wav], job)


def fmt_time(sec: float) -> str:
    sec = max(0.0, sec)
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
