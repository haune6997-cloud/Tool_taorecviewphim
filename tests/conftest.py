import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def fake_tts(monkeypatch):
    """TTS giả: tạo mp3 sin-wave có độ dài ~ số từ / 3.7 giây (không cần mạng)."""
    from reviewcut import tts
    from reviewcut.media import find_tool

    async def fake_one(text, edge_voice, rate, pitch, out, retries=3):
        dur = max(0.8, len(text.split()) / (3.7 * (1 + rate / 100)))
        subprocess.run([find_tool("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i",
                        f"sine=f=180:d={dur:.2f}", "-c:a", "libmp3lame", out], check=True)
    monkeypatch.setattr(tts, "_one", fake_one)
