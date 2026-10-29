import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from reviewcut import analyze, transcribe
from reviewcut.media import Job, find_tool


def test_haar_face_cascade_available():
    assert analyze._face_detector() is not None


def test_classify_uses_physical_metrics():
    S = analyze.Scene
    assert analyze.classify(S(0, 0, 5, motion=0.9)) == "hành động"
    assert analyze.classify(S(0, 0, 5, motion=0.2, face=0.8, face_size=0.12)) == "nội tâm / đối thoại cận mặt"
    assert analyze.classify(S(0, 0, 5, motion=0.3, dark=0.8)) == "tối / căng thẳng"
    assert analyze.classify(S(0, 0, 5, blank=True)) == "bỏ qua"


def test_find_tool_ignores_lfs_pointer(tmp_path, monkeypatch):
    from reviewcut import media
    (tmp_path / "ffmpeg.exe").write_text("version https://git-lfs.github.com/spec/v1\n")
    monkeypatch.setattr(media.Path, "cwd", staticmethod(lambda: tmp_path))
    assert "ffmpeg.exe" not in media.find_tool("ffmpeg") or Path(media.find_tool("ffmpeg")).stat().st_size > 1_000_000


def test_transcribe_with_fake_faster_whisper_and_cache(tmp_path, monkeypatch):
    calls = {"n": 0}

    class Seg:
        def __init__(self, s, e, t): self.start, self.end, self.text = s, e, t

    class Info: duration, language, language_probability = 10.0, "vi", 0.99

    class FakeModel:
        def __init__(self, *a, **k): calls["n"] += 1
        def transcribe(self, wav, **k):
            assert k["vad_filter"] is True
            return iter([Seg(1.0, 2.0, " Xin chào "), Seg(3.0, 4.0, "   "), Seg(5.0, 6.0, "Kẻ phản bội!")]), Info()

    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    cache = str(tmp_path / "t.json")
    job = Job(log=lambda m: None)
    a = transcribe.transcribe("x.wav", "tiny", "auto", "cpu", job, cache)
    assert [(l.start, l.text) for l in a] == [(1.0, "Xin chào"), (5.0, "Kẻ phản bội!")]
    b = transcribe.transcribe("x.wav", "tiny", "auto", "cpu", job, cache)      # lần 2 dùng cache
    assert calls["n"] == 1 and [l.text for l in b] == [l.text for l in a]
    transcribe.transcribe("x.wav", "base", "auto", "cpu", job, cache)           # đổi model → chạy lại
    assert calls["n"] == 2


def test_gemini_script_path_with_fake_client():
    """Gemini trả JSON thiếu 1 id → khối thiếu dùng dự phòng; câu quá dài bị cắt đúng ngân sách từ."""
    from reviewcut import script
    from reviewcut.config import Settings
    from reviewcut.select import Block, Shot

    class FakeGem:
        def generate(self, parts, **k):
            assert "NÚT THẮT" in parts[0]["text"]
            return "```json\n" + json.dumps([{"id": 0, "text": "Giết chết! " + "từ " * 80}]) + "\n```"

    blocks = [Block(0, [Shot(0, 6, 0, caption="Cảnh A")], dialogue=[{"t": 1, "text": "Kẻ phản bội!", "plot": 1.0}]),
              Block(1, [Shot(10, 16, 1, caption="Cảnh B")])]
    st = Settings()
    script.write_script(blocks, st, FakeGem(), Job(log=lambda m: None))
    assert len(blocks[0].text.split()) <= script.max_words(blocks[0], st)
    assert "hạ gục" in blocks[0].text and "Giết" not in blocks[0].text
    assert "Cảnh B" in blocks[1].text
