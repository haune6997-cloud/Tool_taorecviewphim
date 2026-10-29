import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from reviewcut import plot, render, script
from reviewcut.analyze import rank_norm
from reviewcut.config import GENRES, Settings
from reviewcut.media import Job, find_tool, probe
from reviewcut.select import Block, Shot
from reviewcut.transcribe import Line, load_subtitle


# ---------------------------------------------------------------- thoại nút thắt
@pytest.mark.parametrize("text,min_score", [
    ("Kẻ phản bội chính là anh!", 0.95), ("Kich hoat robot ngay!", 0.85),
    ("Thật ra tôi biết từ lâu...", 0.8), ("The traitor is among us", 0.95),
])
def test_key_lines_detected(text, min_score):
    assert plot.score_line(text)[0] >= min_score


@pytest.mark.parametrize("text", ["Trời hôm nay đẹp quá", "Ăn cơm chưa?", "Hello how are you"])
def test_small_talk_not_key(text):
    assert plot.score_line(text)[0] < 0.2


def test_load_srt_and_vtt(tmp_path):
    f = tmp_path / "a.srt"
    f.write_text("1\n00:00:01,000 --> 00:00:02,500\n<i>Xin chào</i>\nbạn\n\n2\n00:01:00,000 --> 00:01:01,000\nHai\n",
                 encoding="utf-8-sig")
    ls = load_subtitle(str(f))
    assert [(l.start, l.end, l.text) for l in ls] == [(1.0, 2.5, "Xin chào bạn"), (60.0, 61.0, "Hai")]


def test_rank_norm_ties_and_range():
    r = rank_norm(np.array([5, 1, 3, 3, 9.0]))
    assert r.min() == 0 and r.max() == 1 and r[2] == r[3]


# ---------------------------------------------------------------- kịch bản
def test_fit_words_and_soften():
    assert script.fit_words("Một hai ba. Bốn năm sáu bảy tám.", 4) == "Một hai ba."
    assert len(script.fit_words("a b c d e f g h", 3).split()) == 3
    assert script.soften("Hắn giết chết nạn nhân, máu me khắp nơi") == "Hắn hạ gục nạn nhân, vết thương khắp nơi"
    assert "giết" not in script.soften("Giết!")


def test_genres_have_all_fields():
    for name, g in GENRES.items():
        for k in ("w_motion", "w_face", "w_plot", "w_audio", "w_dark", "rate", "pitch", "voice", "grade", "style"):
            assert k in g, (name, k)


# ---------------------------------------------------------------- đồng bộ thời lượng
def _block(i, dur, text="x"):
    return Block(i, [Shot(0, dur, 0)], text=text)


def test_timing_voice_shorter_than_video_keeps_video_length():
    st = Settings(speed_ramp=1.0)
    t = render.plan_timing([_block(0, 10)], {0: 5.0}, st)[0]
    assert t.total == pytest.approx(10) and t.stretch == 1 and t.freeze == 0 and t.atempo == 1


def test_timing_voice_longer_slows_then_speeds_then_freezes():
    st = Settings(speed_ramp=1.0)
    t = render.plan_timing([_block(0, 10)], {0: 14.0}, st)[0]
    assert t.atempo > 1 and t.stretch > 1
    assert t.total >= t.voice_dur + render.LEAD + render.TAIL - 1e-6
    assert t.eff_sum * t.stretch + t.freeze == pytest.approx(t.total)
    t2 = render.plan_timing([_block(0, 10)], {0: 40.0}, st)[0]          # quá dài → còn phần đứng đuôi
    assert t2.atempo == pytest.approx(render.MAX_VOICE_SPEEDUP, rel=0.2) or t2.atempo <= render.MAX_VOICE_SPEEDUP
    assert t2.eff_sum * t2.stretch + t2.freeze == pytest.approx(t2.total)


def test_timing_outro_has_no_voice():
    b = Block(0, [Shot(0, 1.4, 0)], role="outro")
    t = render.plan_timing([b], {}, Settings(speed_ramp=1.08))[0]
    assert t.voice_dur == 0 and t.total == pytest.approx(1.4 / 1.08)


def test_split_subtitle_and_ass():
    chunks = render.split_subtitle("Một câu rất dài, nói về một chuyện hết sức quan trọng. Câu thứ hai!", 30)
    assert all(len(c) <= 36 for c in chunks) and " ".join(chunks).replace(" ", "") == \
        "Mộtcâurấtdài,nóivềmộtchuyệnhếtsứcquantrọng.Câuthứhai!"
    b = _block(0, 6, "Xin chào các bạn, đây là lời bình.")
    t = render.plan_timing([b], {0: 3.0}, Settings(speed_ramp=1.0))
    ass, cues = render.build_ass(t, Settings())
    assert "Dialogue:" in ass and cues[0][0] == pytest.approx(render.LEAD) and cues[-1][1] <= t[0].total


# ---------------------------------------------------------------- các tỉ lệ khung hình (ffmpeg thật)
@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("clip") / "c.mp4"
    subprocess.run([find_tool("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=480x270:r=25",
                    "-f", "lavfi", "-i", "sine=f=300", "-t", "4", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", str(p)], check=True)
    return str(p)


@pytest.mark.parametrize("aspect,size", [("16:9", (1920, 1080)), ("2.39", (1920, 1080)), ("9:16blur", (1080, 1920)),
                                         ("9:16crop", (1080, 1920)), ("1:1", (1080, 1080))])
def test_render_shot_aspects(clip, tmp_path, aspect, size):
    st = Settings(aspect=aspect, zoom=True, vignette=True, grain=True, crop_bottom=True, grade="teal_orange")
    sh = Shot(0.5, 2.0, 0)
    t = render.Timing(Block(0, [sh]), 0, 1, 1.5 / 1.08, 1.5 / 1.08 + 0.4, 1.0, 0.4)
    out = str(tmp_path / "o.mp4")
    render.render_shot(clip, sh, t, 0, True, st, out, True, Job(log=lambda m: None))
    info = probe(out)
    assert (info["width"], info["height"]) == size
    assert info["duration"] == pytest.approx(1.5 / 1.0 / 1.08 + 0.4, abs=0.15)
    assert info["has_audio"]
