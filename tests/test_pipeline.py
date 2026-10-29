import subprocess
from pathlib import Path

import pytest

from reviewcut.config import Settings
from reviewcut.media import Job, find_tool, probe
from reviewcut.pipeline import Project


@pytest.fixture(scope="module")
def film(tmp_path_factory):
    d = tmp_path_factory.mktemp("film")
    srcs = ["testsrc2", "mandelbrot", "smptebars", "testsrc", "rgbtestsrc", "mandelbrot"]
    for i, s in enumerate(srcs):
        subprocess.run([find_tool("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", f"{s}=s=480x270:r=25",
                        "-f", "lavfi", "-i", f"sine=f={200 + 60 * i}", "-t", "7", "-c:v", "libx264", "-preset", "ultrafast",
                        "-c:a", "aac", str(d / f"c{i}.mp4")], check=True)
    (d / "l.txt").write_text("".join(f"file c{i}.mp4\n" for i in range(len(srcs))))
    subprocess.run([find_tool("ffmpeg"), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(d / "l.txt"),
                    "-c", "copy", str(d / "film.mp4")], check=True)
    (d / "film.srt").write_text("1\n00:00:09,000 --> 00:00:11,000\nKẻ phản bội chính là anh!\n\n"
                                "2\n00:00:30,000 --> 00:00:32,000\nKích hoạt robot!\n\n"
                                "3\n00:00:20,000 --> 00:00:21,000\nChào bạn.\n", encoding="utf-8")
    return d


def test_full_pipeline_keeps_key_lines_and_renders(film, fake_tts):
    st = Settings(video=str(film / "film.mp4"), workdir=str(film / "w"), target_minutes=0.3,
                  auto_skip_credits=False, outro_shots=1, aspect="9:16blur", block_sec=8, whisper_model="tiny")
    st.speed_ramp = 1.08
    p = Project(st)
    logs = []
    job = Job(log=logs.append)
    blocks = p.stage1_analyze_and_cut(job)

    shots = [s for b in blocks for s in b.shots if s.role == "main"]
    assert shots == sorted(shots, key=lambda s: s.start)                       # đúng thứ tự phim
    for key_t in (9.0, 30.0):                                                  # thoại nút thắt phải nằm trọn trong 1 shot
        assert any(s.start <= key_t and s.end >= key_t + 1.5 for s in shots), key_t
    assert blocks[-1].role == "outro" and all(s.dur < 1.6 for s in blocks[-1].shots)
    assert not any(s.dur > st.shot_max + 1.6 for s in shots)

    p.stage2_script(job)
    assert all(b.text for b in blocks if b.role == "main") and not blocks[-1].text

    # nạp lại từ đĩa vẫn y hệt (người dùng có thể sửa kịch bản giữa các bước)
    q = Project(st)
    assert q.load() and [b.text for b in q.blocks] == [b.text for b in blocks]
    q.blocks[0].text = "Lời bình do người dùng sửa tay."
    out = q.stage3_render(job)
    info = probe(out)
    assert (info["width"], info["height"]) == (1080, 1920) and info["has_audio"]
    expected = sum(s.dur for b in q.blocks for s in b.shots) / 1.08
    assert info["duration"] >= expected * 0.95
    assert Path(out).with_suffix(".srt").read_text(encoding="utf-8").count("-->") >= 2
