"""Chạy không giao diện:  python cli.py video.mp4 --minutes 8 --genre "HÀNH ĐỘNG / CHIẾN ĐẤU" """
import argparse
import sys

from reviewcut.config import GENRES, Settings, load_settings
from reviewcut.media import Job
from reviewcut.pipeline import Project


def main(argv=None):
    ap = argparse.ArgumentParser(description="ReviewCut – cắt cảnh trước, viết kịch bản sau, dựng video review")
    ap.add_argument("video")
    ap.add_argument("--stage", choices=["1", "2", "3", "all"], default="all")
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--genre", default=None, choices=list(GENRES))
    ap.add_argument("--srt", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--workdir", default="")
    ap.add_argument("--music", default="")
    ap.add_argument("--aspect", default=None)
    ap.add_argument("--whisper", default=None)
    ap.add_argument("--no-ai", action="store_true", help="không dùng Gemini")
    a = ap.parse_args(argv)
    st = load_settings()
    st.video, st.srt_path, st.title, st.workdir = a.video, a.srt, a.title, a.workdir
    for k, v in (("target_minutes", a.minutes), ("genre", a.genre), ("aspect", a.aspect), ("whisper_model", a.whisper)):
        if v is not None:
            setattr(st, k, v)
    if a.music:
        st.music_path = a.music
    if a.no_ai:
        st.gemini_key = ""
    p = Project(st)
    job = Job(progress=lambda f, m: print(f"\r[{f:5.0%}] {m:<40}", end="", flush=True))
    if a.stage != "all":
        p.load()
    {"1": p.stage1_analyze_and_cut, "2": p.stage2_script, "3": p.stage3_render, "all": p.run_all}[a.stage](job)
    print()


if __name__ == "__main__":
    sys.exit(main())
