"""Chạy không giao diện:  python cli.py video.mp4 --minutes 8 --genre "HÀNH ĐỘNG / CHIẾN ĐẤU" --output "Review phim X" """
import argparse
import sys

from reviewcut.config import GENRES, load_settings
from reviewcut.media import Job
from reviewcut.pipeline import Project


def console_asker(ctx: dict) -> str:
    """Hỏi trên console (đúng 1 lần mỗi bước nhờ AIClient)."""
    print(f"\n⚠ AI đám mây lỗi ở «{ctx['stage']}»:")
    for e in ctx["errors"]:
        print(f"   • {e}")
    print(f"   Ollama: {ctx['ollama_msg']}")
    opts = "[o] dùng Ollama  " if ctx["can_ollama"] else ""
    while True:
        a = input(f"Chọn cho CẢ BƯỚC này: {opts}[d] kịch bản dự phòng  [x] dừng : ").strip().lower()
        if a == "o" and ctx["can_ollama"]:
            return "ollama"
        if a == "d":
            return "offline"
        if a == "x":
            return "cancel"


def main(argv=None):
    ap = argparse.ArgumentParser(description="ReviewCut – cắt cảnh trước, viết kịch bản sau, dựng video review")
    ap.add_argument("video")
    ap.add_argument("--stage", choices=["1", "2", "3", "all"], default="all")
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--genre", default=None, choices=list(GENRES))
    ap.add_argument("--srt", default="")
    ap.add_argument("--title", default="", help="tên phim (giúp AI viết kịch bản)")
    ap.add_argument("--output", default="", help="tên file kết quả (không cần .mp4)")
    ap.add_argument("--outdir", default="", help="thư mục xuất video")
    ap.add_argument("--workdir", default="")
    ap.add_argument("--music", default="")
    ap.add_argument("--aspect", default=None)
    ap.add_argument("--whisper", default=None)
    ap.add_argument("--fallback", choices=["ask", "ollama", "offline"], default=None,
                    help="khi AI đám mây lỗi: hỏi (1 lần/bước) | tự dùng Ollama | tự dùng kịch bản dự phòng")
    ap.add_argument("--keep-temp", action="store_true", help="giữ file tạm sau khi xuất")
    ap.add_argument("--srt-out", action="store_true", help="xuất thêm file .srt")
    ap.add_argument("--no-ai", action="store_true", help="không dùng AI (kịch bản dự phòng)")
    a = ap.parse_args(argv)
    st = load_settings()
    st.video, st.srt_path, st.title, st.workdir = a.video, a.srt, a.title, a.workdir
    st.output_name, st.output_dir = a.output, a.outdir
    for k, v in (("target_minutes", a.minutes), ("genre", a.genre), ("aspect", a.aspect), ("whisper_model", a.whisper),
                 ("ai_fallback", a.fallback)):
        if v is not None:
            setattr(st, k, v)
    if a.music:
        st.music_path = a.music
    if a.keep_temp:
        st.cleanup_temp = False
    if a.srt_out:
        st.export_srt = True
    if a.no_ai:
        st.groq_key = st.gemini_key = ""
        st.ai_fallback = "offline"
    p = Project(st, asker=console_asker if sys.stdin.isatty() else None)
    job = Job(progress=lambda f, m: print(f"\r[{f:5.0%}] {m:<40}", end="", flush=True))
    if a.stage != "all":
        p.load()
    res = {"1": p.stage1_analyze_and_cut, "2": p.stage2_script, "3": p.stage3_render, "all": p.run_all}[a.stage](job)
    print()
    if isinstance(res, str):
        print(f"Kết quả: {res}")


if __name__ == "__main__":
    sys.exit(main())
