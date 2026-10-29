"""Quy trình 3 giai đoạn: (1) phân tích + cắt cảnh  (2) viết kịch bản  (3) lồng tiếng + dựng."""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from . import analyze, plot, render, tts
from .config import Settings
from .gemini import Gemini
from .media import Job, extract_audio, fmt_time, probe
from .select import Block, Shot, blocks_from_dict, make_blocks, plan_cut
from .transcribe import Line, load_subtitle, transcribe
from .vision import caption_shots


class Project:
    """Trạng thái một lần làm phim, lưu JSON trong workdir để chạy tiếp / sửa tay giữa các bước."""

    def __init__(self, st: Settings):
        if not st.video or not Path(st.video).is_file():
            raise RuntimeError("Chưa chọn video đầu vào.")
        self.st = st
        stem = re.sub(r"[^\w\-]+", "_", Path(st.video).stem)[:40]
        base = Path(st.workdir) if st.workdir else Path(st.video).parent
        self.wd = base / f"{stem}_reviewcut"
        self.wd.mkdir(parents=True, exist_ok=True)
        self.scenes: list[analyze.Scene] = []
        self.lines: list[Line] = []
        self.blocks: list[Block] = []
        self.duration = 0.0

    # ------------------------------------------------------------ lưu / nạp
    @property
    def plan_file(self) -> Path:
        return self.wd / "plan.json"

    def save(self) -> None:
        self.plan_file.write_text(json.dumps({
            "video": self.st.video, "duration": self.duration,
            "scenes": [s.to_dict() for s in self.scenes],
            "lines": [l.to_dict() for l in self.lines],
            "blocks": [b.to_dict() for b in self.blocks]}, ensure_ascii=False), encoding="utf-8")

    def load(self) -> bool:
        if not self.plan_file.exists():
            return False
        d = json.loads(self.plan_file.read_text(encoding="utf-8"))
        self.duration = d.get("duration", 0)
        self.scenes = [analyze.Scene(**s) for s in d["scenes"]]
        self.lines = [Line(**{**l, "tags": tuple(l.get("tags", ()))}) for l in d["lines"]]
        self.blocks = blocks_from_dict(d["blocks"])
        return True

    def gemini(self) -> Optional[Gemini]:
        return Gemini(self.st.gemini_key, self.st.gemini_model) if self.st.gemini_key.strip() else None

    # ------------------------------------------------------------ giai đoạn 1
    def stage1_analyze_and_cut(self, job: Job) -> list[Block]:
        st = self.st
        wav = str(self.wd / "audio16k.wav")
        job.log("① Whisper nghe thoại + phân tích hình ảnh / chuyển động…")
        job.progress(0.0, "Bắt đầu")
        info = probe(st.video)
        self.duration = info["duration"]

        # thoại gốc
        if st.srt_path and Path(st.srt_path).is_file():
            self.lines = load_subtitle(st.srt_path)
            job.log(f"Dùng phụ đề có sẵn: {len(self.lines)} câu.")
        else:
            side = Path(st.video).with_suffix(".srt")
            if side.is_file() and not st.srt_path:
                self.lines = load_subtitle(str(side))
                job.log(f"Tìm thấy phụ đề cạnh video ({side.name}): {len(self.lines)} câu.")
            elif info["has_audio"]:
                if not Path(wav).exists():
                    extract_audio(st.video, wav, job)
                self.lines = transcribe(wav, st.whisper_model, st.whisper_lang, st.whisper_device, job,
                                        cache=str(self.wd / "transcript.json"))
            else:
                self.lines = []
        plot.annotate(self.lines)
        keys = plot.key_lines(self.lines)
        job.log(f"Thoại: {len(self.lines)} câu · {len(keys)} câu nút thắt (kẻ phản bội, kích hoạt, tiết lộ…)")
        for l in keys[:6]:
            job.log(f"   ★ {fmt_time(l.start)}  {l.text[:70]}  [{', '.join(l.tags)}]")
        job.check()

        # hình ảnh
        self.scenes, tl, _ = analyze.analyze_video(st.video, str(self.wd), job, st.skip_head_sec, st.skip_tail_sec,
                                                   st.auto_skip_credits, wav if Path(wav).exists() else None)
        job.check()

        # chọn cảnh
        shots = plan_cut(self.scenes, tl, self.lines, st, log=job.log)
        # mô tả thị giác trên đúng các shot đã chọn
        caption_shots(st.video, shots, str(self.wd), self.gemini(), job)
        self.blocks = make_blocks(shots, self.lines, st)
        self.save()
        total = sum(b.dur for b in self.blocks)
        job.log(f"✔ Đã cắt xong: {sum(len(b.shots) for b in self.blocks)} shot · {len(self.blocks)} khối · {total/60:.1f} phút")
        job.progress(1.0, "Xong bước 1")
        return self.blocks

    # ------------------------------------------------------------ giai đoạn 2
    def stage2_script(self, job: Job) -> list[Block]:
        from .script import write_script
        if not self.blocks:
            raise RuntimeError("Chạy bước 1 (phân tích & cắt cảnh) trước.")
        job.log("② Viết kịch bản bám sát các shot đã cắt…")
        write_script(self.blocks, self.st, self.gemini(), job)
        self.save()
        job.log(f"✔ Kịch bản: {sum(len(b.text.split()) for b in self.blocks)} từ")
        job.progress(1.0, "Xong bước 2")
        return self.blocks

    # ------------------------------------------------------------ giai đoạn 3
    def stage3_render(self, job: Job, out_path: Optional[str] = None) -> str:
        st = self.st
        if not self.blocks or not any(b.text.strip() for b in self.blocks if b.role == "main"):
            raise RuntimeError("Chưa có kịch bản. Chạy bước 2 hoặc nhập lời bình vào bảng.")
        job.log("③ Lồng tiếng → dựng → trộn nhạc → phụ đề…")
        items, files = [], {}
        for b in self.blocks:
            if b.role == "main" and b.text.strip():
                f = tts.cache_path(str(self.wd), b.text, st)
                files[b.id] = f
                items.append((b.text, f))
        tts.synth_many(items, st, job)
        job.check()
        out = out_path or str(self.wd / f"{Path(st.video).stem}_review.mp4")
        render.render_all(st.video, self.blocks, st, str(self.wd), out, job, files)
        job.log(f"✔ Xuất: {out}")
        return out

    def run_all(self, job: Job) -> str:
        self.stage1_analyze_and_cut(job)
        self.stage2_script(job)
        return self.stage3_render(job)
