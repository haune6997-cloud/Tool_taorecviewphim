"""Quy trình 3 giai đoạn: (1) phân tích + cắt cảnh  (2) viết kịch bản  (3) lồng tiếng + dựng."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from . import analyze, plot, render, tts
from .config import Settings, env_key
from .ai import AIClient
from .media import Job, extract_audio, fmt_time, probe
from .select import Block, Shot, blocks_from_dict, make_blocks, plan_cut
from .transcribe import Line, load_subtitle, transcribe
from .vision import caption_shots


def safe_filename(name: str, default: str = "review") -> str:
    """Tên file hợp lệ trên Windows/macOS/Linux (bỏ ký tự cấm, dấu chấm/cách ở cuối, tên dành riêng)."""
    n = re.sub(r'[\\/*?:"<>|\x00-\x1f]', "", name or "").strip().strip(".")
    n = re.sub(r"\s+", " ", n)
    if n.lower().endswith(".mp4"):
        n = n[:-4].rstrip()
    if n.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        n += "_"
    return n[:120] or default


def unique_path(p: Path) -> Path:
    """Không ghi đè file có sẵn: thêm ' (2)', ' (3)'…"""
    if not p.exists():
        return p
    for i in range(2, 1000):
        q = p.with_name(f"{p.stem} ({i}){p.suffix}")
        if not q.exists():
            return q
    return p


class Project:
    """Trạng thái một lần làm phim, lưu JSON trong workdir để chạy tiếp / sửa tay giữa các bước."""

    def __init__(self, st: Settings, asker=None):
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
        self.asker = asker
        self._ai: Optional[AIClient] = None

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

    def ai_client(self, stage: str = "", job: Optional[Job] = None) -> AIClient:
        """Một client DUY NHẤT cho cả dự án; mỗi bước gọi begin_stage → quyết định Ollama/dự phòng chỉ hỏi 1 lần mỗi bước."""
        st = self.st
        gk, mk = st.groq_key.strip() or env_key("GROQ_API_KEY"), st.gemini_key.strip() or env_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
        policy = st.ai_fallback
        if not gk and not mk and policy == "ask":
            policy = "offline"
        if self._ai is None:
            self._ai = AIClient(gk, st.groq_model, st.groq_vision_model, mk, st.gemini_model, st.ollama_model, st.ollama_url,
                                st.ollama_vision_model, policy, self.asker, groq_rpm=st.groq_rpm,
                                ollama_timeout=st.ollama_timeout)
        if stage:
            self._ai.begin_stage(stage, job)
        return self._ai

    gemini = ai_client

    def output_path(self) -> Path:
        """Đường dẫn file kết quả: <thư mục xuất>/<tên file>.mp4 (không ghi đè file có sẵn)."""
        st = self.st
        name = safe_filename(st.output_name or st.title, default=f"{safe_filename(Path(st.video).stem)}_review")
        d = Path(st.output_dir) if st.output_dir else (Path(st.workdir) if st.workdir else Path(st.video).parent)
        return unique_path(d / f"{name}.mp4")

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

        # Ép dùng Local Vision AI bằng cách truyền ai=None
        caption_shots(st.video, shots, str(self.wd), ai=None, job=job)

        self.blocks = make_blocks(shots, self.lines, st)
        self.save()
        total = sum(b.dur for b in self.blocks)
        job.log(f"✔ Đã cắt xong: {sum(len(b.shots) for b in self.blocks)} shot · {len(self.blocks)} khối · {total/60:.1f} phút")
        job.progress(1.0, "Xong bước 1")
        return self.blocks

    # ------------------------------------------------------------ giai đoạn 2
    def stage2_script(self, job: Job) -> list[Block]:
        from .script import generate_story_script, summarize_plot, write_script
        from .matcher import match_story_to_shots

        if not self.blocks:
            raise RuntimeError("Chạy bước 1 (phân tích & cắt cảnh) trước.")

        job.log("② Viết kịch bản dẫn đường (Script-First) & Khớp cảnh ngữ nghĩa…")
        job.progress(0.05, "Chuẩn bị AI và dữ liệu kho cảnh")
        ai = self.ai_client("script", job)

        # 1. Thu thập toàn bộ các shot ứng viên cắt được ở Bước 1 (cả main và outro)
        candidate_shots: list[Shot] = []
        seen_shots = set()
        for b in self.blocks:
            for s in b.shots:
                sid = (round(s.start, 2), round(s.end, 2), s.role)
                if sid not in seen_shots:
                    seen_shots.add(sid)
                    s.dialogue = [
                        {"t": l.start, "text": l.text, "plot": l.plot}
                        for l in self.lines
                        if s.start <= l.end and l.start <= s.end
                    ]
                    candidate_shots.append(s)

        job.log(f"   ↳ Kho cảnh ứng viên: {len(candidate_shots)} shot.")

        # 2. Lấy summary từ hàm summarize_plot()
        job.progress(0.15, "AI đọc thoại & tóm tắt cốt truyện toàn phim…")
        summary = summarize_plot(self.blocks, self.lines, self.st, ai, job, str(self.wd))

        # 3. Gọi hàm generate_story_script để viết câu chuyện 3 hồi
        job.progress(0.35, "AI biên kịch câu chuyện 3 hồi kịch tính…")
        script_sentences: list[str] = []
        try:
            script_sentences = generate_story_script(summary, self.st, ai, job)
        except Exception as ex:
            job.log(f"   ⚠ Lỗi khi tạo kịch bản 3 hồi ({ex}) → chuyển về chế độ bám khối.")

        # 4. Gọi hàm match_story_to_shots để đối chiếu vector ngữ nghĩa + thị giác
        if script_sentences:
            job.log(f"   ↳ Đã tạo xong kịch bản 3 hồi gồm {len(script_sentences)} câu.")
            job.progress(0.70, "Semantic Matcher: Tự động nhặt shot khớp ngữ nghĩa & biểu cảm…")
            try:
                new_blocks = match_story_to_shots(script_sentences, candidate_shots, self.st)
                if new_blocks:
                    self.blocks = new_blocks
                else:
                    job.log("   ⚠ Không ghép được cảnh mới, giữ nguyên blocks cũ và viết kịch bản.")
                    write_script(self.blocks, self.st, ai, job, self.lines, str(self.wd))
            except Exception as ex:
                job.log(f"   ⚠ Lỗi Matcher ({ex}) → Dùng kịch bản theo khối.")
                write_script(self.blocks, self.st, ai, job, self.lines, str(self.wd))
        else:
            job.log("   ⚠ Không sinh được kịch bản 3 hồi → chuyển sang viết kịch bản theo khối.")
            write_script(self.blocks, self.st, ai, job, self.lines, str(self.wd))

        # 5. Lưu kết quả
        self.save()
        total_words = sum(len(b.text.split()) for b in self.blocks if b.role == "main")
        total_dur = sum(b.dur for b in self.blocks)
        job.log(f"✔ Kịch bản hoàn tất: {len(self.blocks)} khối · {total_words} từ · {total_dur/60:.1f} phút")
        job.progress(1.0, "Xong bước 2")
        return self.blocks

    # ------------------------------------------------------------ giai đoạn 3
    def stage3_render(self, job: Job, out_path: Optional[str] = None) -> str:
        st = self.st
        if not self.blocks or not any(b.text.strip() for b in self.blocks if b.role == "main"):
            raise RuntimeError("Chưa có kịch bản. Chạy bước 2 hoặc nhập lời bình vào bảng.")
        job.log("③ Lồng tiếng → dựng (Smart Reframe + Audio Ducking) → phụ đề…")
        items, files = [], {}
        for b in self.blocks:
            if b.role == "main" and b.text.strip():
                f = tts.cache_path(str(self.wd), b.text, st)
                files[b.id] = f
                items.append((b.text, f))
        tts.synth_many(items, st, job)
        job.check()
        out = out_path or str(self.output_path())
        render.render_all(st.video, self.blocks, st, str(self.wd), out, job, files)
        if st.cleanup_temp:
            render.cleanup_work(str(self.wd))
        job.log(f"✔ Đã xuất video: {out}")
        return out

    def run_all(self, job: Job) -> str:
        self.stage1_analyze_and_cut(job)
        self.stage2_script(job)
        return self.stage3_render(job)