"""ReviewCut – giao diện: cắt cảnh trước (Whisper + Vision + motion/mặt) → viết kịch bản sau → lồng tiếng & dựng."""
from __future__ import annotations
# --- VÁ LỖI TỰ ĐỘNG CHO AV / METADATA_ERRORS ---
import av
_orig_av_open = av.open
def _safe_av_open(*args, **kwargs):
    kwargs.pop('metadata_errors', None)
    return _orig_av_open(*args, **kwargs)
av.open = _safe_av_open
# ---------------------------------------------
import os
import sys
# Tự động nạp thư viện CUDA cuBLAS vào PATH
cuda_path = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cublas", "bin")
if os.path.exists(cuda_path):
    os.add_dll_directory(cuda_path)
    os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from reviewcut import __version__
from reviewcut.config import (ASPECTS, GENRES, GRADES, VOICES, Settings, load_settings, save_settings)
from reviewcut.media import Cancelled, Job, find_tool, fmt_time
from reviewcut.pipeline import Project
from reviewcut.script import max_words

BG, CARD, INK, MUTED = "#f2f4f7", "#ffffff", "#1f2937", "#6b7280"
TEAL, ORANGE, BLUE, RED, GREEN = "#0f8b8d", "#e8590c", "#1d6fb8", "#d64545", "#2f9e44"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"ReviewCut {__version__} – Cắt cảnh trước, viết kịch bản sau")
        self.geometry("1380x900")
        self.minsize(1100, 700)
        self.configure(bg=BG)
        self.st = load_settings()
        self.project: Project | None = None
        self.q: queue.Queue = queue.Queue()
        self.job: Job | None = None
        self.busy = False
        self.play_proc: subprocess.Popen | None = None
        self._style()
        self._vars()
        self._build()
        self.after(100, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._close)

    # ------------------------------------------------------------ giao diện
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=INK, font=("Segoe UI", 10))
        s.configure("Card.TLabelframe", background=CARD, bordercolor="#d7dce3")
        s.configure("Card.TLabelframe.Label", background=CARD, foreground=INK, font=("Segoe UI", 10, "bold"))
        s.configure("TLabel", background=CARD)
        s.configure("Muted.TLabel", foreground=MUTED, background=CARD, font=("Segoe UI", 9))
        s.configure("TCheckbutton", background=CARD)
        s.configure("TRadiobutton", background=CARD)
        for name, col in (("Teal", TEAL), ("Orange", ORANGE), ("Blue", BLUE), ("Red", RED), ("Green", GREEN)):
            s.configure(f"{name}.TButton", background=col, foreground="white", font=("Segoe UI", 10, "bold"),
                        padding=8, borderwidth=0)
            s.map(f"{name}.TButton", background=[("active", col), ("disabled", "#b8bec8")])
        s.configure("Treeview", rowheight=24, background="white", fieldbackground="white")
        s.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def _vars(self):
        st = self.st
        V = lambda cls, v: cls(value=v)
        self.v = {
            "gemini_key": V(tk.StringVar, st.gemini_key), "gemini_model": V(tk.StringVar, st.gemini_model),
            "workdir": V(tk.StringVar, st.workdir), "title": V(tk.StringVar, st.title),
            "whisper_model": V(tk.StringVar, st.whisper_model), "whisper_lang": V(tk.StringVar, st.whisper_lang),
            "video": V(tk.StringVar, st.video), "srt_path": V(tk.StringVar, st.srt_path),
            "extraction_mode": V(tk.StringVar, st.extraction_mode), "target_minutes": V(tk.DoubleVar, st.target_minutes),
            "micro_cut": V(tk.BooleanVar, st.micro_cut), "shot_len": V(tk.DoubleVar, st.shot_len),
            "highlight_priority": V(tk.BooleanVar, st.highlight_priority),
            "auto_skip_credits": V(tk.BooleanVar, st.auto_skip_credits),
            "skip_head_sec": V(tk.DoubleVar, st.skip_head_sec), "skip_tail_sec": V(tk.DoubleVar, st.skip_tail_sec),
            "outro_montage": V(tk.BooleanVar, st.outro_montage), "outro_shots": V(tk.IntVar, st.outro_shots),
            "aspect": V(tk.StringVar, next((k for k, x in ASPECTS.items() if x == st.aspect), list(ASPECTS)[0])),
            "grade": V(tk.StringVar, "Theo thể loại" if st.grade == "auto" else next((k for k, x in GRADES.items() if x == st.grade), "Theo thể loại")),
            "zoom": V(tk.BooleanVar, st.zoom), "ramp": V(tk.BooleanVar, st.speed_ramp > 1.0),
            "crop_bottom": V(tk.BooleanVar, st.crop_bottom), "vignette": V(tk.BooleanVar, st.vignette),
            "grain": V(tk.BooleanVar, st.grain), "safety_filter": V(tk.BooleanVar, st.safety_filter),
            "genre": V(tk.StringVar, st.genre),
            "voice": V(tk.StringVar, "Theo thể loại" if st.voice == "auto" else VOICES.get(st.voice, VOICES["hoaimy"])["label"]),
            "speed": V(tk.StringVar, f"{st.speed:.1f}x"),
            "music_path": V(tk.StringVar, st.music_path), "music_volume": V(tk.DoubleVar, st.music_volume),
            "orig_volume": V(tk.DoubleVar, st.orig_volume), "subtitles": V(tk.BooleanVar, st.subtitles),
            "encoder": V(tk.StringVar, st.encoder),
            "preview_text": V(tk.StringVar, "Mọi người ơi, diễn biến tiếp theo của bộ phim cực kỳ gay cấn luôn nhé!"),
        }

    def _card(self, parent, title):
        f = ttk.LabelFrame(parent, text=title, style="Card.TLabelframe", padding=10)
        f.pack(fill="x", padx=8, pady=6)
        f.columnconfigure(1, weight=1)
        return f

    def _row(self, f, r, label, widget_factory, span=1):
        ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=3)
        w = widget_factory(f)
        w.grid(row=r, column=1, columnspan=span, sticky="ew", pady=3)
        return w

    def _build(self):
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)
        # ---- trái (cuộn)
        left_outer = ttk.Frame(paned)
        cv = tk.Canvas(left_outer, bg=BG, highlightthickness=0, width=600)
        sb = ttk.Scrollbar(left_outer, orient="vertical", command=cv.yview)
        left = ttk.Frame(cv)
        left.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        win = cv.create_window((0, 0), window=left, anchor="nw")
        cv.bind("<Configure>", lambda e: cv.itemconfigure(win, width=e.width))
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        cv.bind_all("<MouseWheel>", lambda e: cv.yview_scroll(int(-e.delta / 120), "units") if self.focus_get() and "combobox" not in str(self.focus_get()).lower() else None)
        paned.add(left_outer, weight=2)
        right = ttk.Frame(paned)
        paned.add(right, weight=3)
        self.after(250, lambda: paned.sashpos(0, 540))
        self._left(left)
        self._right(right)

    def _pick(self, var, kind="file", types=None):
        p = (filedialog.askdirectory() if kind == "dir" else filedialog.askopenfilename(filetypes=types or [("Tất cả", "*.*")]))
        if p:
            var.set(p)

    def _entry_btn(self, f, var, kind, types=None, show=None):
        box = ttk.Frame(f, style="TFrame")
        e = ttk.Entry(box, textvariable=var, show=show)
        e.pack(side="left", fill="x", expand=True)
        if kind:
            ttk.Button(box, text="Chọn", width=6, command=lambda: self._pick(var, kind, types)).pack(side="left", padx=(4, 0))
        return box

    def _left(self, p):
        v = self.v
        f = self._card(p, "⚙ Cấu hình API & thư mục")
        self._row(f, 0, "Gemini Key", lambda x: self._entry_btn(x, v["gemini_key"], None, show="•"))
        self._row(f, 1, "Model Gemini", lambda x: ttk.Entry(x, textvariable=v["gemini_model"]))
        self._row(f, 2, "Thư mục lưu", lambda x: self._entry_btn(x, v["workdir"], "dir"))
        self._row(f, 3, "Tên phim", lambda x: ttk.Entry(x, textvariable=v["title"]))
        self._row(f, 4, "Whisper model", lambda x: ttk.Combobox(x, textvariable=v["whisper_model"], state="readonly",
                                                              values=["tiny", "base", "small", "medium", "large-v3"]))
        self._row(f, 5, "Ngôn ngữ phim", lambda x: ttk.Combobox(x, textvariable=v["whisper_lang"],
                                                                values=["auto", "vi", "en", "zh", "ko", "ja", "th", "fr", "es"]))
        ttk.Label(f, text="Không có Gemini key vẫn chạy được (mô tả & kịch bản dự phòng chất lượng thấp hơn).",
                  style="Muted.TLabel", wraplength=440).grid(row=6, column=0, columnspan=2, sticky="w")
        ttk.Button(f, text="💾 Lưu cấu hình", style="Blue.TButton", command=self._save).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        f = self._card(p, "🎞 Video đầu vào & phân cảnh AI")
        self._row(f, 0, "Video gốc", lambda x: self._entry_btn(x, v["video"], "file", [("Video", "*.mp4 *.mkv *.mov *.avi *.webm *.ts"), ("Tất cả", "*.*")]))
        self._row(f, 1, "Phụ đề .srt/.vtt", lambda x: self._entry_btn(x, v["srt_path"], "file", [("Phụ đề", "*.srt *.vtt")]))
        r = ttk.Frame(f)
        ttk.Radiobutton(r, text="Phân cảnh Highlight (lọc sạch)", variable=v["extraction_mode"], value="highlight").pack(side="left")
        ttk.Radiobutton(r, text="Tất cả phân cảnh", variable=v["extraction_mode"], value="all").pack(side="left", padx=10)
        self._row(f, 2, "Chế độ trích xuất", lambda x: r)
        r = ttk.Frame(f)
        ttk.Spinbox(r, from_=1, to=120, increment=1, width=6, textvariable=v["target_minutes"]).pack(side="left")
        ttk.Label(r, text="phút").pack(side="left", padx=4)
        self._row(f, 3, "Thời lượng review", lambda x: r)
        r = ttk.Frame(f)
        ttk.Checkbutton(r, text="Micro-cutting, mỗi shot", variable=v["micro_cut"]).pack(side="left")
        ttk.Spinbox(r, from_=2.0, to=3.5, increment=0.1, width=5, textvariable=v["shot_len"]).pack(side="left", padx=4)
        ttk.Label(r, text="giây (2.0–3.5)").pack(side="left")
        self._row(f, 4, "Nhịp cắt", lambda x: r)
        ttk.Checkbutton(f, text="Ưu tiên vùng chuyển động cao (Highlight Zone)", variable=v["highlight_priority"]).grid(row=5, column=0, columnspan=2, sticky="w")
        r = ttk.Frame(f)
        ttk.Checkbutton(r, text="Tự bỏ credits cuối phim", variable=v["auto_skip_credits"]).pack(side="left")
        self._row(f, 6, "", lambda x: r)
        r = ttk.Frame(f)
        ttk.Label(r, text="Bỏ đầu").pack(side="left")
        ttk.Spinbox(r, from_=0, to=1800, width=6, textvariable=v["skip_head_sec"]).pack(side="left", padx=4)
        ttk.Label(r, text="s   Bỏ cuối").pack(side="left")
        ttk.Spinbox(r, from_=0, to=1800, width=6, textvariable=v["skip_tail_sec"]).pack(side="left", padx=4)
        ttk.Label(r, text="s (intro/outro)").pack(side="left")
        self._row(f, 7, "", lambda x: r)
        self.b1 = ttk.Button(f, text="1. Whisper + Vision: Nhận dạng & CẮT CẢNH", style="Teal.TButton", command=lambda: self._run("1"))
        self.b1.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        f = self._card(p, "🎬 Hiệu ứng hình ảnh")
        self._row(f, 0, "Khung hình", lambda x: ttk.Combobox(x, textvariable=v["aspect"], values=list(ASPECTS), state="readonly"))
        self._row(f, 1, "Chỉnh màu", lambda x: ttk.Combobox(x, textvariable=v["grade"], values=["Theo thể loại"] + list(GRADES), state="readonly"))
        for i, (k, t) in enumerate([("zoom", "Zoom nhẹ chậm (punch-in) theo từng shot"),
                                    ("ramp", "Tăng tốc nhẹ 1.08x"),
                                    ("crop_bottom", "Cắt dải phụ đề cứng dưới đáy (10%)"),
                                    ("vignette", "Tối viền nghệ thuật"), ("grain", "Hạt phim (film grain)"),
                                    ("safety_filter", "Khiên vàng: làm nhẹ từ bạo lực trong lời bình")]):
            ttk.Checkbutton(f, text=t, variable=v[k]).grid(row=2 + i, column=0, columnspan=2, sticky="w")
        r = ttk.Frame(f)
        ttk.Checkbutton(r, text="Tự chèn Outro Montage (highlight liên hoàn)", variable=v["outro_montage"]).pack(side="left")
        ttk.Spinbox(r, from_=2, to=8, width=4, textvariable=v["outro_shots"]).pack(side="left", padx=4)
        ttk.Label(r, text="cảnh").pack(side="left")
        r.grid(row=8, column=0, columnspan=2, sticky="w")

        f = self._card(p, "🔊 Giọng đọc, nhạc & phụ đề")
        self._row(f, 0, "Giọng đọc", lambda x: ttk.Combobox(x, textvariable=v["voice"], state="readonly",
                                                          values=["Theo thể loại"] + [d["label"] for d in VOICES.values()]))
        self._row(f, 1, "Tốc độ đọc", lambda x: ttk.Combobox(x, textvariable=v["speed"], state="readonly",
                                                           values=[f"{i/10:.1f}x" for i in range(7, 16)]))
        self._row(f, 2, "Nhạc nền (file/thư mục)", lambda x: self._entry_btn(x, v["music_path"], "file", [("Âm thanh", "*.mp3 *.wav *.m4a *.ogg *.flac")]))
        ttk.Button(f, text="Chọn thư mục nhạc", command=lambda: self._pick(v["music_path"], "dir")).grid(row=3, column=1, sticky="w")
        self._row(f, 4, "Âm lượng nhạc", lambda x: ttk.Scale(x, from_=0.0, to=0.6, variable=v["music_volume"]))
        self._row(f, 5, "Âm thanh gốc phim", lambda x: ttk.Scale(x, from_=0.0, to=1.0, variable=v["orig_volume"]))
        ttk.Checkbutton(f, text="Đốt phụ đề lên video (xuất kèm .srt)", variable=v["subtitles"]).grid(row=6, column=0, columnspan=2, sticky="w")
        self._row(f, 7, "Bộ mã hoá", lambda x: ttk.Combobox(x, textvariable=v["encoder"], state="readonly", values=["libx264", "h264_nvenc"]))

        f = self._card(p, "🎧 Bộ kiểm âm trực tiếp")
        ttk.Entry(f, textvariable=v["preview_text"]).grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        ttk.Button(f, text="▶ Nghe thử giọng này", style="Blue.TButton", command=self._preview).grid(row=1, column=0, sticky="w")
        ttk.Button(f, text="■ Dừng", style="Red.TButton", command=self._stop_play).grid(row=1, column=1, sticky="w", padx=6)
        self.lbl_prev = ttk.Label(f, text="Sẵn sàng", style="Muted.TLabel")
        self.lbl_prev.grid(row=1, column=2, sticky="w")

    def _right(self, p):
        v = self.v
        top = ttk.LabelFrame(p, text="📋 Bảng phân cảnh đã chọn (sau khi cắt)", style="Card.TLabelframe", padding=6)
        top.pack(fill="both", expand=True, padx=8, pady=(6, 3))
        cols = ("stt", "pos", "dur", "kind", "score", "note")
        self.tv_shots = ttk.Treeview(top, columns=cols, show="headings", height=8)
        for c, t, w in (("stt", "STT", 40), ("pos", "Vị trí phim", 120), ("dur", "Thời lượng", 80),
                        ("kind", "Loại cảnh (motion/mặt)", 180), ("score", "Điểm", 60), ("note", "Ghi chú / Thoại nút thắt", 360)):
            self.tv_shots.heading(c, text=t)
            self.tv_shots.column(c, width=w, anchor="w" if c in ("kind", "note") else "center")
        sb = ttk.Scrollbar(top, orient="vertical", command=self.tv_shots.yview)
        self.tv_shots.configure(yscrollcommand=sb.set)
        self.tv_shots.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        gf = ttk.LabelFrame(p, text="🎭 Thể loại nội dung (Genre Bias)", style="Card.TLabelframe", padding=8)
        gf.pack(fill="x", padx=8, pady=3)
        gf.columnconfigure(0, weight=1)
        cb = ttk.Combobox(gf, textvariable=v["genre"], values=list(GENRES), state="readonly")
        cb.grid(row=0, column=0, sticky="ew")
        cb.bind("<<ComboboxSelected>>", lambda e: self._genre_info())
        self.lbl_genre = ttk.Label(gf, text="", style="Muted.TLabel", wraplength=800, justify="left")
        self.lbl_genre.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._genre_info()

        bf = ttk.Frame(p)
        bf.pack(fill="x", padx=8, pady=3)
        self.b2 = ttk.Button(bf, text="2. Viết kịch bản", style="Orange.TButton", command=lambda: self._run("2"))
        self.b3 = ttk.Button(bf, text="3. Lồng tiếng + Dựng", style="Green.TButton", command=lambda: self._run("3"))
        self.ba = ttk.Button(bf, text="⚡ Tự động 1→2→3", style="Blue.TButton", command=lambda: self._run("all"))
        self.bs = ttk.Button(bf, text="⏹ Dừng", style="Red.TButton", command=self._cancel, state="disabled")
        for i, b in enumerate((self.b2, self.b3, self.ba, self.bs)):
            b.grid(row=0, column=i, sticky="ew", padx=3)
            bf.columnconfigure(i, weight=3 if i < 3 else 1)

        sf = ttk.LabelFrame(p, text="📝 Kịch bản lời bình (chỉnh tay được)", style="Card.TLabelframe", padding=6)
        sf.pack(fill="both", expand=True, padx=8, pady=3)
        sf.columnconfigure(0, weight=1, minsize=250)
        sf.columnconfigure(1, weight=3)
        sf.rowconfigure(0, weight=1)
        cols = ("id", "pos", "dur", "words")
        self.tv_blocks = ttk.Treeview(sf, columns=cols, show="headings", height=6, selectmode="browse")
        for c, t, w in (("id", "#", 28), ("pos", "Khối", 64), ("dur", "Dài", 44), ("words", "Từ/Tối đa", 84)):
            self.tv_blocks.heading(c, text=t)
            self.tv_blocks.column(c, width=w, anchor="center")
        self.tv_blocks.grid(row=0, column=0, sticky="nsew")
        self.tv_blocks.bind("<<TreeviewSelect>>", self._on_block)
        ed = ttk.Frame(sf)
        ed.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ed.rowconfigure(0, weight=1)
        ed.columnconfigure(0, weight=1)
        self.txt = tk.Text(ed, wrap="word", height=6, font=("Segoe UI", 11), undo=True)
        self.txt.grid(row=0, column=0, sticky="nsew")
        self.txt.bind("<KeyRelease>", lambda e: self._count())
        r = ttk.Frame(ed)
        r.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.lbl_count = ttk.Label(r, text="", style="Muted.TLabel")
        self.lbl_count.pack(side="left")
        ttk.Button(r, text="✔ Áp dụng cho khối này", command=self._apply_block).pack(side="right")

        lf = ttk.LabelFrame(p, text="🧾 Nhật ký xử lý", style="Card.TLabelframe", padding=6)
        lf.pack(fill="both", padx=8, pady=(3, 8))
        self.pb = ttk.Progressbar(lf, maximum=1.0)
        self.pb.pack(fill="x")
        self.lbl_prog = ttk.Label(lf, text="", style="Muted.TLabel")
        self.lbl_prog.pack(anchor="w")
        self.log = tk.Text(lf, height=7, wrap="word", bg="#fbfbfc", font=("Consolas", 9), state="disabled")
        self.log.pack(fill="both", expand=True)
        self._log("Sẵn sàng. Chọn video → bấm 1 để cắt cảnh trước; kịch bản chỉ được viết SAU khi đã cắt.")

    # ------------------------------------------------------------ helpers
    def _genre_info(self):
        g = GENRES[self.v["genre"].get()]
        vname = VOICES[g["voice"]]["label"]
        self.lbl_genre.config(text=f"Giọng gợi ý: {vname} · tốc độ {g['rate']:+d}% · cao độ {g['pitch']:+d}Hz · nhạc: {g['music']}\n"
                                   f"Trọng số chọn cảnh → chuyển động ×{g['w_motion']} · mặt cận ×{g['w_face']} · thoại nút thắt ×{g['w_plot']} "
                                   f"· đỉnh âm thanh ×{g['w_audio']} · tối ×{g['w_dark']}\nPhong cách: {g['style']}")

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _collect(self) -> Settings:
        v = self.v
        st = self.st
        g = lambda k: v[k].get()
        st.gemini_key, st.gemini_model, st.workdir, st.title = g("gemini_key").strip(), g("gemini_model").strip(), g("workdir").strip(), g("title").strip()
        st.whisper_model, st.whisper_lang = g("whisper_model"), g("whisper_lang")
        st.video, st.srt_path = g("video").strip(), g("srt_path").strip()
        st.extraction_mode, st.target_minutes = g("extraction_mode"), float(g("target_minutes"))
        st.micro_cut, st.shot_len = g("micro_cut"), min(3.5, max(2.0, float(g("shot_len"))))
        st.highlight_priority, st.auto_skip_credits = g("highlight_priority"), g("auto_skip_credits")
        st.skip_head_sec, st.skip_tail_sec = float(g("skip_head_sec")), float(g("skip_tail_sec"))
        st.outro_montage, st.outro_shots = g("outro_montage"), int(g("outro_shots"))
        st.aspect = ASPECTS[g("aspect")]
        st.grade = "auto" if g("grade") == "Theo thể loại" else GRADES[g("grade")]
        st.zoom, st.speed_ramp = g("zoom"), 1.08 if g("ramp") else 1.0
        st.crop_bottom, st.vignette, st.grain, st.safety_filter = g("crop_bottom"), g("vignette"), g("grain"), g("safety_filter")
        st.genre = g("genre")
        vl = g("voice")
        st.voice = "auto" if vl == "Theo thể loại" else next(k for k, d in VOICES.items() if d["label"] == vl)
        st.speed = float(g("speed").rstrip("x"))
        st.music_path, st.music_volume, st.orig_volume = g("music_path").strip(), float(g("music_volume")), float(g("orig_volume"))
        st.subtitles, st.encoder = g("subtitles"), g("encoder")
        return st

    def _save(self):
        save_settings(self._collect())
        messagebox.showinfo("ReviewCut", f"Đã lưu cấu hình (kể cả Gemini key, dạng văn bản thường) tại:\n{Path.home() / '.reviewcut' / 'config.json'}")

    # ------------------------------------------------------------ chạy nền
    def _set_busy(self, b):
        self.busy = b
        for w in (self.b1, self.b2, self.b3, self.ba):
            w.configure(state="disabled" if b else "normal")
        self.bs.configure(state="normal" if b else "disabled")

    def _run(self, which):
        if self.busy:
            return
        try:
            st = self._collect()
            save_settings(st)
            if not st.video or not Path(st.video).is_file():
                raise RuntimeError("Hãy chọn video gốc trước.")
            find_tool("ffmpeg")
            proj = Project(st)
            if which in ("2", "3"):
                if not proj.load():
                    raise RuntimeError("Chưa có kết quả bước 1 cho video này – hãy chạy bước 1 trước.")
                if self.project and self.project.st.video == st.video and self.project.blocks:
                    proj.blocks = self.project.blocks          # giữ bản chỉnh tay đang hiển thị
            self.project = proj
        except Exception as ex:  # noqa: BLE001
            messagebox.showerror("ReviewCut", str(ex))
            return
        self.job = Job(log=lambda m: self.q.put(("log", m)), progress=lambda f, m: self.q.put(("prog", (f, m))))
        self._set_busy(True)

        def work():
            try:
                if which == "1":
                    proj.stage1_analyze_and_cut(self.job)
                elif which == "2":
                    proj.stage2_script(self.job)
                elif which == "3":
                    out = proj.stage3_render(self.job)
                    self.q.put(("done_render", out))
                else:
                    proj.stage1_analyze_and_cut(self.job)
                    self.q.put(("refresh", None))
                    proj.stage2_script(self.job)
                    self.q.put(("refresh", None))
                    self.q.put(("done_render", proj.stage3_render(self.job)))
            except Cancelled:
                self.q.put(("log", "⏹ Đã dừng."))
            except Exception as ex:  # noqa: BLE001
                self.q.put(("error", str(ex)))
            finally:
                self.q.put(("refresh", None))
                self.q.put(("idle", None))
        threading.Thread(target=work, daemon=True).start()

    def _cancel(self):
        if self.job:
            self.job.cancel()
            self._log("Đang dừng…")

    def _poll(self):
        try:
            while True:
                k, d = self.q.get_nowait()
                if k == "log":
                    self._log(d)
                elif k == "prog":
                    self.pb["value"] = d[0]
                    self.lbl_prog.config(text=d[1])
                elif k == "refresh":
                    self._refresh()
                elif k == "idle":
                    self._set_busy(False)
                elif k == "error":
                    self._log("✖ " + d)
                    messagebox.showerror("ReviewCut", d)
                elif k == "done_render":
                    if messagebox.askyesno("ReviewCut", f"Dựng xong!\n{d}\n\nMở thư mục chứa video?"):
                        self._open_dir(Path(d).parent)
        except queue.Empty:
            pass
        self.after(120, self._poll)

    def _open_dir(self, p):
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(p))          # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------ bảng
    def _refresh(self):
        pr = self.project
        self.tv_shots.delete(*self.tv_shots.get_children())
        self.tv_blocks.delete(*self.tv_blocks.get_children())
        if not pr:
            return
        n = 0
        for b in pr.blocks:
            for s in b.shots:
                n += 1
                note = s.caption
                if s.must and s.tags:
                    note = f"★ {', '.join(s.tags)} · " + note
                if s.role == "outro":
                    note = "OUTRO · " + note
                self.tv_shots.insert("", "end", values=(n, f"{fmt_time(s.start)}–{fmt_time(s.end)}", f"{s.dur:.1f}s",
                                                        s.kind, f"{s.score:.2f}", note))
            words = len(b.text.split())
            lim = max_words(b, self.st) if b.role == "main" else 0
            self.tv_blocks.insert("", "end", iid=str(b.id), values=(b.id + 1, "OUTRO" if b.role == "outro" else f"{fmt_time(b.shots[0].start)}",
                                                                  f"{b.dur:.0f}s", f"{words}/{lim}" if lim else "—"))

    def _block(self):
        sel = self.tv_blocks.selection()
        if not sel or not self.project:
            return None
        return next((b for b in self.project.blocks if str(b.id) == sel[0]), None)

    def _on_block(self, _=None):
        b = self._block()
        self.txt.delete("1.0", "end")
        if b:
            self.txt.insert("1.0", b.text)
            self.txt.configure(state="disabled" if b.role == "outro" else "normal")
        self._count()

    def _count(self):
        b = self._block()
        if not b:
            return
        n = len(self.txt.get("1.0", "end").split())
        lim = max_words(b, self.st) if b.role == "main" else 0
        self.lbl_count.config(text=f"{n} từ / tối đa ≈{lim}" + ("  ⚠ dài hơn hình: giọng sẽ tăng tốc / hình chậm lại" if lim and n > lim else ""))

    def _apply_block(self):
        b = self._block()
        if not b or b.role == "outro":
            return
        b.text = self.txt.get("1.0", "end").strip()
        self.project.save()
        self._refresh()
        self.tv_blocks.selection_set(str(b.id))

    # ------------------------------------------------------------ nghe thử
    def _preview(self):
        self._stop_play()
        st = self._collect()
        self.lbl_prev.config(text="Đang tạo giọng…")
        text = self.v["preview_text"].get()

        def work():
            try:
                from reviewcut import tts
                mp3 = os.path.join(tempfile.gettempdir(), "reviewcut_preview.mp3")
                tts.preview(text, st, mp3)
                self.after(0, lambda: self._play(mp3))
            except Exception as ex:  # noqa: BLE001
                self.after(0, lambda: self.lbl_prev.config(text="Lỗi: " + str(ex)[:80]))
        threading.Thread(target=work, daemon=True).start()

    def _play(self, mp3):
        try:
            if sys.platform.startswith("win"):
                import winsound
                wav = mp3.replace(".mp3", ".wav")
                subprocess.run([find_tool("ffmpeg"), "-y", "-v", "error", "-i", mp3, wav], check=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                winsound.PlaySound(wav, winsound.SND_FILENAME | winsound.SND_ASYNC)
            elif sys.platform == "darwin":
                self.play_proc = subprocess.Popen(["afplay", mp3])
            else:
                ffplay = shutil.which("ffplay")
                cmd = [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", mp3] if ffplay else ["xdg-open", mp3]
                self.play_proc = subprocess.Popen(cmd)
            self.lbl_prev.config(text="Đang phát…")
        except Exception as ex:  # noqa: BLE001
            self.lbl_prev.config(text="Không phát được: " + str(ex)[:60])

    def _stop_play(self):
        if sys.platform.startswith("win"):
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:  # noqa: BLE001
                pass
        if self.play_proc and self.play_proc.poll() is None:
            self.play_proc.kill()
        self.lbl_prev.config(text="Sẵn sàng")

    def _close(self):
        self._stop_play()
        if self.job:
            self.job.cancel()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
