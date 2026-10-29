"""Whisper: bóc tách thoại gốc + mốc thời gian. Hỗ trợ nạp .srt/.vtt có sẵn."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .media import Job


@dataclass
class Line:
    start: float
    end: float
    text: str
    plot: float = 0.0          # điểm "nút thắt" 0..1 (do plot.py gán)
    tags: tuple = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d


_TS = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")
_TAG = re.compile(r"<[^>]+>|\{\\[^}]*\}")


def _sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def load_subtitle(path: str) -> list[Line]:
    """Đọc .srt hoặc .vtt (tự nhận encoding utf-8 / utf-8-sig / cp1258 / latin-1)."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1258", "latin-1"):
        try:
            txt = raw.decode(enc)
            break
        except UnicodeError:
            continue
    lines: list[Line] = []
    cur: Optional[Line] = None
    for ln in txt.replace("\r", "").split("\n"):
        m = _TS.search(ln)
        if m:
            if cur and cur.text.strip():
                lines.append(cur)
            cur = Line(_sec(*m.groups()[:4]), _sec(*m.groups()[4:]), "")
        elif cur is not None:
            if not ln.strip():
                if cur.text.strip():
                    lines.append(cur)
                cur = None
            elif not ln.strip().isdigit():
                cur.text += (" " if cur.text else "") + _TAG.sub("", ln).strip()
    if cur and cur.text.strip():
        lines.append(cur)
    return lines


def transcribe(wav: str, model: str, lang: str, device: str, job: Job, cache: Optional[str] = None) -> list[Line]:
    """Chạy faster-whisper (ưu tiên) hoặc openai-whisper. Kết quả được cache theo (model, lang)."""
    if cache and Path(cache).exists():
        try:
            d = json.loads(Path(cache).read_text(encoding="utf-8"))
            if d.get("model") == model and d.get("lang") == lang:
                job.log(f"Dùng lại thoại Whisper đã lưu ({len(d['lines'])} câu).")
                return [Line(**{**x, "tags": tuple(x.get("tags", ()))}) for x in d["lines"]]
        except Exception:
            pass
    lines: list[Line] = []
    language = None if lang in ("auto", "", None) else lang
    try:
        from faster_whisper import WhisperModel
        dev = device
        if dev == "auto":
            try:
                import ctranslate2
                dev = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                dev = "cpu"
        compute = "float16" if dev == "cuda" else "int8"
        job.log(f"Whisper (faster-whisper) model={model} device={dev} – lần đầu sẽ tải model…")
        wm = WhisperModel(model, device=dev, compute_type=compute)
        segs, info = wm.transcribe(wav, language=language, vad_filter=True, beam_size=1,
                                   condition_on_previous_text=False)
        total = max(1.0, info.duration)
        job.log(f"Ngôn ngữ phát hiện: {info.language} ({info.language_probability:.0%})")
        for s in segs:
            job.check()
            if s.text.strip():
                lines.append(Line(round(s.start, 2), round(s.end, 2), s.text.strip()))
            job.progress(0.1 + 0.8 * min(1.0, s.end / total), f"Whisper nghe thoại… {s.end/60:.1f}/{total/60:.1f} phút")
    except ImportError:
        try:
            import whisper  # openai-whisper
        except ImportError:
            raise RuntimeError("Chưa cài Whisper. Chạy: pip install faster-whisper  (hoặc nạp file .srt có sẵn).")
        job.log(f"Whisper (openai-whisper) model={model}…")
        wm = whisper.load_model(model)
        res = wm.transcribe(wav, language=language, verbose=False)
        for s in res["segments"]:
            if s["text"].strip():
                lines.append(Line(round(s["start"], 2), round(s["end"], 2), s["text"].strip()))
    if cache:
        Path(cache).write_text(json.dumps({"model": model, "lang": lang, "lines": [l.to_dict() for l in lines]},
                                          ensure_ascii=False), encoding="utf-8")
    return lines
