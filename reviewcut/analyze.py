"""Phân tích video: cắt cảnh, motion score, khuôn mặt cận cảnh, đỉnh âm thanh, cảnh đen/credits."""
from __future__ import annotations

import re
import subprocess
import threading
import wave
from dataclasses import asdict, dataclass, field
from typing import Optional

import cv2
import numpy as np

from .media import NO_WINDOW, Job, extract_audio, find_tool, probe

SAMPLE_FPS = 4          # số mẫu hình/giây
FW, FH = 320, 180       # kích thước khung phân tích


@dataclass
class Scene:
    id: int
    start: float
    end: float
    motion: float = 0.0        # 0..1 (xếp hạng trong cả phim)
    motion_raw: float = 0.0
    face: float = 0.0          # tỉ lệ mẫu có mặt cận cảnh 0..1
    face_size: float = 0.0     # diện tích mặt lớn nhất / khung
    audio: float = 0.0         # 0..1 đỉnh âm thanh (xếp hạng)
    dark: float = 0.0          # tỉ lệ điểm ảnh tối
    brightness: float = 0.0
    blank: bool = False        # cảnh đen / phẳng
    credits: bool = False      # chữ chạy cuối phim
    kind: str = ""             # nhãn: hành động / nội tâm / đối thoại / tối / thường

    @property
    def dur(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Timeline:
    """Chuỗi đặc trưng theo mẫu (SAMPLE_FPS) – dùng để chấm điểm từng cửa sổ."""
    fps: float
    motion: np.ndarray            # xếp hạng 0..1
    face: np.ndarray              # 0/1 (mặt cận cảnh)
    dark: np.ndarray
    audio: np.ndarray             # xếp hạng 0..1, cùng nhịp mẫu
    duration: float

    def span(self, arr: np.ndarray, t0: float, t1: float) -> np.ndarray:
        i0, i1 = int(t0 * self.fps), max(int(t0 * self.fps) + 1, int(np.ceil(t1 * self.fps)))
        return arr[max(0, i0):min(len(arr), i1)]


def rank_norm(x: np.ndarray) -> np.ndarray:
    """Xếp hạng phần trăm 0..1 (bền với giá trị ngoại lai)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    order = x.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(x.size)
    # các giá trị bằng nhau nhận cùng hạng trung bình
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return ranks / max(1, x.size - 1)


def audio_energy(wav_path: str, hop: float = 1.0 / SAMPLE_FPS) -> np.ndarray:
    """RMS (dB) theo bước `hop` giây, đọc streaming để không tốn RAM."""
    out = []
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        step = max(1, int(sr * hop))
        while True:
            buf = w.readframes(step * 400)
            if not buf:
                break
            a = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
            n = len(a) // step
            if n == 0:
                if len(a):
                    out.append(20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-6))
                break
            blocks = a[: n * step].reshape(n, step)
            out.extend(20 * np.log10(np.sqrt((blocks ** 2).mean(axis=1)) + 1e-6))
            if len(a) > n * step:
                tail = a[n * step:]
                out.append(20 * np.log10(np.sqrt(np.mean(tail ** 2)) + 1e-6))
    return np.asarray(out, dtype=np.float32)


MIN_SCENE_GAP = 0.5      # giây: hai điểm cắt gần hơn mức này coi là một (chớp sáng, nổ)


class SceneCutScanner:
    """Chạy ffmpeg bộ lọc `scene` song song với luồng phân tích chính → điểm cắt CHÍNH XÁC tới từng khung hình
    (bản cũ chỉ lấy mẫu 4 khung/giây nên lệch tới 0.25 s và làm shot dính vài khung của cảnh trước/sau)."""

    def __init__(self, video: str):
        self.times: list[float] = []
        self.scores: list[float] = []
        self.ok = False
        cmd = [find_tool("ffmpeg"), "-v", "error", "-i", video, "-an", "-vf",
               "scale=160:90:flags=fast_bilinear,select='gte(scene,0)',metadata=print:key=lavfi.scene_score:file=-",
               "-f", "null", "-"]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        except OSError:
            self.proc = None
            return
        self.th = threading.Thread(target=self._read, daemon=True)
        self.th.start()

    def _read(self) -> None:
        t = None
        for raw in self.proc.stdout:                     # type: ignore[union-attr]
            line = raw.decode("utf-8", "replace")
            m = re.search(r"pts_time:([\d.]+)", line)
            if m:
                t = float(m.group(1))
                continue
            m = re.search(r"lavfi\.scene_score=([\d.eE+-]+)", line)
            if m and t is not None:
                self.times.append(t)
                self.scores.append(float(m.group(1)))
                t = None

    def result(self, min_gap: float = MIN_SCENE_GAP) -> Optional[list[float]]:
        """Danh sách thời điểm cắt (giây) hoặc None nếu bộ lọc không chạy được."""
        if self.proc is None:
            return None
        try:
            self.th.join(timeout=None)
            self.proc.wait(timeout=30)
        except Exception:                                # noqa: BLE001
            self.proc.kill()
        if len(self.times) < 8:
            return None
        sc = np.array(self.scores)
        thr = float(min(0.45, max(0.28, sc.mean() + 8 * sc.std())))
        cuts: list[tuple[float, float]] = []
        for t, v in zip(self.times, self.scores):
            if v < thr or t < 0.3:
                continue
            if cuts and t - cuts[-1][0] < min_gap:        # quá gần → giữ điểm cắt mạnh hơn
                if v > cuts[-1][1]:
                    cuts[-1] = (t, v)
                continue
            cuts.append((t, v))
        self.thr = thr
        return [t for t, _ in cuts]

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()


def refine_cut(video: str, t_lo: float, t_hi: float, fps: float) -> Optional[float]:
    """Tìm khung có thay đổi lớn nhất trong [t_lo, t_hi] (đọc đủ số khung/giây) → thời điểm cắt chính xác."""
    w, h = 160, 90
    cmd = [find_tool("ffmpeg"), "-v", "error", "-ss", f"{max(0.0, t_lo):.3f}", "-i", video, "-t", f"{t_hi - t_lo:.3f}",
           "-an", "-vf", f"scale={w}:{h}:flags=fast_bilinear,format=gray", "-f", "rawvideo", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=60, creationflags=NO_WINDOW).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    nfr = len(out) // (w * h)
    if nfr < 3:
        return None
    arr = np.frombuffer(out[: nfr * w * h], dtype=np.uint8).reshape(nfr, h, w).astype(np.int16)
    d = np.abs(np.diff(arr, axis=0)).mean(axis=(1, 2))
    i = int(d.argmax())
    return max(0.0, t_lo) + (i + 1) / max(1.0, fps) if d[i] >= 8.0 else None


def _face_detector():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    det = cv2.CascadeClassifier(path)
    return det if not det.empty() else None


def analyze_video(video: str, workdir: str, job: Job, skip_head: float = 0.0, skip_tail: float = 0.0,
                  auto_credits: bool = True, wav_path: Optional[str] = None):
    """Trả về (scenes, timeline, info). Mất khoảng 1/10–1/5 thời lượng phim."""
    info = probe(video)
    dur = info["duration"]
    job.log(f"Video {info['width']}x{info['height']} · {dur/60:.1f} phút · {info['fps']:.1f} fps")

    # ---- âm thanh
    energy = np.zeros(int(dur * SAMPLE_FPS) + 2, dtype=np.float32)
    if info["has_audio"]:
        wav_path = wav_path or f"{workdir}/audio16k.wav"
        job.progress(0.02, "Tách âm thanh…")
        extract_audio(video, wav_path, job)
        e = audio_energy(wav_path)
        energy[: min(len(e), len(energy))] = e[: len(energy)]
    job.check()

    # ---- đọc khung hình qua ffmpeg (nhanh hơn seek bằng OpenCV)
    cmd = [find_tool("ffmpeg"), "-v", "error", "-i", video, "-an", "-vf",
           f"fps={SAMPLE_FPS},scale={FW}:{FH}:flags=fast_bilinear", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    scanner = SceneCutScanner(video)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
    fsize = FW * FH * 3
    det = _face_detector()
    hists, diffs, bright, darkf, brightf, facesz = [], [], [], [], [], []
    prev_gray, prev_hist, n = None, None, 0
    expected = max(1, int(dur * SAMPLE_FPS))
    try:
        while True:
            buf = proc.stdout.read(fsize)
            if len(buf) < fsize:
                break
            if n % 40 == 0:
                job.check()
                job.progress(0.05 + 0.80 * min(1.0, n / expected), "Phân tích hình ảnh…")
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(FH, FW, 3)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(frame, (80, 45), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
            h = cv2.normalize(h, None, 1.0, 0.0, cv2.NORM_L1).flatten()
            hdist = 0.0 if prev_hist is None else float(0.5 * np.abs(h - prev_hist).sum())  # 0..1
            mdiff = 0.0 if prev_gray is None else float(cv2.absdiff(gray, prev_gray).mean() / 255.0)
            hists.append(hdist)
            diffs.append(mdiff)
            bright.append(float(gray.mean()))
            darkf.append(float((gray < 50).mean()))
            brightf.append(float((gray > 200).mean()))
            fs = 0.0
            if det is not None and n % 2 == 0:
                eq = cv2.equalizeHist(gray)
                faces = det.detectMultiScale(eq, scaleFactor=1.15, minNeighbors=5, minSize=(22, 22))
                if len(faces):
                    fs = max(w * h_ for (_, _, w, h_) in faces) / float(FW * FH)
            elif facesz:
                fs = facesz[-1] * 0.9           # giữ gần đúng ở mẫu bỏ qua
            facesz.append(fs)
            prev_gray, prev_hist, n = gray, h, n + 1
    except BaseException:
        scanner.stop()
        raise
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    if n < 4:
        scanner.stop()
        raise RuntimeError("Không đọc được khung hình nào từ video.")

    hists, diffs = np.array(hists), np.array(diffs)
    bright, darkf, brightf, facesz = map(np.array, (bright, darkf, brightf, facesz))
    energy = energy[:n] if len(energy) >= n else np.pad(energy, (0, n - len(energy)), constant_values=energy.min() if len(energy) else -80)

    # ---- phát hiện điểm cắt cảnh: (1) bộ lọc scene của ffmpeg (chính xác từng khung)
    #      (2) histogram 4 fps; điểm cắt chỉ (2) thấy được sẽ được AFFINE lại tới từng khung → hợp nhất
    end_t = n / SAMPLE_FPS
    exact = scanner.result() or []
    thr = max(0.30, float(hists.mean() + 3.5 * hists.std()))
    hist_cuts, last_i = [], 0
    for i in range(1, n):
        if hists[i] > thr and (i - last_i) >= max(2, int(SAMPLE_FPS * MIN_SCENE_GAP)):
            hist_cuts.append(i)
            last_i = i
    extra = []
    for i in hist_cuts:
        lo, hi = (i - 1) / SAMPLE_FPS, i / SAMPLE_FPS
        if any(lo - 0.3 <= t <= hi + 0.3 for t in exact):
            continue                                    # đã có điểm cắt chính xác gần đó
        t = refine_cut(video, lo - 0.08, hi + 0.08, info["fps"])
        extra.append(t if t is not None else hi)
    merged: list[float] = []
    for t in sorted(exact + extra):
        if 0.5 < t < end_t - 0.3 and (not merged or t - merged[-1] >= MIN_SCENE_GAP):
            merged.append(t)
    cut_times = [0.0] + merged + [end_t]
    job.log(f"Điểm cắt cảnh: {len(merged)} (ffmpeg scene {len(exact)} + histogram-affine {len(extra)}).")
    cut_idx = {min(n - 1, int(t * SAMPLE_FPS)) for t in cut_times[1:-1]}
    # chuyển động: bỏ mẫu nằm ngay tại điểm cắt (nhảy cảnh ≠ chuyển động)
    motion = diffs.copy()
    for i in cut_idx:
        for j in (i, i + 1):
            if 0 < j < n:
                motion[j] = motion[i - 1]
    motion_rank = rank_norm(motion)
    audio_rank = rank_norm(energy)

    # ---- credits: cuối phim, nền tối + chữ sáng thưa
    credits_from = None
    if auto_credits and n > SAMPLE_FPS * 120:
        is_cr = (darkf > 0.86) & (brightf > 0.002) & (brightf < 0.09)
        tail0 = int(n * 0.75)
        csum = np.cumsum(is_cr[::-1])[::-1]               # số mẫu credits từ i tới cuối
        for i in range(tail0, n):
            if csum[i] / max(1, n - i) > 0.82 and (n - i) > SAMPLE_FPS * 20:
                credits_from = i / SAMPLE_FPS
                break
        if credits_from:
            job.log(f"Phát hiện credits từ {credits_from/60:.1f} phút – sẽ bỏ qua.")

    head, tail_t = skip_head, dur - skip_tail
    scenes: list[Scene] = []
    for k in range(len(cut_times) - 1):
        t0, t1 = cut_times[k], cut_times[k + 1]
        a = min(n - 1, int(round(t0 * SAMPLE_FPS)))
        b = min(n, max(a + 1, int(round(t1 * SAMPLE_FPS))))
        sl = slice(a, b)
        m = motion[sl]
        fsz = facesz[sl]
        sc = Scene(id=len(scenes), start=round(t0, 3), end=round(t1, 3),
                   motion_raw=float(np.percentile(m, 75)) if len(m) else 0.0,
                   motion=float(np.percentile(motion_rank[sl], 75)) if len(m) else 0.0,
                   face=float((fsz > 0.035).mean()) if len(fsz) else 0.0,
                   face_size=float(fsz.max()) if len(fsz) else 0.0,
                   audio=float(np.percentile(audio_rank[sl], 90)) if len(m) else 0.0,
                   dark=float(darkf[sl].mean()), brightness=float(bright[sl].mean()))
        sc.blank = sc.brightness < 8 or (float(np.std(bright[sl])) < 1.0 and sc.brightness < 25)   # đen hẳn / màn hình phẳng
        sc.credits = bool(credits_from and t0 >= credits_from - 1)
        if t1 <= head or t0 >= tail_t:
            sc.blank = True
        sc.kind = classify(sc)
        scenes.append(sc)

    tl = Timeline(fps=SAMPLE_FPS, motion=motion_rank, face=(facesz > 0.035).astype(float),
                  dark=darkf, audio=audio_rank, duration=dur)
    job.progress(0.95, "Hoàn tất phân tích hình ảnh")
    job.log(f"{len(scenes)} phân cảnh · ngưỡng cắt {thr:.2f} · "
            f"{sum(1 for s in scenes if s.blank or s.credits)} cảnh bị loại (đen/credits/ngoài vùng)")
    return scenes, tl, info


def classify(s: Scene) -> str:
    """Gắn nhãn theo chỉ số vật lý: motion cao → hành động; mặt cận + ít động → nội tâm/đối thoại."""
    if s.blank or s.credits:
        return "bỏ qua"
    if s.motion >= 0.78:
        return "hành động"
    if s.face >= 0.4 and s.face_size >= 0.06 and s.motion < 0.6:
        return "nội tâm / đối thoại cận mặt"
    if s.dark >= 0.55:
        return "tối / căng thẳng"
    if s.face >= 0.25:
        return "đối thoại"
    return "thường"
