"""Dựng video: hiệu ứng hình, đồng bộ shot ↔ giọng đọc, nhạc nền có ducking, phụ đề, xuất file."""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import Settings
from .media import Job, duration_of, find_tool, has_encoder, has_filter, probe, run
from .select import Block

LEAD = 0.15           # khoảng lặng trước khi giọng vào
TAIL = 0.25
MAX_VOICE_SPEEDUP = 1.18
MAX_STRETCH = 1.30    # làm chậm shot tối đa để chờ giọng đọc; quá nữa thì đứng hình đuôi


def canvas(aspect: str) -> tuple[int, int]:
    return {"16:9": (1920, 1080), "2.39": (1920, 1080), "9:16blur": (1080, 1920), "9:16crop": (1080, 1920),
            "1:1": (1080, 1080)}.get(aspect, (1920, 1080))


@dataclass
class Timing:
    block: Block
    voice_dur: float        # độ dài giọng sau khi tăng tốc (nếu cần)
    atempo: float
    eff_sum: float          # tổng độ dài shot sau speed-ramp, chưa kéo giãn
    total: float            # độ dài khối cuối cùng
    stretch: float
    freeze: float


def plan_timing(blocks: list[Block], raw_voice: dict[int, float], st: Settings) -> list[Timing]:
    ramp = max(1.0, st.speed_ramp)
    out = []
    for b in blocks:
        eff = sum(s.dur for s in b.shots) / ramp
        v, at = 0.0, 1.0
        if b.role == "main" and b.text.strip() and raw_voice.get(b.id, 0) > 0:
            v = raw_voice[b.id]
            if v + LEAD + TAIL > eff:
                at = min(MAX_VOICE_SPEEDUP, (v) / max(0.5, eff * MAX_STRETCH - LEAD - TAIL))
                at = max(1.0, at)
            v /= at
        total = max(eff, v + LEAD + TAIL if v else eff)
        stretch = min(total / eff, MAX_STRETCH) if eff else 1.0
        freeze = max(0.0, total - eff * stretch)
        out.append(Timing(b, v, at, eff, total, stretch, freeze))
    return out


# ------------------------------------------------------------------ bộ lọc hình
def _grade(g: str) -> str:
    return {
        "teal_orange": "colorbalance=rs=-0.08:gs=-0.02:bs=0.10:rm=0.05:bm=-0.04:rh=0.10:gh=0.02:bh=-0.10,"
                       "eq=contrast=1.08:saturation=1.12",
        "cold": "colorbalance=rs=-0.04:bs=0.08:bm=0.05,eq=contrast=1.10:saturation=0.88:brightness=-0.02",
        "warm": "colorbalance=rs=0.06:bs=-0.06:rm=0.04:bm=-0.03,eq=contrast=1.04:saturation=1.08",
    }.get(g, "")


def video_filter(st: Settings, out_dur: float, factor: float, freeze: float, zoom_dir: int) -> str:
    """Trả về filtergraph [0:v] -> [v]."""
    W, H = canvas(st.aspect)
    chain: list[str] = []
    if st.crop_bottom:
        chain.append("crop=iw:trunc(ih*0.90/2)*2:0:0")
    chain.append(f"setpts={factor:.5f}*PTS")
    pad = ""
    if st.aspect == "2.39":
        cw, ch = 1920, 804
        fit = f"scale={cw}:{ch}:force_original_aspect_ratio=increase,crop={cw}:{ch}"
        pad = f"pad={W}:{H}:0:{(H - ch) // 2}:black"
    elif st.aspect == "9:16crop":
        cw, ch = W, H
        fit = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"
    elif st.aspect == "1:1":
        cw, ch = W, H
        fit = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"
    elif st.aspect == "9:16blur":
        cw, ch = W, H
        fit = None
    else:
        cw, ch = W, H
        fit = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"

    g = ",".join(chain)
    if fit is None:
        g += (f",split=2[bg][fg];[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
              f"boxblur=25:3[b];[fg]scale={W}:-2[f];[b][f]overlay=(W-w)/2:(H-h)/2")
    else:
        g += "," + fit

    post: list[str] = []
    if st.zoom and out_dur > 0.8:
        z = 0.05
        tot_frames = max(1, int(st.fps * out_dur))
        expr = f"max(1,(1+{z}*n/{tot_frames}))" if zoom_dir >= 0 else f"max(1,(1+{z}*(1-n/{tot_frames})))"
        post.append(f"scale=w='trunc({cw}*{expr}/2)*2':h='trunc({ch}*{expr}/2)*2':eval=frame,crop={cw}:{ch}")

    if pad:
        post.append(pad)
    gr = _grade(st.resolved_grade())
    if gr:
        post.append(gr)
    if st.vignette:
        post.append("vignette=PI/5")
    if st.grain:
        post.append("noise=alls=7:allf=t")
    post.append(f"fps={st.fps}")
    if freeze > 0.02:
        post.append(f"tpad=stop_mode=clone:stop_duration={freeze:.3f}")
    post += ["format=yuv420p", "setsar=1"]
    return f"[0:v]{g},{','.join(post)}[v]"


def _enc(st: Settings, crf: int) -> list[str]:
    if st.encoder == "h264_nvenc" and has_encoder("h264_nvenc"):
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(crf + 2), "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p"]


def render_shot(video: str, shot, t: Timing, k: int, last_in_block: bool, st: Settings, out: str,
                has_audio: bool, job: Job) -> None:
    ramp = max(1.0, st.speed_ramp)
    factor = t.stretch / ramp
    out_dur = shot.dur * factor + (t.freeze if last_in_block else 0.0)
    vf = video_filter(st, shot.dur * factor, factor, t.freeze if last_in_block else 0.0, 1 if k % 2 == 0 else -1)
    atempo = 1.0 / factor
    ff = find_tool("ffmpeg")
    cmd = [ff, "-y", "-v", "error", "-ss", f"{shot.start:.3f}", "-t", f"{shot.dur:.3f}", "-avoid_negative_ts", "make_zero", "-i", video]
    if has_audio:
        af = f"[0:a:0]atempo={min(2.0, max(0.5, atempo)):.4f},aresample=48000,aformat=channel_layouts=stereo,apad[a]"
    else:
        cmd += ["-f", "lavfi", "-t", f"{out_dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        af = "[1:a]anull[a]"
    cmd += ["-filter_complex", f"{vf};{af}", "-map", "[v]", "-map", "[a]", "-t", f"{out_dur:.3f}"]
    cmd += _enc(st, 16) + ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", out]
    run(cmd, job)


# ------------------------------------------------------------------ phụ đề ASS
def split_subtitle(text: str, max_chars: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…,;:])\s+", text)
    out: list[str] = []
    for p in parts:
        words, cur = p.split(), ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > max_chars:
                out.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        if cur:
            out.append(cur)
    merged: list[str] = []
    for c in out:
        if merged and len(c) < 10 and len(merged[-1]) + 1 + len(c) <= max_chars:
            merged[-1] += " " + c
        else:
            merged.append(c)
    return merged


def _ass_time(t: float) -> str:
    cs = int(round(t * 100))
    h, r = divmod(cs, 360000)
    m, r = divmod(r, 6000)
    s, c = divmod(r, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def build_ass(timings: list[Timing], st: Settings) -> tuple[str, list[tuple[float, float, str]]]:
    W, H = canvas(st.aspect)
    fs = max(18, int(H * st.sub_size * (0.85 if W < H else 1.0)))
    mv = int(H * (0.14 if W < H else 0.07))
    max_chars = 26 if W < H else 44
    head = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {W}\nPlayResY: {H}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,"
            "Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
            f"Style: Default,{st.sub_font},{fs},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,"
            f"{max(2, fs // 14)},1,2,60,60,{mv},1\n\n"
            "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    ev, cues, t0 = [], [], 0.0
    for t in timings:
        if t.voice_dur > 0:
            chunks = split_subtitle(t.block.text, max_chars)
            tot = sum(len(c) for c in chunks) or 1
            cur = t0 + LEAD
            for c in chunks:
                d = t.voice_dur * len(c) / tot
                a, b = cur, cur + d
                cues.append((a, b, c))
                esc = c.replace("\\", "").replace("{", "(").replace("}", ")")
                ev.append(f"Dialogue: 0,{_ass_time(a)},{_ass_time(b)},Default,,0,0,0,,{esc}")
                cur = b
        t0 += t.total
    return head + "\n".join(ev) + "\n", cues


def write_srt(cues: list[tuple[float, float, str]], path: str) -> None:
    def ts(x):
        ms = int(round(x * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    Path(path).write_text("\n".join(f"{i}\n{ts(a)} --> {ts(b)}\n{c}\n" for i, (a, b, c) in enumerate(cues, 1)),
                          encoding="utf-8")


# ------------------------------------------------------------------ nhạc
def pick_music(path: str, seed: int = 0) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return str(p)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"))
        if files:
            return str(random.Random(seed).choice(files))
    return None


# ------------------------------------------------------------------ điều phối
def render_all(video: str, blocks: list[Block], st: Settings, workdir: str, out_path: str, job: Job,
               voice_files: dict[int, str]) -> str:
    ff = find_tool("ffmpeg")
    wd = Path(workdir)
    segd = wd / "segments"
    segd.mkdir(exist_ok=True)
    info = probe(video)

    raw = {bid: duration_of(f) for bid, f in voice_files.items() if Path(f).exists()}
    timings = plan_timing(blocks, raw, st)
    total = sum(t.total for t in timings)
    over = [t for t in timings if t.freeze > 0.5]
    job.log(f"Tổng thời lượng dự kiến {total/60:.1f} phút · {len(over)} khối lời bình dài hơn hình (đã tự chậm/đứng đuôi).")

    # 1) từng shot
    sig = hashlib.sha1(repr((st.aspect, st.resolved_grade(), st.zoom, st.speed_ramp, st.crop_bottom, st.vignette,
                             st.grain, st.fps, st.encoder, Path(video).stat().st_size)).encode()).hexdigest()[:8]
    segs: list[str] = []
    n_all = sum(len(t.block.shots) for t in timings)
    done = 0
    k = 0
    for t in timings:
        for j, sh in enumerate(t.block.shots):
            job.check()
            last = j == len(t.block.shots) - 1
            h = hashlib.sha1(f"{sig}|{sh.start:.3f}|{sh.end:.3f}|{t.stretch:.4f}|{t.freeze if last else 0:.3f}|{k % 2}".encode()).hexdigest()[:12]
            f = str(segd / f"s_{h}.mp4")
            if not Path(f).exists():
                render_shot(video, sh, t, k, last, st, f, info["has_audio"], job)
            segs.append(f)
            k += 1
            done += 1
            job.progress(0.05 + 0.60 * done / n_all, f"Dựng shot {done}/{n_all}")

    # Chuyển đổi nhanh