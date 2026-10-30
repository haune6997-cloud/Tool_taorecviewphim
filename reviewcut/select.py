"""Chọn cảnh nòng cốt: chấm điểm cửa sổ theo Genre Bias, giữ câu thoại nút thắt, micro-cut, chia khối lời bình."""
from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from .analyze import Scene, Timeline
from .config import Settings
from .transcribe import Line

EDGE = 0.08            # co vào trong ranh giới cảnh để không dính nhịp cắt
OUTRO_LEN = 1.4


@dataclass
class Shot:
    start: float
    end: float
    scene_id: int
    score: float = 0.0
    motion: float = 0.0
    face: float = 0.0
    kind: str = ""
    tags: list = field(default_factory=list)       # nhãn nút thắt trong shot
    must: bool = False                              # chứa câu thoại nút thắt
    role: str = "main"                              # main | outro
    caption: str = ""
    caption_ai: bool = False                        # True = do AI nhìn ảnh mô tả (sự thật); False = suy luận

    @property
    def dur(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Block:
    id: int
    shots: list
    text: str = ""
    role: str = "main"
    dialogue: list = field(default_factory=list)    # [{t, text, plot}]

    @property
    def dur(self) -> float:
        return sum(s.dur for s in self.shots)

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "role": self.role, "dialogue": self.dialogue,
                "shots": [s.to_dict() for s in self.shots]}


class LineIndex:
    def __init__(self, lines: list[Line]):
        self.lines = sorted(lines, key=lambda l: l.start)
        self.starts = [l.start for l in self.lines]

    def within(self, t0: float, t1: float) -> list[Line]:
        i = bisect.bisect_left(self.starts, t0 - 8.0)
        out = []
        while i < len(self.lines) and self.lines[i].start < t1:
            l = self.lines[i]
            if l.end > t0 and l.start < t1:
                out.append(l)
            i += 1
        return out


def score_window(t0: float, t1: float, tl: Timeline, idx: LineIndex, st: Settings) -> tuple[float, dict]:
    g = st.genre_cfg()
    mo, fa = tl.span(tl.motion, t0, t1), tl.span(tl.face, t0, t1)
    au, da = tl.span(tl.audio, t0, t1), tl.span(tl.dark, t0, t1)
    motion = float(np.percentile(mo, 75)) if len(mo) else 0.0
    face = float(fa.mean()) if len(fa) else 0.0
    audio = float(np.percentile(au, 90)) if len(au) else 0.0
    dark = float(da.mean()) if len(da) else 0.0
    plot, tags = 0.0, []
    for l in idx.within(t0, t1):
        # phạt cửa sổ chỉ chạm một phần câu thoại: ưu tiên cửa sổ giữ TRỌN câu nút thắt
        cover = (min(t1, l.end) - max(t0, l.start)) / max(0.3, l.end - l.start)
        eff = l.plot * min(1.0, max(0.0, cover)) ** 1.5
        if eff > plot:
            plot = eff
        if cover >= 0.6:
            tags.extend(l.tags)
    wsum = g["w_motion"] + g["w_face"] + g["w_audio"] + g["w_dark"] + g["w_plot"]
    s = (g["w_motion"] * motion + g["w_face"] * face + g["w_audio"] * audio
         + g["w_dark"] * dark + g["w_plot"] * plot) / wsum
    if st.highlight_priority and motion >= 0.8:      # Highlight Zone
        s += 0.12 + 0.08 * (motion - 0.8) / 0.2
    return s, dict(motion=motion, face=face, audio=audio, dark=dark, plot=plot, tags=sorted(set(tags)))


def _length_for(motion: float, has_key: bool, st: Settings) -> float:
    L = st.shot_len
    if motion >= 0.8:
        L *= 0.85          # hành động: nhịp cắt nhanh hơn
    if has_key:
        L *= 1.15          # chừa chỗ cho câu thoại
    return float(min(st.shot_max, max(st.shot_min, L)))


def build_candidates(scenes: list[Scene], tl: Timeline, lines: list[Line], st: Settings) -> list[Shot]:
    idx = LineIndex(lines)
    cands: list[Shot] = []
    for sc in scenes:
        if sc.blank or sc.credits:
            continue
        a, b = sc.start + EDGE, sc.end - EDGE
        if b - a < 0.8:
            continue
        wins: list[tuple[float, float]] = []
        if st.micro_cut:
            step = max(0.5, st.shot_len / 2)
            t = a
            while t + st.shot_min <= b + 1e-6:
                m = float(np.percentile(tl.span(tl.motion, t, t + st.shot_len), 75))
                L = min(_length_for(m, False, st), b - t)
                if L >= min(st.shot_min, b - a):
                    wins.append((t, t + L))
                t += step
            if not wins:
                wins.append((a, b if b - a <= st.shot_max else a + st.shot_max))
        else:
            cap = 8.0
            if b - a <= cap:
                wins.append((a, b))
            else:
                t = a
                while t + 4 <= b:
                    wins.append((t, min(b, t + cap)))
                    t += cap / 2
        # cửa sổ neo vào câu thoại nút thắt (giữ trọn câu)
        for l in idx.within(a, b):
            if l.plot >= 0.7:
                t0 = max(a, l.start - 0.4)
                L = min(max(st.shot_min, l.end - t0 + 0.4), st.shot_max + 1.5, b - t0)
                if L >= 1.0:
                    wins.append((t0, t0 + L))
        for (t0, t1) in wins:
            s, f = score_window(t0, t1, tl, idx, st)
            cands.append(Shot(round(t0, 2), round(t1, 2), sc.id, s, f["motion"], f["face"], sc.kind,
                              f["tags"], must=f["plot"] >= 0.7))
    return cands


def _overlaps(sel_by_scene: dict, c: Shot, gap: float = 0.3) -> bool:
    for s in sel_by_scene.get(c.scene_id, []):
        if c.start < s.end + gap and c.end > s.start - gap:
            return True
    return False


SNAP_MAX = 1.2          # giây: tối đa kéo dài/lùi một đầu shot để khớp ranh giới câu thoại


def snap_shots(shots: list[Shot], scenes: list[Scene], lines: list[Line], st: Settings) -> list[Shot]:
    """Khớp đầu/cuối shot với ranh giới câu thoại: không cắt ngang giữa câu, và KHÔNG BAO GIỜ vượt ranh giới cảnh.
    Shot nút thắt đã được neo trọn câu nên giữ nguyên."""
    idx = LineIndex(lines)
    bounds = {sc.id: (sc.start + EDGE, sc.end - EDGE) for sc in scenes}
    for sh in shots:
        if sh.must or sh.role != "main" or sh.scene_id not in bounds:
            continue
        lo, hi = bounds[sh.scene_id]
        for l in idx.within(sh.start, sh.end):
            if l.start < sh.start - 0.05 < l.end - 0.25:              # đầu shot rơi giữa câu
                if sh.start - l.start <= SNAP_MAX and l.start >= lo:
                    sh.start = l.start - 0.05 if l.start - 0.05 >= lo else l.start
                elif l.end + 0.05 < sh.end - st.shot_min * 0.8:
                    sh.start = l.end + 0.05
            if l.start < sh.end - 0.05 < l.end - 0.25:                # cuối shot rơi giữa câu
                if l.end - sh.end <= SNAP_MAX and l.end + 0.12 <= hi:
                    sh.end = l.end + 0.12
                elif l.start - 0.05 > sh.start + st.shot_min * 0.8:
                    sh.end = l.start - 0.05
        sh.start, sh.end = round(max(lo, sh.start), 2), round(min(hi, sh.end), 2)
    out: list[Shot] = []
    for sh in sorted(shots, key=lambda x: x.start):                    # gỡ chồng lấn do kéo dài
        prev = out[-1] if out else None
        if prev and prev.scene_id == sh.scene_id and sh.start < prev.end + 0.05:
            sh.start = round(prev.end + 0.05, 2)
        if sh.end - sh.start >= 1.0:
            out.append(sh)
    return out


def plan_cut(scenes: list[Scene], tl: Timeline, lines: list[Line], st: Settings, log=print) -> list[Shot]:
    cands = build_candidates(scenes, tl, lines, st)
    if not cands:
        raise RuntimeError("Không có cảnh khả dụng (kiểm tra video / vùng bỏ qua đầu-cuối).")
    outro_total = st.outro_shots * OUTRO_LEN if st.outro_montage else 0.0
    main_target = max(20.0, st.target_sec() - outro_total)

    if st.extraction_mode == "highlight":           # lọc 20% cảnh điểm thấp nhất, trừ cảnh có nút thắt
        thr = float(np.percentile([c.score for c in cands], 20))
        pool = [c for c in cands if c.score >= thr or c.must]
    else:
        pool = list(cands)

    avg = float(np.mean([c.dur for c in pool]))
    n_target = max(3, int(round(main_target / avg)))
    film = max(tl.duration, 1.0)
    B = int(min(24, max(6, film // 240)))
    bucket_of = lambda c: min(B - 1, int(c.start / film * B))
    by_bucket: dict[int, list[Shot]] = {b: [] for b in range(B)}
    for c in pool:
        by_bucket[bucket_of(c)].append(c)
    top_sum = np.array([sum(sorted((c.score for c in by_bucket[b]), reverse=True)[:5]) for b in range(B)])
    w = 0.6 / B + 0.4 * (top_sum / top_sum.sum() if top_sum.sum() > 0 else 1.0 / B)
    quota = np.maximum(1, np.round(w * n_target)).astype(int)

    selected: list[Shot] = []
    sel_scene: dict[int, list[Shot]] = {}
    per_scene_cap = 3 if st.micro_cut else 1

    def take(c: Shot) -> bool:
        lst = sel_scene.setdefault(c.scene_id, [])
        if len(lst) >= per_scene_cap or _overlaps(sel_scene, c):
            return False
        lst.append(c)
        selected.append(c)
        return True

    # 1) bắt buộc giữ các phân đoạn nút thắt mạnh nhất (tối đa 30% số shot)
    musts = sorted((c for c in pool if c.must), key=lambda c: (c.score + (0.5 if c.tags else 0)), reverse=True)
    cap_must = max(1, int(n_target * 0.30))
    for c in musts:
        if sum(1 for s in selected if s.must) >= cap_must:
            break
        take(c)
    filled = np.zeros(B, dtype=int)
    for s in selected:
        filled[bucket_of(s)] += 1
    # 2) lấp theo hạn mức từng đoạn phim để cốt truyện đủ đầu-giữa-cuối
    for b in range(B):
        for c in sorted(by_bucket[b], key=lambda c: c.score, reverse=True):
            if filled[b] >= quota[b]:
                break
            if take(c):
                filled[b] += 1
    # 3) còn thiếu → lấy theo điểm toàn phim
    total = lambda: sum(s.dur for s in selected)
    for c in sorted(pool, key=lambda c: c.score, reverse=True):
        if total() >= main_target * 0.97:
            break
        if c not in selected:
            take(c)
    # 4) dư → bỏ cảnh điểm thấp nhất (không đụng cảnh nút thắt)
    while total() > main_target * 1.04 and len(selected) > 3:
        drop = min((s for s in selected if not s.must), key=lambda s: s.score, default=None)
        if drop is None:
            break
        selected.remove(drop)
        sel_scene[drop.scene_id].remove(drop)

    selected.sort(key=lambda s: s.start)
    selected = snap_shots(selected, scenes, lines, st)
    log(f"Chọn {len(selected)} shot · {total():.0f}s (mục tiêu {main_target:.0f}s) · "
        f"{sum(1 for s in selected if s.must)} shot giữ vì thoại nút thắt")

    if st.outro_montage and st.outro_shots > 0:
        selected += plan_outro(cands, selected, tl, st)
    return selected


def plan_outro(cands: list[Shot], selected: list[Shot], tl: Timeline, st: Settings) -> list[Shot]:
    """Highlight liên hoàn: các khoảnh khắc mạnh nhất, mỗi cái ~1.4s, đặt cuối video, không đứng hình.
    Lượt 1 ưu tiên cảnh chưa dùng; lượt 2 cho phép cảnh đã dùng nhưng khác thời điểm với shot đã chọn."""
    used = {s.scene_id for s in selected}
    pool = sorted(cands, key=lambda c: 0.6 * c.motion + 0.4 * c.score, reverse=True)
    pick: list[Shot] = []

    def moment(c: Shot) -> Optional[Shot]:
        seg = tl.span(tl.motion, c.start, c.end)
        off = int(np.argmax(seg)) / tl.fps if len(seg) else 0.0
        t0 = min(max(c.start, c.start + off - OUTRO_LEN / 2), max(c.start, c.end - OUTRO_LEN))
        t1 = min(c.end, t0 + OUTRO_LEN)
        if t1 - t0 < 0.9:
            return None
        if any(abs(p.start - t0) < 2.0 for p in pick):
            return None
        return Shot(round(t0, 2), round(t1, 2), c.scene_id, c.score, c.motion, c.face, c.kind, [], False, "outro")

    for strict in (True, False):
        for c in pool:
            if len(pick) >= st.outro_shots:
                return pick
            if strict and c.scene_id in used:
                continue
            m = moment(c)
            if m is None:
                continue
            if not strict and any(m.start < s.end and m.end > s.start for s in selected):
                continue                       # không lặp lại đúng đoạn đã dùng ở thân bài
            pick.append(m)
    return pick


def make_blocks(shots: list[Shot], lines: list[Line], st: Settings) -> list[Block]:
    """Gom shot liên tiếp thành khối lời bình ~block_sec; outro thành 1 khối riêng không lời."""
    main = [s for s in shots if s.role == "main"]
    outro = [s for s in shots if s.role == "outro"]
    idx = LineIndex(lines)
    groups: list[list[Shot]] = []
    cur: list[Shot] = []
    for s in main:
        cur.append(s)
        if sum(x.dur for x in cur) >= st.block_sec:
            groups.append(cur)
            cur = []
    if cur:
        if groups and sum(x.dur for x in cur) < 0.4 * st.block_sec:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    blocks = []
    for i, g in enumerate(groups):
        dlg, seen = [], set()
        for s in g:
            for l in idx.within(s.start, s.end):
                if (l.start, l.text) not in seen:
                    seen.add((l.start, l.text))
                    dlg.append({"t": round(l.start, 1), "text": l.text, "plot": round(l.plot, 2)})
        blocks.append(Block(i, g, dialogue=dlg))
    if outro:
        blocks.append(Block(len(blocks), outro, role="outro"))
    return blocks


def shots_from_dict(d: dict) -> Shot:
    return Shot(**{k: v for k, v in d.items() if k in Shot.__dataclass_fields__})


def blocks_from_dict(ds: list[dict]) -> list[Block]:
    out = []
    for d in ds:
        out.append(Block(d["id"], [shots_from_dict(s) for s in d["shots"]], d.get("text", ""),
                         d.get("role", "main"), d.get("dialogue", [])))
    return out
