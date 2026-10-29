"""Giọng đọc tiếng Việt qua edge-tts, có preset, prosody theo thể loại, cache theo nội dung."""
from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path

from .config import VOICES, Settings
from .media import Job


def resolve_prosody(st: Settings) -> tuple[int, int]:
    """(rate %, pitch Hz) = nền giọng + hiệu chỉnh thể loại + tốc độ người dùng, kẹp trong giới hạn Edge."""
    v = VOICES.get(st.resolved_voice(), VOICES["hoaimy"])
    g = st.genre_cfg()
    rate = v["rate"] + g["rate"] + round((st.speed - 1.0) * 100)
    pitch = v["pitch"] + g["pitch"]
    return max(-50, min(100, rate)), max(-50, min(50, pitch))


def clean(text: str) -> str:
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f*#_~`<>|\[\]{}\\]", " ", text)
    t = t.replace("&", " và ").replace("%", " phần trăm ").replace("...", ". ").replace("…", ". ")
    t = re.sub(r"[\"“”‘’]", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


async def _one(text: str, edge_voice: str, rate: int, pitch: int, out: str, retries: int = 3) -> None:
    import edge_tts
    last = None
    for a in range(retries):
        try:
            buf = bytearray()
            comm = edge_tts.Communicate(text, edge_voice, rate=f"{rate:+d}%", pitch=f"{pitch:+d}Hz")
            async def go():
                async for ch in comm.stream():
                    if ch["type"] == "audio":
                        buf.extend(ch["data"])
            await asyncio.wait_for(go(), timeout=60)
            if len(buf) > 500:
                Path(out).write_bytes(bytes(buf))
                return
        except Exception as ex:  # noqa: BLE001
            last = ex
        await asyncio.sleep(1.0 * (a + 1))
    raise RuntimeError(f"edge-tts thất bại sau {retries} lần ({type(last).__name__ if last else 'rỗng'}). "
                       "Kiểm tra kết nối Internet.")


def synth_many(items: list[tuple[str, str]], st: Settings, job: Job, concurrency: int = 3) -> None:
    """items = [(text, out_mp3)]. Bỏ qua file đã có (cache)."""
    v = VOICES.get(st.resolved_voice(), VOICES["hoaimy"])
    rate, pitch = resolve_prosody(st)
    todo = [(clean(t), o) for t, o in items if t.strip() and not Path(o).exists()]

    async def main():
        sem = asyncio.Semaphore(concurrency)
        done = 0
        async def w(t, o):
            nonlocal done
            async with sem:
                job.check()
                await _one(t, v["edge"], rate, pitch, o)
                done += 1
                job.progress(0.05 + 0.9 * done / max(1, len(todo)), f"Lồng tiếng {done}/{len(todo)}")
        await asyncio.gather(*(w(t, o) for t, o in todo))
    if todo:
        asyncio.run(main())


def cache_path(workdir: str, text: str, st: Settings) -> str:
    v = VOICES.get(st.resolved_voice(), VOICES["hoaimy"])
    rate, pitch = resolve_prosody(st)
    h = hashlib.sha1(f"{clean(text)}|{v['edge']}|{rate}|{pitch}".encode()).hexdigest()[:16]
    d = Path(workdir) / "tts"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f"{h}.mp3")


def preview(text: str, st: Settings, out: str) -> None:
    job = Job(log=lambda m: None)
    Path(out).unlink(missing_ok=True)
    synth_many([(text, out)], st, job)
