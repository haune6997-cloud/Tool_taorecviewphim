"""Giọng đọc tiếng Việt qua edge-tts: preset, prosody theo thể loại, cache theo nội dung, chống lỗi mạng bước 3.

Chống lỗi mạng
* Mỗi lần thử dùng KẾT NỐI MỚI, backoff luỹ thừa + ngẫu nhiên, timeout kết nối/nhận rõ ràng, tôn trọng proxy hệ thống.
* Lỗi được PHÂN LOẠI và nói rõ nguyên nhân: 403 (đồng hồ máy lệch / edge-tts quá cũ), DNS/mất mạng, timeout, bị chặn.
* Ghi file nguyên tử (không để lại mp3 hỏng) → bấm chạy lại bước 3 sẽ TIẾP TỤC từ câu còn thiếu, không đọc lại từ đầu.
* Hai lượt: lượt 1 cả danh sách; câu lỗi được thử lại ở lượt 2 sau một khoảng nghỉ; còn lỗi → thử gTTS (nếu đã cài).
* Mạng sập hẳn (nhiều câu liên tiếp lỗi cùng nguyên nhân) → dừng sớm, không "đập" máy chủ thêm.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

from .config import VOICES, Settings
from .media import Job


class TTSError(RuntimeError):
    def __init__(self, kind: str, detail: str, hint: str = ""):
        self.kind, self.detail, self.hint = kind, detail, hint
        super().__init__(f"[Giọng đọc · {kind}] {detail}" + (f"\n→ {hint}" if hint else ""))


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


def classify_error(ex: BaseException) -> TTSError:
    """Đổi lỗi edge-tts/aiohttp thành nguyên nhân + cách xử lý bằng tiếng Việt."""
    name, raw = type(ex).__name__, str(ex)
    low = (name + " " + raw).lower()
    if "403" in low or "invalid response status" in low or "handshake" in low:
        return TTSError("bị từ chối 403", "Microsoft từ chối kết nối giọng đọc (thường do mã bảo vệ Sec-MS-GEC lệch giờ hoặc bản edge-tts cũ).",
                        "① Bật đồng bộ giờ tự động của Windows (đồng hồ lệch vài phút là bị 403). "
                        "② Chạy: pip install -U edge-tts. ③ Tắt VPN/proxy rồi thử lại.")
    if isinstance(ex, (asyncio.TimeoutError, TimeoutError)) or "timeout" in low or "timed out" in low:
        return TTSError("hết thời gian chờ", "Máy chủ giọng đọc không trả lời kịp.",
                        "Mạng chậm hoặc bị nghẽn: thử lại sau, đổi mạng, hoặc tắt VPN.")
    if "noaudioreceived" in low or "no audio" in low:
        return TTSError("không nhận được âm thanh", "Máy chủ không trả âm thanh cho câu này (câu rỗng/ký tự lạ).",
                        "Sửa lại câu lời bình (bỏ ký tự đặc biệt) rồi thử lại.")
    if any(k in low for k in ("getaddrinfo", "name resolution", "connectorerror", "cannot connect", "unreachable",
                              "connection reset", "connection refused", "network", "clientoserror", "serverdisconnected")):
        return TTSError("lỗi mạng", "Không kết nối được tới máy chủ giọng đọc (speech.platform.bing.com).",
                        "Kiểm tra Internet, đổi DNS (8.8.8.8), tắt/bật VPN. Nếu mạng vẫn tốt, nhà mạng có thể đang chặn "
                        "dịch vụ này – thử đổi sang 4G/mạng khác.")
    return TTSError("không xác định", f"{name}: {raw[:160]}", "Chạy: pip install -U edge-tts rồi thử lại.")


def _proxy() -> Optional[str]:
    return os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("ALL_PROXY") or None


def _communicate(text: str, edge_voice: str, rate: int, pitch: int):
    import edge_tts
    kw: dict = {"rate": f"{rate:+d}%", "pitch": f"{pitch:+d}Hz"}
    params = inspect.signature(edge_tts.Communicate.__init__).parameters
    if "connect_timeout" in params:
        kw["connect_timeout"] = 15
    if "receive_timeout" in params:
        kw["receive_timeout"] = 60
    if "proxy" in params and _proxy():
        kw["proxy"] = _proxy()
    return edge_tts.Communicate(text, edge_voice, **kw)


async def _one(text: str, edge_voice: str, rate: int, pitch: int, out: str, retries: int = 4) -> None:
    """Đọc 1 câu → file mp3 (ghi nguyên tử). Mỗi lần thử một kết nối mới, backoff 2-4-8-16 s."""
    last: Optional[TTSError] = None
    for a in range(retries):
        try:
            buf = bytearray()
            comm = _communicate(text, edge_voice, rate, pitch)

            async def go():
                async for ch in comm.stream():
                    if ch["type"] == "audio":
                        buf.extend(ch["data"])

            await asyncio.wait_for(go(), timeout=90)
            if len(buf) > 500:
                tmp = out + ".part"
                Path(tmp).write_bytes(bytes(buf))
                os.replace(tmp, out)
                await asyncio.sleep(0.25)             # nghỉ nhẹ giữa các câu để tránh bị chặn vì dồn dập
                return
            last = TTSError("không nhận được âm thanh", "âm thanh trả về rỗng/quá ngắn")
        except Exception as ex:  # noqa: BLE001
            last = classify_error(ex)
        await asyncio.sleep(min(20.0, 2.0 * (2 ** a)) + random.uniform(0, 1.0))
    raise last or TTSError("không xác định", "thất bại không rõ lý do")


def _gtts(text: str, out: str) -> None:
    from gtts import gTTS                              # tuỳ chọn: pip install gTTS
    tmp = out + ".part"
    gTTS(text, lang="vi").save(tmp)
    os.replace(tmp, out)


def synth_many(items: list[tuple[str, str]], st: Settings, job: Job, concurrency: int = 1) -> None:
    """items = [(text, out_mp3)]. Bỏ qua file đã có (cache). Chạy tuần tự để tránh bị chặn; 2 lượt + gTTS dự phòng."""
    v = VOICES.get(st.resolved_voice(), VOICES["hoaimy"])
    rate, pitch = resolve_prosody(st)
    todo = [(clean(t), o) for t, o in items if t.strip() and not Path(o).exists()]
    for _, o in todo:
        Path(o + ".part").unlink(missing_ok=True)
    if not todo:
        return
    job.log(f"Lồng tiếng: {len(todo)} câu cần tạo ({len(items) - len(todo)} câu đã có sẵn trong cache).")
    failed: list[tuple[str, str]] = []
    errors: list[TTSError] = []

    async def run(batch: list[tuple[str, str]], label: str, stop_after: int) -> list[tuple[str, str]]:
        bad: list[tuple[str, str]] = []
        streak = 0
        for k, (t, o) in enumerate(batch, 1):
            job.check()
            try:
                await _one(t, v["edge"], rate, pitch, o)
                streak = 0
            except TTSError as e:
                errors.append(e)
                bad.append((t, o))
                streak += 1
                job.log(f"   ⚠ {label}: câu {k}/{len(batch)} lỗi – {e.kind}")
                if streak >= stop_after:            # lỗi liên tiếp cùng một kiểu → mạng/dịch vụ sập, đừng đập thêm
                    bad.extend(batch[k:])
                    job.log(f"   ⛔ {streak} câu liên tiếp lỗi ({e.kind}) → dừng lượt này để không bị chặn thêm.")
                    break
            job.progress(0.05 + 0.9 * (k / len(batch) if label == "lượt 1" else 1.0), f"Lồng tiếng {label}: {k}/{len(batch)}")
        return bad

    failed = asyncio.run(run(todo, "lượt 1", 4))
    if failed:
        wait = 8.0 if len(failed) < len(todo) else 15.0
        job.log(f"   ↻ {len(failed)} câu lỗi → nghỉ {wait:.0f}s rồi thử lại (lượt 2)…")
        for _ in range(int(wait * 2)):
            job.check()
            time.sleep(0.5)
        failed = asyncio.run(run(failed, "lượt 2", 3))
    if failed:
        try:
            import gtts  # noqa: F401
        except ImportError:
            gtts = None
        if gtts is not None:
            job.log("   ↻ Thử giọng dự phòng gTTS (khác giọng Edge) cho các câu còn lỗi…")
            still = []
            for t, o in failed:
                job.check()
                try:
                    _gtts(t, o)
                except Exception:  # noqa: BLE001
                    still.append((t, o))
            failed = still
    if failed:
        main = max(errors, key=lambda e: sum(1 for x in errors if x.kind == e.kind)) if errors else None
        raise TTSError(main.kind if main else "lỗi", f"còn {len(failed)}/{len(todo)} câu chưa đọc được. "
                       + (main.detail if main else ""),
                       ((main.hint + "\n") if main else "") + "Các câu đã đọc xong được GIỮ LẠI: sau khi khắc phục chỉ cần bấm lại bước 3, "
                       "chương trình sẽ đọc tiếp phần còn thiếu. (Tuỳ chọn: pip install gTTS để có giọng dự phòng.)")


def check_connection(st: Optional[Settings] = None) -> tuple[bool, str]:
    """Thử đọc một câu ngắn (1 lần, không thử lại) để báo chính xác lỗi giọng đọc."""
    import tempfile
    out = os.path.join(tempfile.gettempdir(), "reviewcut_tts_check.mp3")
    Path(out).unlink(missing_ok=True)
    v = VOICES["hoaimy"]
    try:
        asyncio.run(_one("Xin chào", v["edge"], 0, 0, out, retries=1))
        return True, "Giọng đọc Edge hoạt động tốt."
    except TTSError as e:
        return False, str(e)


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
