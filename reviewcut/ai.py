"""Bộ điều phối AI: Groq (ưu tiên 1) → Gemini (ưu tiên 2) → Ollama cục bộ (hỏi 1 lần mỗi bước) → kịch bản dự phòng.

Điểm chính
* Mọi lỗi được PHÂN LOẠI (mạng / timeout / key sai / model không tồn tại / vượt tốc độ / hết hạn mức ngày /
  yêu cầu quá lớn / máy chủ quá tải / bị lọc an toàn…) kèm HƯỚNG DẪN xử lý bằng tiếng Việt, có ghi rõ nhà cung cấp.
* ``RateLimiter``: không bao giờ gửi quá ``rpm`` request và quá ngân sách token trong 60 giây (Groq free: 30 RPM,
  12K token/phút với llama-3.3-70b) → chờ trước khi gửi thay vì bị 429.
* Chỉ thử lại lỗi có thể tự khỏi (mạng chập chờn, 429 theo phút, 5xx) với backoff + đọc Retry-After.
* Khi một nhà cung cấp hỏng ở mức dịch vụ (key/model/hạn mức/mạng) → nhớ là "hỏng" trong CẢ BƯỚC, không thử lại ở từng lô.
* Hết nhà cung cấp đám mây → hỏi người dùng ĐÚNG 1 LẦN CHO MỖI BƯỚC: dùng Ollama / kịch bản dự phòng / dừng.
  Sang bước sau mới thử lại đám mây và hỏi lại nếu cần.
* Hộp thoại do giao diện cung cấp qua hàm ``asker`` (luồng nền không tự mở cửa sổ Tkinter).
"""
from __future__ import annotations

import base64
import json
import random
import re
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

import requests

from .media import Cancelled

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_ROOT = "https://generativelanguage.googleapis.com/v1beta"
OLLAMA_URL = "http://localhost:11434"

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GROQ_VISION = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
RETIRED_GEMINI = ("gemini-1.0", "gemini-1.5", "gemini-2.0", "gemini-pro")

# Giới hạn token/phút gói free của Groq (docs: console.groq.com/docs/rate-limits). Model lạ → 6000 cho an toàn.
GROQ_TPM = {"llama-3.3-70b-versatile": 12000, "llama-3.1-8b-instant": 6000,
            "meta-llama/llama-4-scout-17b-16e-instruct": 30000,
            "meta-llama/llama-4-maverick-17b-128e-instruct": 6000}

# ---------------------------------------------------------------- các loại lỗi
NETWORK, TIMEOUT, AUTH, MODEL, RATE, QUOTA_DAY, QUOTA_NONE = (
    "network", "timeout", "auth", "model", "rate", "quota_day", "quota_none")
SERVER, BAD_REQUEST, TOO_LARGE, LOCATION, SAFETY, EMPTY, OLLAMA, UNAVAILABLE, UNKNOWN = (
    "server", "bad_request", "too_large", "location", "safety", "empty", "ollama", "unavailable", "unknown")

# lỗi của dịch vụ/tài khoản (đáng để đổi nhà cung cấp) ≠ lỗi của riêng yêu cầu (bị chặn an toàn, rỗng, quá lớn…)
SERVICE_LEVEL = {NETWORK, TIMEOUT, AUTH, MODEL, RATE, QUOTA_DAY, QUOTA_NONE, SERVER, LOCATION, UNKNOWN}
RETRYABLE = {NETWORK, TIMEOUT, RATE, SERVER}
LABEL = {
    NETWORK: "Lỗi mạng", TIMEOUT: "Hết thời gian chờ", AUTH: "API key / quyền truy cập",
    MODEL: "Model không tồn tại", RATE: "Vượt giới hạn tốc độ (theo phút)", QUOTA_DAY: "Hết hạn mức (theo ngày)",
    QUOTA_NONE: "Model không có hạn mức cho key này", SERVER: "Máy chủ quá tải", BAD_REQUEST: "Yêu cầu bị từ chối",
    TOO_LARGE: "Yêu cầu quá lớn", LOCATION: "Khu vực không được hỗ trợ", SAFETY: "Bị bộ lọc an toàn chặn",
    EMPTY: "Phản hồi rỗng", OLLAMA: "Lỗi Ollama", UNAVAILABLE: "AI đang tắt", UNKNOWN: "Lỗi không xác định",
}


class AIError(RuntimeError):
    """Lỗi AI có phân loại. ``str(err)`` = [nhà cung cấp · loại · HTTP] mô tả + hướng dẫn xử lý (đã xoá API key)."""

    def __init__(self, kind: str, detail: str = "", status: Optional[int] = None, hint: str = "",
                 retry_after: Optional[float] = None, provider: str = ""):
        self.kind, self.detail, self.status, self.hint = kind, detail, status, hint
        self.retry_after, self.provider = retry_after, provider
        head = (f"[{provider + ' · ' if provider else ''}{LABEL.get(kind, kind)}"
                f"{f' · HTTP {status}' if status else ''}] {detail}").strip()
        super().__init__(head + (f"\n→ {hint}" if hint else ""))

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE

    @property
    def service_level(self) -> bool:
        return self.kind in SERVICE_LEVEL

    @property
    def short(self) -> str:
        return f"{self.provider + ': ' if self.provider else ''}{LABEL.get(self.kind, self.kind)}" \
               f"{f' (HTTP {self.status})' if self.status else ''}"


GeminiError = AIError            # tên cũ, giữ để mã cũ vẫn chạy


class AIUnavailable(AIError):
    """AI không dùng được trong bước này (người dùng chọn kịch bản dự phòng). Không phải lỗi mới."""

    def __init__(self, why: str = "AI đang tắt cho bước này – dùng chế độ dự phòng."):
        super().__init__(UNAVAILABLE, why)


def scrub(text: str, *keys: str) -> str:
    """Xoá API key khỏi chuỗi lỗi/log."""
    for k in keys:
        if k:
            text = text.replace(k, "***")
    text = re.sub(r"(key=)[\w\-]{12,}", r"\1***", text)
    return re.sub(r"(gsk_|AIza)[\w\-]{10,}", r"\1***", text)


# ---------------------------------------------------------------- phân loại lỗi
def _wait_from_text(low: str) -> Optional[float]:
    """'try again in 1m3.5s' / '12.4s' / '850ms' → giây."""
    m = re.search(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)(ms|s)", low)
    if not m:
        return None
    sec = float(m.group(2)) / (1000.0 if m.group(3) == "ms" else 1.0) + 60.0 * int(m.group(1) or 0)
    return sec


def classify_http(provider: str, status: int, body: str, model: str = "", headers: Optional[dict] = None) -> AIError:
    """Chuyển (nhà cung cấp, HTTP status, nội dung lỗi) thành AIError có hướng dẫn cụ thể."""
    msg, st_name, details, code = body.strip()[:400], "", [], ""
    try:
        err = json.loads(body).get("error", {})
        msg = err.get("message", msg) or msg
        st_name = err.get("status", "") or ""
        code = str(err.get("code", "") or err.get("type", ""))
        details = err.get("details", []) or []
    except (ValueError, AttributeError, TypeError):
        pass
    low, flat = msg.lower(), (json.dumps(details) + " " + code).lower()
    hdr = {k.lower(): v for k, v in (headers or {}).items()}
    retry = None
    ra = hdr.get("retry-after")
    if ra and str(ra).replace(".", "", 1).isdigit():
        retry = float(ra)
    m = re.search(r'"retrydelay":\s*"([\d.]+)s"', flat)
    if m:
        retry = float(m.group(1))
    retry = retry or _wait_from_text(low)
    short = scrub(msg, "")[:240]
    P = provider
    keyurl = {"Groq": "https://console.groq.com/keys", "Gemini": "https://aistudio.google.com/apikey"}.get(P, "")

    if status == 401 or code == "invalid_api_key" or (status == 400 and (
            "api key not valid" in low or "api_key_invalid" in flat + low or "api key expired" in low)):
        return AIError(AUTH, short, status, f"API key {P} sai, hết hạn hoặc đã bị thu hồi. Tạo key mới tại {keyurl} "
                       "rồi dán lại (không để thừa dấu cách). Lỗi này KHÔNG phải do mạng hay model.", provider=P)
    if status == 403:
        if "leaked" in low or "reported" in low:
            return AIError(AUTH, short, status, f"Key {P} bị khoá vì đã lộ công khai. Tạo key mới, đừng đăng key lên mạng.", provider=P)
        if "has not been used" in low or "disabled" in low or "service_disabled" in flat:
            return AIError(AUTH, short, status, "Dự án của key chưa bật Generative Language API. Tạo key bằng Google AI Studio.", provider=P)
        if "location" in low or "region" in low or "country" in low:
            return AIError(LOCATION, short, status, f"Khu vực/mạng của bạn bị {P} chặn – thử VPN hoặc dùng Ollama.", provider=P)
        return AIError(AUTH, short, status, f"Key {P} không có quyền dùng model/API này, hoặc tổ chức bị hạn chế. "
                       f"Kiểm tra tại {keyurl}.", provider=P)
    if status == 404 or code in ("model_not_found", "model_decommissioned") or "decommissioned" in low \
            or ("model" in low and "does not exist" in low):
        return AIError(MODEL, short, status, f"Model '{model}' không tồn tại, đã ngừng hoặc key không được dùng. "
                       f"Đổi 'Model {P}' trong cấu hình (Groq khuyên dùng {DEFAULT_GROQ_MODEL}; "
                       f"Gemini khuyên dùng {DEFAULT_GEMINI_MODEL}).", provider=P)
    if status == 413 or (status == 400 and ("too large" in low or "too long" in low or "context length" in low
                                            or "maximum context" in low)):
        return AIError(TOO_LARGE, short, status, "Yêu cầu vượt giới hạn kích thước/token của model. "
                       "Chương trình sẽ tự chia nhỏ lô; nếu vẫn lỗi hãy giảm thời lượng review hoặc đổi model.", provider=P)
    if status == 429:
        if "limit: 0" in low or '"limit": 0' in flat or "limit\": 0" in flat:
            return AIError(QUOTA_NONE, short, status, f"Model '{model}' không có hạn mức miễn phí với key này. "
                           "Chọn model khác hoặc bật thanh toán.", provider=P)
        per_day = ("per day" in low or "perday" in flat or "(rpd)" in low or "(tpd)" in low or "daily" in low
                   or "tokens per day" in low or "requests per day" in low)
        if per_day:
            return AIError(QUOTA_DAY, short, status,
                           f"{P} đã hết hạn mức NGÀY của model '{model}'. Chờ reset (Groq: 0h UTC ≈ 7h sáng VN; "
                           "Gemini: ≈ 14–15h VN), đổi model/nhà cung cấp, hoặc dùng Ollama.", retry_after=retry, provider=P)
        return AIError(RATE, short, status, f"Gửi quá nhanh / quá nhiều token trong 1 phút với {P}. "
                       "Chương trình tự chờ rồi thử lại; nếu lặp lại hãy giảm 'Giới hạn request/phút'.",
                       retry_after=retry, provider=P)
    if status in (500, 502, 503, 504):
        return AIError(SERVER, short or f"{P} trả lỗi tạm thời", status,
                       f"Lỗi phía {P} (quá tải/bảo trì), không phải lỗi của bạn. Thường tự khỏi sau vài phút.",
                       retry_after=retry, provider=P)
    if status == 400:
        if "location is not supported" in low or st_name == "FAILED_PRECONDITION":
            return AIError(LOCATION, short, status, "Khu vực của bạn chưa được hỗ trợ (hoặc cần bật thanh toán) – "
                           "thử VPN hoặc dùng Ollama.", provider=P)
        return AIError(BAD_REQUEST, short, status, "Yêu cầu không hợp lệ (ảnh/prompt quá lớn, hoặc model không hỗ trợ ảnh). "
                       "Kiểm tra 'Model ảnh' hoặc giảm số shot mỗi lô.", provider=P)
    return AIError(UNKNOWN, short or f"HTTP {status}", status, "Lỗi lạ – xem nhật ký; thử lại sau hoặc đổi model.", provider=P)


def classify_exception(provider: str, ex: Exception, timeout: float) -> AIError:
    """Phân biệt rõ: timeout / DNS / SSL / proxy / mất kết nối."""
    raw = scrub(f"{type(ex).__name__}: {ex}")[:200]
    low = raw.lower()
    host = {"Groq": "api.groq.com", "Gemini": "generativelanguage.googleapis.com"}.get(provider, provider)
    if isinstance(ex, requests.Timeout):
        return AIError(TIMEOUT, f"không có phản hồi sau {timeout:.0f}s", hint="Mạng chậm hoặc yêu cầu nặng. "
                       "Thử lại; nếu lặp lại hãy kiểm tra Internet/VPN.", provider=provider)
    if "getaddrinfo" in low or "name resolution" in low or "nodename" in low or "name or service" in low:
        why, hint = f"không phân giải được tên miền {host} (DNS/mất Internet)", "Kiểm tra Internet, đổi DNS (8.8.8.8 / 1.1.1.1) hoặc bật/tắt VPN."
    elif isinstance(ex, requests.exceptions.SSLError) or "ssl" in low or "certificate" in low:
        why, hint = "lỗi chứng chỉ SSL", "Giờ hệ thống sai, phần mềm diệt virus/proxy chặn HTTPS, hoặc mạng công ty chèn chứng chỉ."
    elif isinstance(ex, requests.exceptions.ProxyError) or "proxy" in low:
        why, hint = "lỗi proxy", "Kiểm tra biến HTTPS_PROXY hoặc tắt proxy."
    elif "refused" in low or "unreachable" in low:
        why, hint = "kết nối bị từ chối/không tới được máy chủ", "Tường lửa hoặc nhà mạng đang chặn. Thử VPN/đổi mạng."
    else:
        why, hint = f"mất kết nối ({type(ex).__name__})", "Đường truyền chập chờn hoặc bị ngắt giữa chừng. Kiểm tra Internet rồi thử lại."
    return AIError(NETWORK, why, hint=hint + " (Key và model không phải nguyên nhân.)", provider=provider)


# ---------------------------------------------------------------- giới hạn tốc độ
class RateLimiter:
    """Cửa sổ trượt 60 giây theo số request và số token ước tính. ``wait`` chặn tới khi được phép gửi."""

    def __init__(self, rpm: int = 25, tpm: int = 0, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.rpm, self.tpm = max(1, int(rpm)), int(tpm)
        self.clock, self.sleep = clock, sleep
        self.events: deque[tuple[float, int]] = deque()
        self.block_until = 0.0              # do 429/Retry-After đặt ra
        self.lock = threading.Lock()
        self.total_waited = 0.0

    def _prune(self, now: float) -> None:
        while self.events and now - self.events[0][0] >= 60.0:
            self.events.popleft()

    def needed_wait(self, tokens: int) -> float:
        now = self.clock()
        self._prune(now)
        w = max(0.0, self.block_until - now)
        if len(self.events) >= self.rpm:
            w = max(w, 60.0 - (now - self.events[0][0]) + 0.05)
        if self.tpm and self.events:
            used, need = sum(t for _, t in self.events), min(tokens, self.tpm)
            if used + need > self.tpm:             # chờ tới khi đủ sự kiện cũ rơi khỏi cửa sổ
                freed = used
                for ts, t in self.events:
                    freed -= t
                    if freed + need <= self.tpm:
                        w = max(w, 60.0 - (now - ts) + 0.05)
                        break
                else:
                    w = max(w, 60.0)
        return w

    def wait(self, tokens: int = 0, log: Optional[Callable[[str], None]] = None,
             check: Optional[Callable[[], None]] = None, what: str = "") -> float:
        waited, told = 0.0, False
        while True:
            with self.lock:
                w = self.needed_wait(tokens)
                if w <= 0:
                    self.events.append((self.clock(), tokens))
                    return waited
            if log and not told and w > 2.0:
                log(f"⏳ Chờ ~{w:.0f}s để không vượt giới hạn {what or 'API'} ({self.rpm} request/phút"
                    f"{f', {self.tpm} token/phút' if self.tpm else ''}).")
                told = True
            step = min(w, 1.0)
            if check:
                check()
            self.sleep(step)
            waited += step
            self.total_waited += step

    def penalize(self, seconds: float) -> None:
        with self.lock:
            self.block_until = max(self.block_until, self.clock() + max(0.0, seconds))


def estimate_tokens(parts: list, max_out: int = 0, image_tokens: int = 1500) -> int:
    """Ước lượng thô: tiếng Việt ≈ 1 token / 2.6 ký tự; mỗi ảnh ≈ image_tokens; cộng phần trả lời tối đa."""
    chars = sum(len(p) if isinstance(p, str) else len(p.get("text", "")) for p in parts
                if isinstance(p, str) or (isinstance(p, dict) and "text" in p))
    imgs = sum(1 for p in parts if isinstance(p, dict) and "inline_data" in p)
    return int(chars / 2.6) + imgs * image_tokens + max_out


# ---------------------------------------------------------------- tiện ích phần nội dung
def text_of(parts: list) -> str:
    return "\n".join(p if isinstance(p, str) else p["text"] for p in parts
                     if isinstance(p, str) or (isinstance(p, dict) and "text" in p))


def images_of(parts: list) -> list[dict]:
    return [p["inline_data"] for p in parts if isinstance(p, dict) and "inline_data" in p]


def image_part(jpg_path: str) -> dict:
    with open(jpg_path, "rb") as f:
        return {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(f.read()).decode()}}


def parse_json(raw: str) -> Any:
    """Chịu được ```json fence, văn bản thừa quanh JSON, và JSON bị cắt cụt ở cuối (cố gắng vá)."""
    text = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    sb, eb, sc, ec = text.find("["), text.rfind("]"), text.find("{"), text.rfind("}")
    cands = []
    if sb != -1 and eb > sb:
        cands.append((sb, text[sb:eb + 1]))
    if sc != -1 and ec > sc:
        cands.append((sc, text[sc:ec + 1]))
    for _, c in sorted(cands):
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    raise ValueError("không đọc được JSON trong phản hồi AI")


# ---------------------------------------------------------------- các nhà cung cấp
class Provider:
    name = ""
    vision = False

    def __init__(self, log: Callable[[str], None]):
        self.log = log

    def model_for(self, needs_vision: bool) -> str:
        raise NotImplementedError

    def call(self, parts: list, json_mode: bool, temperature: float, timeout: float, max_out: int,
             needs_vision: bool) -> str:
        raise NotImplementedError


class GroqProvider(Provider):
    name = "Groq"

    def __init__(self, key: str, model: str, vision_model: str, log, rpm: int = 25):
        super().__init__(log)
        self.key = key.strip()
        self.model = (model or DEFAULT_GROQ_MODEL).strip()
        self.vision_model = (vision_model or DEFAULT_GROQ_VISION).strip()
        self.vision = True
        self.limiters: dict[str, RateLimiter] = {}
        self.rpm = rpm

    def model_for(self, needs_vision: bool) -> str:
        return self.vision_model if needs_vision else self.model

    def limiter(self, model: str) -> RateLimiter:
        if model not in self.limiters:
            tpm = int(GROQ_TPM.get(model, 6000) * 0.85)           # chừa 15% cho sai số ước lượng
            self.limiters[model] = RateLimiter(self.rpm, tpm)
        return self.limiters[model]

    def call(self, parts, json_mode, temperature, timeout, max_out, needs_vision) -> str:
        model = self.model_for(needs_vision)
        imgs = images_of(parts)
        if imgs:
            content: Any = [{"type": "text", "text": text_of(parts)}] + [
                {"type": "image_url", "image_url": {"url": f"data:{i['mime_type']};base64,{i['data']}"}} for i in imgs[:5]]
        else:
            content = text_of(parts)
        body = {"model": model, "messages": [{"role": "user", "content": content}],
                "temperature": temperature, "max_tokens": max_out}
        if json_mode and not imgs:
            body["response_format"] = {"type": "json_object"}
        for attempt in (0, 1):
            try:
                r = requests.post(GROQ_URL, json=body, timeout=timeout,
                                  headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            except requests.RequestException as ex:
                raise classify_exception(self.name, ex, timeout)
            if r.status_code == 400 and attempt == 0 and "response_format" in body and \
                    ("json_validate_failed" in r.text or "response_format" in r.text):
                body.pop("response_format")                      # model không xuất nổi JSON ép buộc → bỏ ràng buộc, tự parse
                continue
            break
        self._learn(model, r)
        if r.status_code != 200:
            raise classify_http(self.name, r.status_code, scrub(r.text, self.key), model, dict(r.headers))
        try:
            d = r.json()
            ch = d["choices"][0]
            text = (ch.get("message") or {}).get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError):
            raise AIError(EMPTY, "phản hồi Groq không đúng định dạng", hint="Thử lại sau.", provider=self.name)
        if not text.strip():
            raise AIError(EMPTY, f"phản hồi rỗng (finish_reason={ch.get('finish_reason', '?')})",
                          hint="Thử lại hoặc đổi model.", provider=self.name)
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S)   # model suy luận (qwen/deepseek) kèm thẻ <think>

    def _learn(self, model: str, r: requests.Response) -> None:
        """Dùng header x-ratelimit-* của Groq để chờ đúng lúc khi sắp cạn token/phút."""
        try:
            rem = int(r.headers.get("x-ratelimit-remaining-tokens", "-1"))
            reset = r.headers.get("x-ratelimit-reset-tokens", "")
            m = re.fullmatch(r"(?:(\d+)m)?([\d.]+)(ms|s)?", reset or "")
            if rem == 0 and m:
                sec = 60 * int(m.group(1) or 0) + float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
                self.limiter(model).penalize(sec + 0.2)
        except (ValueError, TypeError):
            pass


class GeminiProvider(Provider):
    name = "Gemini"
    vision = True

    def __init__(self, key: str, model: str, log):
        super().__init__(log)
        self.key = key.strip()
        m = (model or DEFAULT_GEMINI_MODEL).strip().split("/")[-1]
        self.model = DEFAULT_GEMINI_MODEL if m.startswith(RETIRED_GEMINI) or not m else m
        self.limiter = RateLimiter(12, 200000)
        self._probed = False

    def model_for(self, needs_vision: bool) -> str:
        return self.model

    def call(self, parts, json_mode, temperature, timeout, max_out, needs_vision) -> str:
        gparts = []
        for p in parts:
            if isinstance(p, str):
                gparts.append({"text": p})
            else:
                gparts.append(p)
        body: dict = {"contents": [{"parts": gparts}],
                      "generationConfig": {"temperature": temperature, "maxOutputTokens": max(max_out, 2048)}}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        for attempt in (0, 1):
            try:
                r = requests.post(f"{GEMINI_ROOT}/models/{self.model}:generateContent", json=body, timeout=timeout,
                                  headers={"x-goog-api-key": self.key, "Content-Type": "application/json"})
            except requests.RequestException as ex:
                raise classify_exception(self.name, ex, timeout)
            if r.status_code == 404 and attempt == 0 and not self._probed:
                self._probed = True
                alt = self._discover()
                if alt and alt != self.model:
                    self.log(f"Model Gemini '{self.model}' không còn → tự đổi sang '{alt}'.")
                    self.model = alt
                    continue
            break
        if r.status_code != 200:
            raise classify_http(self.name, r.status_code, scrub(r.text, self.key), self.model, dict(r.headers))
        try:
            d = r.json()
        except ValueError:
            raise AIError(EMPTY, "phản hồi không phải JSON", hint="Thử lại sau.", provider=self.name)
        fb = d.get("promptFeedback", {})
        if fb.get("blockReason"):
            raise AIError(SAFETY, f"prompt bị chặn ({fb['blockReason']})",
                          hint="Nội dung (thoại/bạo lực) bị bộ lọc an toàn chặn. Khối này sẽ dùng bản dự phòng.", provider=self.name)
        cands = d.get("candidates") or []
        if not cands:
            raise AIError(EMPTY, "không có candidate nào", hint="Thử lại hoặc đổi model.", provider=self.name)
        ps = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in ps if not p.get("thought"))
        fin = cands[0].get("finishReason", "")
        if not text.strip():
            if fin in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"):
                raise AIError(SAFETY, f"phản hồi bị chặn ({fin})", hint="Khối này sẽ dùng bản dự phòng.", provider=self.name)
            raise AIError(EMPTY, f"phản hồi rỗng (finishReason={fin or '?'})", hint="Thử lại hoặc đổi model.", provider=self.name)
        return text

    def _discover(self) -> Optional[str]:
        try:
            r = requests.get(GEMINI_ROOT + "/models?pageSize=200", headers={"x-goog-api-key": self.key}, timeout=15)
            if r.status_code != 200:
                return None
            names = [m["name"].split("/", 1)[-1] for m in r.json().get("models", [])
                     if "generateContent" in m.get("supportedGenerationMethods", [])]
        except (requests.RequestException, ValueError, KeyError):
            return None
        stable = [(float(m.group(1)), n) for n in names for m in [re.fullmatch(r"gemini-(\d+(?:\.\d+)?)-flash", n)] if m]
        if stable:
            return max(stable)[1]
        flash = [n for n in names if "flash" in n and not re.search(r"lite|image|tts|live|audio|exp|thinking", n)]
        return (flash or names or [None])[0]


def ollama_status(url: str, model: str, timeout: float = 3.0) -> tuple[bool, str]:
    """Kiểm tra Ollama đang chạy + đã tải model chưa. Trả (ok, mô_tả_hướng_dẫn)."""
    try:
        r = requests.get(url.rstrip("/") + "/api/tags", timeout=timeout)
    except requests.RequestException:
        return False, "Ollama chưa chạy. Mở ứng dụng Ollama (hoặc chạy `ollama serve`) rồi thử lại."
    if r.status_code != 200:
        return False, f"Ollama trả HTTP {r.status_code}."
    try:
        names = [m.get("name", "") for m in r.json().get("models", [])]
    except ValueError:
        names = []
    base = model.split(":")[0]
    if model in names or (":" not in model and any(n.split(":")[0] == base for n in names)):
        return True, f"Ollama sẵn sàng, đã có model {model}."
    return False, f"Ollama đang chạy nhưng chưa có model '{model}' (hiện có: {', '.join(names[:5]) or 'chưa có'}). Chạy: ollama pull {model}"


class OllamaProvider(Provider):
    name = "Ollama"

    def __init__(self, url: str, model: str, vision_model: str, timeout: float, log):
        super().__init__(log)
        self.url = (url or OLLAMA_URL).rstrip("/")
        self.model = model or DEFAULT_OLLAMA_MODEL
        self.vision_model = (vision_model or "").strip()
        self.vision = bool(self.vision_model)
        self.timeout = timeout

    def model_for(self, needs_vision: bool) -> str:
        return self.vision_model if needs_vision else self.model

    def call(self, parts, json_mode, temperature, timeout, max_out, needs_vision) -> str:
        imgs = images_of(parts)
        model = self.model_for(bool(imgs))
        if imgs and not self.vision_model:
            raise AIUnavailable("Ollama chưa có model đọc ảnh – dùng mô tả dự phòng.")
        # num_ctx lớn: mặc định 2048 sẽ CẮT MẤT prompt dài → kịch bản lặp/vô nghĩa
        body = {"model": model, "prompt": text_of(parts), "stream": False, "keep_alive": "10m",
                "options": {"temperature": max(temperature, 0.75), "top_p": 0.92, "repeat_penalty": 1.2,
                            "num_ctx": 8192, "num_predict": max(max_out, 512)}}
        if json_mode:
            body["format"] = "json"
        if imgs:
            body["images"] = [i["data"] for i in imgs]
        t = max(timeout, self.timeout)
        try:
            r = requests.post(self.url + "/api/generate", json=body, timeout=t)
        except requests.Timeout:
            raise AIError(OLLAMA, f"Ollama không trả lời sau {t:.0f}s", hint="Model quá nặng cho máy. Dùng model nhỏ hơn (vd qwen2.5:3b).", provider=self.name)
        except requests.RequestException:
            raise AIError(OLLAMA, "không kết nối được Ollama", hint="Mở ứng dụng Ollama hoặc chạy `ollama serve`.", provider=self.name)
        if r.status_code == 404:
            raise AIError(OLLAMA, f"chưa có model '{model}'", 404, f"Chạy: ollama pull {model}", provider=self.name)
        if r.status_code != 200:
            raise AIError(OLLAMA, r.text[:200], r.status_code, "Xem log của Ollama.", provider=self.name)
        try:
            out = r.json().get("response", "")
        except ValueError:
            out = ""
        if not out.strip():
            raise AIError(EMPTY, "Ollama trả về nội dung rỗng", hint="Thử model khác hoặc giảm độ dài prompt.", provider=self.name)
        return out


# ---------------------------------------------------------------- client chính
# asker(ctx) -> "ollama" | "offline" | "cancel";  ctx: stage, errors, can_ollama, needs_vision, ollama_model, ollama_msg
Asker = Callable[[dict], str]


class AIClient:
    """Một client DUY NHẤT cho cả dự án. Mỗi bước gọi ``begin_stage`` → quyết định chỉ hỏi 1 lần mỗi bước."""

    def __init__(self, groq_key: str = "", groq_model: str = DEFAULT_GROQ_MODEL, groq_vision_model: str = DEFAULT_GROQ_VISION,
                 gemini_key: str = "", gemini_model: str = DEFAULT_GEMINI_MODEL,
                 ollama_model: str = DEFAULT_OLLAMA_MODEL, ollama_url: str = OLLAMA_URL, ollama_vision_model: str = "",
                 policy: str = "ask", asker: Optional[Asker] = None, log: Optional[Callable[[str], None]] = None,
                 groq_rpm: int = 25, ollama_timeout: float = 300, max_attempts: int = 4,
                 sleep: Callable[[float], None] = time.sleep):
        self.log = log or (lambda m: None)
        self.cloud: list[Provider] = []
        if groq_key.strip():
            self.cloud.append(GroqProvider(groq_key, groq_model, groq_vision_model, self.log, groq_rpm))
        if gemini_key.strip():
            self.cloud.append(GeminiProvider(gemini_key, gemini_model, self.log))
        self.ollama = OllamaProvider(ollama_url, ollama_model, ollama_vision_model, ollama_timeout, self.log)
        self.policy, self.asker = policy, asker
        self.max_attempts, self.sleep = max_attempts, sleep
        self.job = None
        self.stage = ""
        self.mode: Optional[str] = None            # None = đám mây · "ollama" · "offline"   (áp dụng trong 1 bước)
        self.dead: dict[str, AIError] = {}          # nhà cung cấp đã hỏng trong bước này
        self.asked_count = 0                        # tổng số lần hỏi (kiểm thử: phải ≤ 1 / bước)

    # -------- vòng đời bước
    def begin_stage(self, name: str, job=None) -> None:
        """Bắt đầu bước mới: quên quyết định + danh sách hỏng của bước trước (sẽ thử đám mây lại, hỏi lại nếu lỗi)."""
        self.stage, self.mode, self.dead = name, None, {}
        if job is not None:
            self.job, self.log = job, job.log
            for p in self.cloud + [self.ollama]:
                p.log = job.log

    @property
    def has_cloud(self) -> bool:
        return bool(self.cloud)

    @property
    def usable(self) -> bool:
        return self.mode != "offline"

    @property
    def offline(self) -> bool:
        return self.mode == "offline"

    def _check(self) -> None:
        if self.job is not None:
            self.job.check()

    def _sleep(self, sec: float) -> None:
        """Ngủ theo bước 0.5 s để huỷ được ngay (job.check) – dùng self.sleep nên kiểm thử có thể giả lập."""
        left = sec
        while left > 0:
            self._check()
            step = min(0.5, left)
            self.sleep(step)
            left -= step

    # -------- một nhà cung cấp, có thử lại
    def _call_provider(self, p: Provider, parts: list, json_mode: bool, temperature: float, timeout: float,
                       max_out: int) -> str:
        needs_vision = bool(images_of(parts))
        model = p.model_for(needs_vision)
        last: Optional[AIError] = None
        for attempt in range(self.max_attempts):
            self._check()
            lim = p.limiter(model) if isinstance(p, GroqProvider) else getattr(p, "limiter", None)
            if lim is not None:
                lim.wait(estimate_tokens(parts, max_out), self.log, self._check, p.name)
            try:
                return p.call(parts, json_mode, temperature, timeout, max_out, needs_vision)
            except AIError as e:
                last = e
                if not e.retryable or attempt == self.max_attempts - 1:
                    break
                wait = e.retry_after if e.retry_after else 2.0 * (2 ** attempt)
                wait = min(65.0, wait) + random.uniform(0, 1.0)
                if lim is not None and e.kind == RATE:
                    lim.penalize(wait)
                self.log(f"{e.short} – thử lại sau {wait:.0f}s ({attempt + 2}/{self.max_attempts})…")
                self._sleep(wait)
        assert last is not None
        raise last

    # -------- hỏi người dùng (tối đa 1 lần / bước)
    def _decide(self, needs_vision: bool) -> str:
        model = self.ollama.vision_model if needs_vision and self.ollama.vision_model else self.ollama.model
        ok, msg = ollama_status(self.ollama.url, model)
        can = ok and (not needs_vision or bool(self.ollama.vision_model))
        if needs_vision and not self.ollama.vision_model:
            msg += " Bước này cần đọc ảnh: hãy đặt 'Model Ollama (ảnh)' (vd qwen2.5vl:7b, llava) để dùng Ollama."
        choice = self.policy
        if choice == "ask":
            if self.asker is None:
                choice = "offline"
            else:
                self.asked_count += 1
                choice = self.asker(dict(stage=self.stage, errors=list(self.dead.values()), can_ollama=can,
                                         needs_vision=needs_vision, ollama_model=model, ollama_msg=msg))
        if choice == "cancel":
            raise Cancelled()
        if choice == "ollama" and not can:
            self.log(f"⚠ Không dùng được Ollama: {msg}")
            choice = "offline"
        return "ollama" if choice == "ollama" else "offline"

    def generate(self, parts: list, json_mode: bool = True, temperature: float = 0.8, timeout: float = 60,
                 max_out: int = 1500) -> str:
        parts = [{"text": p} if isinstance(p, str) else p for p in parts]
        needs_vision = bool(images_of(parts))
        if self.mode == "offline":
            raise AIUnavailable()
        if self.mode == "ollama":
            try:
                return self.ollama.call(parts, json_mode, temperature, timeout, max_out, needs_vision)
            except AIUnavailable:
                raise
            except AIError as e:
                if e.kind in (SAFETY, EMPTY):
                    raise
                self.log(f"Ollama lỗi: {e} → dùng dự phòng cho phần còn lại của bước này.")
                self.mode = "offline"
                raise AIUnavailable()
        # --- đám mây: lần lượt Groq → Gemini, bỏ qua nhà cung cấp đã hỏng trong bước này
        for p in self.cloud:
            if p.name in self.dead or (needs_vision and not p.vision):
                continue
            try:
                return self._call_provider(p, parts, json_mode, temperature, timeout, max_out)
            except AIError as e:
                if not e.service_level:                 # lỗi riêng của yêu cầu (an toàn/rỗng/quá lớn) → để nơi gọi xử lý
                    raise
                self.dead[p.name] = e
                nxt = next((q.name for q in self.cloud if q.name not in self.dead and (q.vision or not needs_vision)), None)
                self.log(f"✖ {e.short} ở '{self.stage or 'bước này'}': {e}")
                self.log(f"→ {p.name} bị bỏ qua cho phần còn lại của bước này" + (f"; chuyển sang {nxt}." if nxt else "."))
        # --- hết đám mây
        if not self.has_cloud:
            self.log("Chưa cấu hình key Groq/Gemini.")
        self.mode = self._decide(needs_vision)
        if self.mode == "offline":
            self.log("→ Dùng chế độ dự phòng cho phần còn lại của bước này (bước sau sẽ thử Groq/Gemini lại).")
            raise AIUnavailable()
        self.log(f"→ Chuyển sang Ollama ({self.ollama.model_for(needs_vision)}) cho phần còn lại của bước này.")
        return self.generate(parts, json_mode, temperature, timeout, max_out)

    # -------- chẩn đoán từng nhà cung cấp (nút "Kiểm tra kết nối")
    def diagnose(self) -> list[tuple[str, bool, str]]:
        out: list[tuple[str, bool, str]] = []
        saved = self.max_attempts
        self.max_attempts = 1
        try:
            if not self.cloud:
                out.append(("Groq / Gemini", False, "Chưa nhập key nào."))
            for p in self.cloud:
                try:
                    lim = p.limiter(p.model_for(False)) if isinstance(p, GroqProvider) else None
                    if lim:
                        lim.wait(50)
                    p.call([{"text": "Trả lời đúng một từ: OK"}], False, 0.0, 20, 16, False)
                    out.append((p.name, True, f"hoạt động tốt (model {p.model_for(False)})."))
                except AIError as e:
                    out.append((p.name, False, str(e)))
        finally:
            self.max_attempts = saved
        ok, msg = ollama_status(self.ollama.url, self.ollama.model)
        out.append(("Ollama", ok, msg))
        return out
