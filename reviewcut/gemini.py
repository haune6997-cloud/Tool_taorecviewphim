"""Client Gemini REST tối giản (văn bản + ảnh) kèm cơ chế dự phòng Ollama Local."""
from __future__ import annotations

import base64
import json
import re
import time
from typing import Optional

import requests

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
OLLAMA_API = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"


class GeminiError(RuntimeError):
    pass


def _extract_text(parts: list) -> str:
    """Trích xuất toàn bộ văn bản thuần từ danh sách parts (bỏ qua ảnh inline)."""
    texts = []
    for p in parts:
        if isinstance(p, str):
            texts.append(p)
        elif isinstance(p, dict) and "text" in p:
            texts.append(p["text"])
    return "\n".join(texts)


def query_ollama(prompt: str, model: str = DEFAULT_OLLAMA_MODEL, json_mode: bool = True, timeout: int = 120) -> str:
    """Gửi yêu cầu sinh nội dung tới Ollama chạy nội bộ trên máy."""
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    if json_mode:
        body["format"] = "json"

    try:
        r = requests.post(OLLAMA_API, json=body, timeout=timeout)
        if r.status_code == 200:
            return r.json().get("response", "")
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        raise RuntimeError(f"Không thể kết nối với Ollama ({e}). Hãy đảm bảo Ollama đang chạy.")


def _ask_ollama_fallback(error_detail: str) -> bool:
    """Bật hộp thoại hỏi người dùng có muốn chuyển sang mô hình Local không."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        res = messagebox.askyesno(
            "Gemini AI không phản hồi",
            f"Gemini gặp sự cố ({error_detail}).\n\n"
            f"Bạn có muốn chuyển sang chạy mô hình nội bộ Ollama ({DEFAULT_OLLAMA_MODEL}) để tiếp tục không?"
        )
        root.destroy()
        return bool(res)
    except Exception:
        return False


class Gemini:
    def __init__(self, key: str, model: str = "gemini-1.5-flash", ollama_model: str = DEFAULT_OLLAMA_MODEL):
        self.key = key.strip()
        self.model = model.strip() if model.strip() else "gemini-1.5-flash"
        self.ollama_model = ollama_model

    def generate(self, parts: list, json_mode: bool = True, temperature: float = 0.8, timeout: int = 35) -> str:
        body = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": temperature}
        }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        last_err = ""
        # Thử gọi Gemini tối đa 2 lần với timeout 35s
        for attempt in range(2):
            try:
                r = requests.post(
                    API.format(model=self.model),
                    json=body,
                    timeout=timeout,
                    headers={"x-goog-api-key": self.key, "Content-Type": "application/json"}
                )
            except requests.RequestException as ex:
                last_err = f"lỗi mạng: {type(ex).__name__}"
                time.sleep(1.5 * (attempt + 1))
                continue

            if r.status_code == 200:
                try:
                    cand = r.json()["candidates"][0]["content"]["parts"]
                    return "".join(p.get("text", "") for p in cand)
                except (KeyError, IndexError, ValueError):
                    raise GeminiError("Gemini trả về nội dung rỗng (có thể bị bộ lọc an toàn chặn).")

            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}"
                time.sleep(2.0 * (attempt + 1))
                continue

            msg = ""
            try:
                msg = r.json().get("error", {}).get("message", "")
            except ValueError:
                pass
            last_err = f"HTTP {r.status_code}: {msg[:200]}"
            break

        # Nếu Gemini thất bại, bật popup hỏi người dùng chuyển sang Ollama
        err_summary = last_err or "Hết thời gian phản hồi"
        if _ask_ollama_fallback(err_summary):
            try:
                prompt_text = _extract_text(parts)
                if not prompt_text.strip():
                    raise RuntimeError("Dữ liệu chỉ có ảnh, không có văn bản prompt để Ollama xử lý.")
                return query_ollama(prompt_text, model=self.ollama_model, json_mode=json_mode)
            except Exception as o_err:
                raise GeminiError(f"Gemini lỗi ({err_summary}) & Ollama lỗi ({o_err})")

        raise GeminiError(f"Gemini không phản hồi ({err_summary}).")

    @staticmethod
    def image_part(jpg_path: str) -> dict:
        return {"inline_data": {"mime_type": "image/jpeg",
                                "data": base64.b64encode(open(jpg_path, "rb").read()).decode()}}


def parse_json(text: str):
    """Chịu được ```json fence hoặc văn bản thừa quanh JSON."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    try:
        return json.loads(t)
    except ValueError:
        m = re.search(r"(\[.*\]|\{.*\})", t, flags=re.S)
        if m:
            return json.loads(m.group(1))
        raise