"""Client Gemini REST tối giản (văn bản + ảnh) kèm cơ chế dự phòng Ollama Local thông minh."""
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
            f"Bạn có muốn chuyển TẤT CẢ các phần còn lại sang chạy Ollama ({DEFAULT_OLLAMA_MODEL}) không?\n"
            "(Hệ thống sẽ ghi nhớ và tự động xử lý, không hỏi lại nữa)"
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

        # Biến trạng thái: Ghi nhớ quyết định chuyển đổi của người dùng
        self.use_ollama_only = False

    def generate(self, parts: list, json_mode: bool = True, temperature: float = 0.8, timeout: int = 35) -> str:
        has_image = any(isinstance(p, dict) and "inline_data" in p for p in parts)

        # 1. NẾU ĐÃ ĐỒNG Ý CHUYỂN OLLAMA TỪ TRƯỚC -> Bỏ qua Gemini, gọi thẳng Ollama
        if self.use_ollama_only and not has_image:
            prompt_text = _extract_text(parts)
            if prompt_text.strip():
                return query_ollama(prompt_text, model=self.ollama_model, json_mode=json_mode)

        # 2. Quy trình gửi API lên Gemini bình thường
        body = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": temperature}
        }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        last_err = ""
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

        # 3. KHI GEMINI LỖI: Hỏi người dùng (nếu trước đó chưa hỏi)
        err_summary = last_err or "Hết thời gian phản hồi"

        if not has_image and _ask_ollama_fallback(err_summary):
            # CHỐT QUYẾT ĐỊNH: Bật cờ ghi nhớ để các vòng lặp sau tự động chạy Ollama
            self.use_ollama_only = True
            try:
                prompt_text = _extract_text(parts)
                if prompt_text.strip():
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