"""Client Gemini REST tối giản (văn bản + ảnh)."""
from __future__ import annotations

import base64
import json
import re
import time

import requests

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiError(RuntimeError):
    pass


class Gemini:
    def __init__(self, key: str, model: str = "gemini-2.5-flash"):
        self.key, self.model = key.strip(), model.strip()

    def generate(self, parts: list, json_mode: bool = True, temperature: float = 0.8, timeout: int = 150) -> str:
        body = {"contents": [{"parts": parts}],
                "generationConfig": {"temperature": temperature}}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        last = ""
        for attempt in range(4):
            try:
                r = requests.post(API.format(model=self.model), json=body, timeout=timeout,
                                  headers={"x-goog-api-key": self.key, "Content-Type": "application/json"})
            except requests.RequestException as ex:
                last = f"lỗi mạng: {type(ex).__name__}"
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 200:
                try:
                    cand = r.json()["candidates"][0]["content"]["parts"]
                    return "".join(p.get("text", "") for p in cand)
                except (KeyError, IndexError, ValueError):
                    raise GeminiError("Gemini trả về nội dung rỗng (có thể bị bộ lọc an toàn chặn).")
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                time.sleep(2.0 * (attempt + 1))
                continue
            msg = ""
            try:
                msg = r.json().get("error", {}).get("message", "")
            except ValueError:
                pass
            raise GeminiError(f"Gemini HTTP {r.status_code}: {msg[:200]}")
        raise GeminiError(f"Gemini không phản hồi ({last}).")

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
