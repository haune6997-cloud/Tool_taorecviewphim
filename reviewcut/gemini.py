"""Bộ điều phối AI tự động 3 tầng: Groq (Số 1) -> Gemini (Số 2) -> Ollama Offline (Dự phòng)."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


class GeminiError(Exception):
    pass


def parse_json(raw: str) -> Any:
    text = raw.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_b, end_b = text.find("["), text.rfind("]")
        start_c, end_c = text.find("{"), text.rfind("}")
        if start_b != -1 and end_b != -1 and (start_c == -1 or start_b < start_c):
            return json.loads(text[start_b : end_b + 1])
        if start_c != -1 and end_c != -1:
            return json.loads(text[start_c : end_c + 1])
        raise


class Gemini:
    def __init__(self, api_key: str = "", model: str = ""):
        # Tự động nạp cấu hình từ config.json để luôn có đủ key của cả Groq và Gemini
        cfg = {}
        cfg_path = Path.home() / ".reviewcut" / "config.json"
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        self.groq_key = (cfg.get("groq_key") or "").strip()
        self.groq_model = (cfg.get("groq_model") or "llama-3.3-70b-versatile").strip()
        self.gemini_key = (api_key or cfg.get("gemini_key") or "").strip()
        self.gemini_model = (model or cfg.get("gemini_model") or "gemini-1.5-flash").strip()

        # Kiểm tra nếu key truyền trực tiếp vào là key Groq
        if self.gemini_key.startswith("gsk_"):
            self.groq_key = self.gemini_key
            self.gemini_key = cfg.get("gemini_key", "").strip()

    def generate(self, contents: list[dict[str, Any]]) -> str:
        prompt_parts = []
        for c in contents:
            if "text" in c:
                prompt_parts.append(str(c["text"]))
            elif "parts" in c:
                for p in c["parts"]:
                    if isinstance(p, dict) and "text" in p:
                        prompt_parts.append(str(p["text"]))
        full_prompt = "\n".join(prompt_parts).strip()

        # ------------------ TẦNG 1: ƯU TIÊN SỐ 1 - GROQ API ------------------
        if self.groq_key and self.groq_key.startswith("gsk_"):
            try:
                return self._call_groq(full_prompt)
            except Exception as ex:
                print(f"[AI Cascade] Groq gặp lỗi ({ex}) -> Tự động chuyển xuống Gemini...")

        # ------------------ TẦNG 2: ƯU TIÊN SỐ 2 - GOOGLE GEMINI -------------
        if self.gemini_key and not self.gemini_key.lower().startswith("ollama"):
            try:
                return self._call_gemini(full_prompt)
            except Exception as ex:
                print(f"[AI Cascade] Gemini gặp lỗi ({ex}) -> Tự động chuyển xuống Ollama...")

        # ------------------ TẦNG 3: PHAO CỨU SINH - OLLAMA OFFLINE -----------
        print("[AI Cascade] Chạy kịch bản bằng Ollama Offline (qwen2.5:7b)...")
        return self._call_ollama(full_prompt)

    def _call_groq(self, prompt: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": self.groq_model or "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.groq_key}",
                "Content-Type": "application/json",
                "User-Agent": "ReviewCut/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            raise GeminiError(f"Groq HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}")
        except Exception as e:
            raise GeminiError(f"Lỗi mạng Groq: {e}")

    def _call_gemini(self, prompt: str) -> str:
        model = self.gemini_model or "gemini-1.5-flash"
        if "/" in model:
            model = model.split("/")[-1]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gemini_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            raise GeminiError(f"Gemini HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}")
        except Exception as e:
            raise GeminiError(f"Lỗi mạng Gemini: {e}")

    def _call_ollama(self, prompt: str) -> str:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": 4096, "temperature": 0.7},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except Exception as e:
            raise GeminiError(
                f"Không thể kết nối Ollama (http://localhost:11434). Đảm bảo Ollama đã được bật. Lỗi: {e}"
            )