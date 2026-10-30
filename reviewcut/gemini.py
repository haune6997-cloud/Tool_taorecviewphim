"""Tương thích ngược: mã cũ import từ đây. Toàn bộ logic AI nay nằm ở ``reviewcut.ai``."""
from .ai import (AIClient as Gemini, AIError as GeminiError, AIUnavailable, image_part, parse_json,  # noqa: F401
                 classify_http, classify_exception, ollama_status, RateLimiter)

Gemini.image_part = staticmethod(image_part)      # type: ignore[attr-defined]
