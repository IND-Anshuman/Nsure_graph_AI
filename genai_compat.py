"""Compatibility wrapper for Google's GenAI Python SDKs.

This repo historically mixed two different packages:
- google-generativeai  -> imported as `import google.generativeai as genai` (has genai.configure + genai.GenerativeModel)
- google-genai         -> imported as `from google import genai` or `import google.genai` (uses Client().models.generate_content)

This module provides a tiny stable surface used by the rest of the codebase.
"""

from __future__ import annotations

from typing import Optional
import os


# Prefer the older SDK if present since much of the project was written for it.
try:  # google-generativeai
    import google.generativeai as _gai  # type: ignore
except Exception:  # pragma: no cover
    _gai = None  # type: ignore

try:  # google-genai (new)
    # NOTE: Some environments expose this as `from google import genai`.
    from google import genai as _genai  # type: ignore
except Exception:  # pragma: no cover
    _genai = None  # type: ignore


_client = None
_configured = False


def is_available() -> bool:
    return _gai is not None or _genai is not None


def init(api_key: Optional[str] = None) -> None:
    """Best-effort initialization. Safe to call multiple times."""
    global _client, _configured

    if _configured:
        return

    key = (api_key or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        _configured = True
        return

    if _gai is not None:
        try:
            _gai.configure(api_key=key)  # type: ignore[attr-defined]
        except Exception:
            pass
        _configured = True
        return

    if _genai is not None:
        try:
            _client = _genai.Client(api_key=key)  # type: ignore[attr-defined]
        except Exception:
            _client = None
        _configured = True
        return

    _configured = True


def generate_text(model: str, prompt: str, *, temperature: float = 0.1) -> str:
    """Generate text from Gemini using whichever SDK is installed.

    Returns an empty string on any error (callers generally have fallbacks).
    """
    init()

    if not prompt:
        return ""

    if _gai is not None:
        try:
            m = _gai.GenerativeModel(model)  # type: ignore[attr-defined]
            resp = m.generate_content(prompt, generation_config={"temperature": float(temperature)})
            return (getattr(resp, "text", "") or "").strip()
        except Exception:
            return ""

    if _genai is not None:
        global _client
        if _client is None:
            init()
        if _client is None:
            return ""
        try:
            resp = _client.models.generate_content(  # type: ignore[union-attr]
                model=model,
                contents=prompt,
                config={"temperature": float(temperature)},
            )
            return (getattr(resp, "text", "") or "").strip()
        except Exception:
            return ""

    return ""
