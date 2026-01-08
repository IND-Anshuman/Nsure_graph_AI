"""Compatibility wrapper for Google's GenAI Python SDK.

This module uses the google-genai package:
- google-genai -> imported as `from google import genai` (uses Client().models.generate_content)

This module provides a stable surface used by the rest of the codebase.
"""

from __future__ import annotations

from typing import Optional
import os


# Use the new google-genai SDK
try:
    from google import genai as _genai  # type: ignore
except Exception:  # pragma: no cover
    _genai = None  # type: ignore


_client = None
_configured = False


def is_available() -> bool:
    return _genai is not None


def init(api_key: Optional[str] = None) -> None:
    """Best-effort initialization. Safe to call multiple times."""
    global _client, _configured

    if _configured:
        return

    key = (api_key or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
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
        except Exception as e:
            print(f"[GenAI] google-genai error: {e}")
            return ""

    return ""
