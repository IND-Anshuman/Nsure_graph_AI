# cache_utils.py
from __future__ import annotations
import json
import hashlib
import threading
import os
from pathlib import Path
from typing import Any, Optional


class DiskJSONCache:
    """
    Simple thread-safe JSON cache.
    Stores a dict[str, Any] on disk and keeps it in memory.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.Lock()
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def get(self, key: str) -> Optional[Any]:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            tmp = self.path.with_suffix(".tmp")
            try:
                with tmp.open("w", encoding="utf-8") as f:
                    json.dump(self._data, f, ensure_ascii=False)
                # Windows-compatible file replacement
                if os.name == 'nt':  # Windows
                    if self.path.exists():
                        self.path.unlink()  # Delete the original file first
                    tmp.rename(self.path)
                else:
                    tmp.replace(self.path)
            except Exception as e:
                # Clean up tmp file if replacement failed
                if tmp.exists():
                    tmp.unlink()
                # Don't raise - cache write failures shouldn't break the pipeline
                print(f"[WARN] Cache write failed: {e}")

    @staticmethod
    def hash_key(*parts: str) -> str:
        """
        Create a stable SHA256 key from arbitrary string parts.
        Useful when prompts are long.
        """
        raw = "||".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
