"""Compatibility wrapper for Google's GenAI Python SDK.

This module provides a unified interface for Gemini with API pooling, 
rate limiting (RPM), and purpose-based routing (ENTITY, RELATION, QA).
"""

from __future__ import annotations

import os
import time
import threading
import random
import logging
import re
from typing import Optional, List, Dict, Any, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Provider SDK Imports ---
try:
    from google import genai as _genai  # type: ignore
except ImportError:
    _genai = None


class RateLimiter:
    """Manages RPM (Requests Per Minute) for an API key."""
    def __init__(self, key: str, max_rpm: int = 15):
        self.key = key
        self.max_rpm = max_rpm
        self.request_times: List[float] = []
        self.cooldown_until: float = 0
        self.lock = threading.Lock()

    def wait_if_needed(self, purpose_rpm: Optional[int] = None):
        rpm = purpose_rpm if purpose_rpm is not None else self.max_rpm
        with self.lock:
            while True:
                now = time.time()
                
                # 1. Global cooldown check (from 429)
                if now < self.cooldown_until:
                    sleep_time = self.cooldown_until - now
                    logging.warning(f"[RateLimit] Key {self.key[:6]} cooldown: sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                    continue
                
                # 2. RPM sliding window check
                self.request_times = [t for t in self.request_times if now - t < 60]
                
                if len(self.request_times) < rpm:
                    self.request_times.append(now)
                    return
                
                # Wait for nearest window opening
                sleep_time = 60.1 - (now - self.request_times[0])
                if sleep_time > 0:
                    jitter = random.uniform(0.5, 2.5)
                    time.sleep(sleep_time + jitter)

    def trigger_cooldown(self, duration: float = 60.0):
        with self.lock:
            new_cooldown = time.time() + duration
            if new_cooldown > self.cooldown_until:
                self.cooldown_until = new_cooldown


class UnifiedClient:
    """Wraps a provider client with its specific rate limiter."""
    def __init__(self, provider: str, client: Any, limiter: RateLimiter):
        self.provider = provider.upper()
        self.client = client
        self.limiter = limiter

    def generate(self, model: str, prompt: str, temperature: float) -> str:
        if self.provider == "GEMINI":
            resp = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": float(temperature),
                    "max_output_tokens": 65536,
                },
            )
            return (getattr(resp, "text", "") or "").strip()
        raise ValueError(f"Unknown provider or unsupported: {self.provider}")


class MultiPool:
    def __init__(self, name: str):
        self.name = name
        self.clients: List[UnifiedClient] = []
        self.index = 0
        self.lock = threading.Lock()

    def add_client(self, client: UnifiedClient):
        with self.lock:
            self.clients.append(client)

    def get_next(self) -> Optional[UnifiedClient]:
        with self.lock:
            if not self.clients:
                return None
            client = self.clients[self.index]
            self.index = (self.index + 1) % len(self.clients)
            return client


# Global Registry
_pools: Dict[str, MultiPool] = {}
_configured = False

def _parse_keys(env_var: str) -> List[str]:
    val = (os.getenv(env_var) or "").strip()
    val = re.sub(r"#.*$", "", val).strip() # Remove inline comments
    return [k.strip() for k in val.split(",") if k.strip()]

def init():
    global _configured, _pools
    if _configured: return
    
    purposes = ["ENTITY", "RELATION", "QA", "GENERAL"]
    for p in purposes:
        _pools[p] = MultiPool(p)

    # Load Gemini Keys
    gemini_keys = _parse_keys("GOOGLE_API_KEY")
    gemini_rpm = int(os.getenv("GEMINI_RPM", "15"))
    for key in gemini_keys:
        if _genai:
            try:
                client = _genai.Client(api_key=key)
                limiter = RateLimiter(key, max_rpm=gemini_rpm)
                u_client = UnifiedClient("GEMINI", client, limiter)
                _pools["GENERAL"].add_client(u_client)
                # Add to specific pools if configured in env
                if os.getenv("GOOGLE_API_KEY_ENTITY"): _pools["ENTITY"].add_client(u_client)
                if os.getenv("GOOGLE_API_KEY_RELATION"): _pools["RELATION"].add_client(u_client)
                if os.getenv("GOOGLE_API_KEY_QA"): _pools["QA"].add_client(u_client)
            except Exception as e:
                logging.error(f"[GenAI] Failed to init client for key {key[:6]}: {e}")

    _configured = True

def is_available() -> bool:
    init()
    return any(p.clients for p in _pools.values())

def is_gemini_available() -> bool:
    return is_available()

@retry(
    wait=wait_exponential(multiplier=2, min=3, max=40),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def _generate_inner(model, prompt, temperature, purpose):
    pool = _pools.get(purpose.upper()) or _pools["GENERAL"]
    client = pool.get_next()
    
    if not client:
        raise ValueError(f"No Gemini API keys configured for {purpose}")

    # Allow RPM override for specific purpose
    override_rpm = os.getenv(f"GEMINI_RPM_{purpose.upper()}")
    purpose_rpm = int(override_rpm) if override_rpm else None
    
    client.limiter.wait_if_needed(purpose_rpm=purpose_rpm)
    
    try:
        return client.generate(model, prompt, temperature)
    except Exception as e:
        err = str(e).upper()
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            logging.warning(f"[GenAI] 429 detected. Throttling for 60s.")
            client.limiter.trigger_cooldown(60.0)
        raise e

def generate_text(model: Optional[str] = None, prompt: str = "", *, temperature: float = 0.1, purpose: str = "GENERAL") -> str:
    """Unified text generation using Gemini."""
    init()
    if not prompt: return ""
    
    if not model:
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    try:
        return _generate_inner(model, prompt, temperature, purpose)
    except Exception as e:
        logging.error(f"[GenAI] Final failure after retries: {e}")
        return ""
