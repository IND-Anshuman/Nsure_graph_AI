"""Compatibility wrapper for Google's GenAI Python SDK.

This module uses the google-genai package:
- google-genai -> imported as `from google import genai` (uses Client().models.generate_content)

This module provides a stable surface used by the rest of the codebase.
"""

from __future__ import annotations

import os
import time
import threading
import random
from typing import Optional, List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Use the new google-genai SDK
try:
    from google import genai as _genai  # type: ignore
except Exception:  # pragma: no cover
    _genai = None  # type: ignore


class RateLimiter:
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
                    print(f"[GenAI] [{self.key[:6]}...] Cooldown: sleeping {sleep_time:.2f}s")
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
                    # Add jitter to prevent multiple workers from bursting at the start of a window
                    jitter = random.uniform(0.1, 1.5)
                    time.sleep(min(sleep_time + jitter, 2.0))

    def trigger_cooldown(self, duration: float = 40.0):
        with self.lock:
            self.cooldown_until = time.time() + duration


# Global registry to share clients/limiters across pools if they use the same key
_client_registry: Dict[str, tuple[_genai.Client, RateLimiter]] = {}
_registry_lock = threading.Lock()


class ClientPool:
    def __init__(self, name: str, keys: List[str]):
        self.name = name
        self.clients: List[_genai.Client] = []
        self.dead_keys: set = set()
        self.index = 0
        self.lock = threading.Lock()
        self.rate_limiters: Dict[_genai.Client, RateLimiter] = {} # Per-client rate limiter
        
        # Free tier is usually 15 RPM. Paid tier varies. We'll stick to 15 for safety.
        rpm = int(os.getenv("GEMINI_RPM", "15") or 15)

        for key in keys:
            try:
                with _registry_lock:
                    if key in _client_registry:
                        client, limiter = _client_registry[key]
                    else:
                        client = _genai.Client(api_key=key)
                        limiter = RateLimiter(key, max_rpm=rpm)
                        _client_registry[key] = (client, limiter)
                
                self.clients.append(client)
                self.rate_limiters[client] = limiter
            except Exception as e:
                print(f"[GenAI] [{name}] Failed to initialize client for key {key[:8]}...: {e}")
        
        if self.clients:
            print(f"[GenAI] [{name}] Initialized {len(self.clients)} API clients.")

    def get_next_client(self) -> Optional[tuple[_genai.Client, RateLimiter]]:
        with self.lock:
            if not self.clients:
                return None
            
            client = self.clients[self.index]
            self.index = (self.index + 1) % len(self.clients)
            return client, self.rate_limiters[client]

    def mark_dead(self, client: _genai.Client):
        with self.lock:
            if client in self.clients:
                print(f"[GenAI] [{self.name}] Removing dead/expired client from pool.")
                self.clients.remove(client)
                # Note: We keep it in the registry but remove from this pool
                if client in self.rate_limiters:
                    del self.rate_limiters[client]
                if self.clients:
                    self.index = self.index % len(self.clients)
                else:
                    self.index = 0


_pools: Dict[str, ClientPool] = {}
_configured = False


def is_available() -> bool:
    return _genai is not None


def _parse_keys(env_var: str) -> List[str]:
    val = (os.getenv(env_var) or "").strip()
    return [k.strip() for k in val.split(",") if k.strip()]


def init(api_key: Optional[str] = None) -> None:
    """Best-effort initialization of task-specific pools."""
    global _pools, _configured

    if _configured:
        return

    if _genai is not None:
        # 1. Initialize specific pools
        _pools["ENTITY"] = ClientPool("ENTITY", _parse_keys("GOOGLE_API_KEY_ENTITY"))
        _pools["RELATION"] = ClientPool("RELATION", _parse_keys("GOOGLE_API_KEY_RELATION"))
        _pools["QA"] = ClientPool("QA", _parse_keys("GOOGLE_API_KEY_QA"))
        
        # 2. General pool (fallback)
        general_keys = _parse_keys("GOOGLE_API_KEY")
        if api_key:
            general_keys.insert(0, api_key)
        _pools["GENERAL"] = ClientPool("GENERAL", list(dict.fromkeys(general_keys))) # unique
        
        _configured = True
        return

    _configured = True


def _get_client_from_purpose(purpose: str) -> Optional[tuple[ClientPool, _genai.Client, RateLimiter]]:
    pool = _pools.get(purpose.upper())
    if pool:
        res = pool.get_next_client()
        if res:
            return pool, res[0], res[1]
    
    # Fallback to general
    gen_pool = _pools.get("GENERAL")
    if gen_pool:
        res = gen_pool.get_next_client()
        if res:
            return gen_pool, res[0], res[1]
            
    return None


@retry(
    wait=wait_exponential(multiplier=2, min=3, max=40),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda rs: print(f"[GenAI] Retry {rs.attempt_number} after error: {rs.outcome.exception()}"),
    reraise=True
)
def _generate_text_inner(model, prompt, temperature, purpose):
    res = _get_client_from_purpose(purpose)
    if res is None:
        raise ValueError(f"No API clients available for purpose: {purpose}")
    
    pool, client, limiter = res
    
    # Allow purpose-specific RPM override (e.g. GEMINI_RPM_RELATION=5)
    override_rpm = os.getenv(f"GEMINI_RPM_{purpose.upper()}")
    purpose_rpm = int(override_rpm) if override_rpm else None
    
    # Ensure RPM safety
    limiter.wait_if_needed(purpose_rpm=purpose_rpm)
    
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": float(temperature),
                "max_output_tokens": 65536,
            },
        )
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        err_msg = str(e).upper()
        
        # Handle specifically 429
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            print(f"[GenAI] [{pool.name}] 429 detected. Triggering 40s cooldown for key.")
            limiter.trigger_cooldown(40.0)
            
        # If key is expired or invalid, remove it from the pool permanently for this session
        if "EXPIRED" in err_msg or "INVALID_ARGUMENT" in err_msg or "API_KEY_INVALID" in err_msg:
            print(f"[GenAI] [{pool.name}] Automatically removing invalid/expired key.")
            pool.mark_dead(client)
        raise e


def generate_text(model: Optional[str] = None, prompt: str = "", *, temperature: float = 0.1, purpose: str = "GENERAL") -> str:
    """Generate text from Gemini using task-specific pools.

    Purposes: ENTITY, RELATION, QA, GENERAL
    Returns an empty string on any error after retries fail.
    """
    init()
    
    # Use model from parameter, environment variable, or default
    if not model:
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

    if not prompt:
        return ""

    if _genai is not None:
        try:
            # Wrap the actual call in a retrying inner function
            return _generate_text_inner(model, prompt, temperature, purpose)
        except Exception as e:
            print(f"[GenAI] All retry attempts failed for purpose {purpose}. Final error: {e}")
            return ""

    return ""
