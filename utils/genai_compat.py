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


# --- Global Rate Limiter Registry ---
# This ensures that if the same API key is used in multiple pools (ENTITY, QA, etc.),
# they all share the same RPM state and cooldowns.
_LIMITER_REGISTRY: Dict[str, RateLimiter] = {}
_REGISTRY_LOCK = threading.Lock()

def get_limiter(key: str, max_rpm: int = 15) -> RateLimiter:
    with _REGISTRY_LOCK:
        if key not in _LIMITER_REGISTRY:
            _LIMITER_REGISTRY[key] = RateLimiter(key, max_rpm=max_rpm)
        return _LIMITER_REGISTRY[key]


class UnifiedClient:
    """Wraps a provider client with its specific rate limiter."""
    def __init__(self, provider: str, client: Any, limiter: RateLimiter):
        self.provider = provider.upper()
        self.client = client
        self.limiter = limiter

    def generate(self, model: str, prompt: str, temperature: float) -> str:
        if self.provider == "GEMINI":
            # 400 Bad Request often happens with Experimental models or max_output_tokens being too high.
            # 65536 is safe for 1.5 Pro/Flash, but for 2.0 Flash we might want to be conservative.
            # We also explicitly disable AFC logic to stop the "AFC is enabled" logs and potential errors.
            # Explicitly disable safety filters to prevent rejections on legal terminology (liability, injury, etc.)
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            try:
                resp = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "temperature": float(temperature),
                        "max_output_tokens": 16384 if "2.0-flash" in model else 64000,
                        "automatic_function_calling": {"disable": True},
                        "safety_settings": safety_settings
                    },
                )
                return (getattr(resp, "text", "") or "").strip()
            except Exception as e:
                err_str = str(e).upper()
                if "400" in err_str:
                    logging.warning(f"[GenAI] 400 Bad Request for {model}. Retrying with relaxed config.")
                    resp = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config={
                            "temperature": float(temperature),
                            "max_output_tokens": 4096,
                            "automatic_function_calling": {"disable": True},
                            "safety_settings": safety_settings
                        },
                    )
                    return (getattr(resp, "text", "") or "").strip()
                raise e
        raise ValueError(f"Unknown provider or unsupported: {self.provider}")

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        if self.provider == "GEMINI":
            resp = self.client.models.embed_content(
                model=model,
                contents=texts,
                config={"task_type": "RETRIEVAL_DOCUMENT"}
            )
            return [getattr(e, "values", []) for e in resp.embeddings]
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
    gemini_rpm = int(os.getenv("GEMINI_RPM", "15"))
    
    # 1. Load General Keys
    for key in _parse_keys("GOOGLE_API_KEY"):
        if _genai:
            try:
                client = _genai.Client(api_key=key)
                limiter = get_limiter(key, max_rpm=gemini_rpm)
                u_client = UnifiedClient("GEMINI", client, limiter)
                _pools["GENERAL"].add_client(u_client)
            except Exception as e:
                logging.error(f"[GenAI] Failed to init GENERAL client: {e}")

    # 2. Load Task-Specific Keys
    for p in ["ENTITY", "RELATION", "QA"]:
        env_var = f"GOOGLE_API_KEY_{p}"
        purpose_keys = _parse_keys(env_var)
        if purpose_keys:
            # Use specific RPM if provided, else fall back to general
            p_rpm = int(os.getenv(f"GEMINI_RPM_{p}", str(gemini_rpm)))
            for key in purpose_keys:
                try:
                    client = _genai.Client(api_key=key)
                    limiter = get_limiter(key, max_rpm=p_rpm)
                    _pools[p].add_client(UnifiedClient("GEMINI", client, limiter))
                except Exception as e:
                    logging.error(f"[GenAI] Failed to init {p} client: {e}")
        else:
            # Fallback: link GENERAL clients to this pool if no specific keys exist
            with _pools[p].lock:
                _pools[p].clients = _pools["GENERAL"].clients

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

@retry(
    wait=wait_exponential(multiplier=2, min=3, max=40),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def _embed_inner(model: str, texts: List[str]):
    init()
    pool = _pools["GENERAL"]
    client = pool.get_next()
    if not client:
        raise ValueError("No Gemini API keys configured for embedding")
    
    # Simple rate limiting for embeddings
    client.limiter.wait_if_needed()
    
    try:
        return client.embed(model, texts)
    except Exception as e:
        err = str(e).upper()
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            logging.warning(f"[GenAI-Embed] 429 detected. Throttling.")
            client.limiter.trigger_cooldown(60.0)
        raise e

def embed_texts(model: Optional[str] = None, texts: List[str] = []) -> List[List[float]]:
    """Unified text embedding using Gemini."""
    if not texts: return []
    if not model:
        model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
        
    try:
        # Gemini allows batching up to 100 texts usually.
        # We'll rely on the caller or provide a small internal chunking if needed.
        return _embed_inner(model, texts)
    except Exception as e:
        logging.error(f"[GenAI-Embed] Final failure: {e}")
        return []
