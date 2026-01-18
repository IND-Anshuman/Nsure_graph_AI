# embedding_cache.py
"""
Centralized embedding cache module.
Reuses the existing DiskJSONCache infrastructure for storing and retrieving embeddings.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
import threading
import sqlite3
import time
import logging
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
from utils.cache_utils import DiskJSONCache
import hashlib
import os

# Global instances
_EMB_MODEL: Optional["SentenceTransformer"] = None  # type: ignore[name-defined]
_JSON_CACHE = DiskJSONCache("cache_embeddings.json")


class _SQLiteKV:
    """Tiny sqlite-backed KV store for embeddings.

    Stores float32 vectors as raw bytes.
    This avoids rewriting a large JSON file on every insert.
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, dim INTEGER NOT NULL, v BLOB NOT NULL)"
        )
        self._conn.commit()

    def get_many(self, keys: List[str]) -> dict[str, np.ndarray]:
        if not keys:
            return {}
        out: dict[str, np.ndarray] = {}
        # Chunk to avoid sqlite's max variable count.
        chunk = 500
        with self._lock:
            for i in range(0, len(keys), chunk):
                ks = keys[i : i + chunk]
                qmarks = ",".join(["?"] * len(ks))
                cur = self._conn.execute(f"SELECT k, dim, v FROM kv WHERE k IN ({qmarks})", ks)
                for k, dim, blob in cur.fetchall():
                    arr = np.frombuffer(blob, dtype=np.float32)
                    if int(dim) > 0 and arr.size != int(dim):
                        continue
                    out[str(k)] = arr
        return out

    def set_many(self, items: List[tuple[str, np.ndarray]]) -> None:
        if not items:
            return
        rows: List[tuple[str, int, bytes]] = []
        for k, vec in items:
            v = np.asarray(vec, dtype=np.float32).reshape(-1)
            rows.append((k, int(v.size), v.tobytes()))
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO kv (k, dim, v) VALUES (?, ?, ?)",
                rows,
            )
            self._conn.commit()


_SQLITE_CACHE: Optional[_SQLiteKV] = None


def _get_cache_backend() -> str:
    return (os.getenv("KG_EMBEDDING_CACHE_BACKEND", "sqlite") or "sqlite").strip().lower()


def _get_sqlite_cache() -> _SQLiteKV:
    global _SQLITE_CACHE
    if _SQLITE_CACHE is not None:
        return _SQLITE_CACHE
    path = os.getenv("KG_EMBEDDING_SQLITE_PATH", "cache_embeddings.sqlite3")
    _SQLITE_CACHE = _SQLiteKV(path)
    return _SQLITE_CACHE


def _hash_embed(texts: List[str], *, dim: int = 384) -> np.ndarray:
    """Deterministic offline embedding fallback.

    Produces a fixed-size dense vector using token hashing. This is less accurate than
    a transformer but keeps the pipeline functional when HuggingFace downloads are
    unavailable (common on restricted networks).
    """
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)

    mat = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        s = (t or "").lower()
        if not s:
            continue
        # Simple whitespace tokenization is fast and stable.
        for tok in s.split():
            h = hashlib.sha256(tok.encode("utf-8", errors="ignore")).digest()
            idx = int.from_bytes(h[:4], "little", signed=False) % dim
            sign = 1.0 if (h[4] % 2 == 0) else -1.0
            mat[i, idx] += sign

    # L2 normalize to make cosine similarity meaningful.
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    mat = mat / norms
    return mat


def get_embedding_model() -> Optional["SentenceTransformer"]:  # type: ignore[name-defined]
    """
    Get or initialize the global embedding model.
    Uses sentence-transformers/all-MiniLM-L6-v2 (fast, accurate).
    """
    global _EMB_MODEL
    if _EMB_MODEL is not None:
        return _EMB_MODEL

    if SentenceTransformer is None:
        # Keep pipeline functional without the dependency.
        return None

    model_name = os.getenv("KG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    local_only = (os.getenv("KG_EMBEDDING_LOCAL_ONLY", "1") or "1").strip() != "0"
    try:
        # local_files_only prevents network calls when model is already cached.
        _EMB_MODEL = SentenceTransformer(model_name, local_files_only=local_only)  # type: ignore[misc]
        return _EMB_MODEL
    except Exception as exc:
        # Offline / restricted network fallback.
        print(f"[WARN] Embedding model unavailable ({exc}); using hashing fallback embeddings.")
        _EMB_MODEL = None
        return None


def get_embeddings_with_cache(texts: List[str]) -> np.ndarray:
    """
    Compute embeddings for a list of texts with disk caching.
    
    Parameters
    ----------
    texts : List[str]
        List of text strings to embed
    
    Returns
    -------
    np.ndarray
        Embedding matrix of shape [len(texts), embedding_dim]
        Each row is the embedding vector for the corresponding text.
    """
    if not texts:
        return np.array([]).reshape(0, 384)  # default dim

    model = get_embedding_model()

    model_id = os.getenv("KG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    dim_default = int(os.getenv("KG_EMBEDDING_DIM", "384") or 384)

    def _cache_key_for_text(t: str) -> str:
        # Namespaced key avoids mixing vectors from different models/dims.
        digest = hashlib.sha256((t or "").encode("utf-8", errors="ignore")).hexdigest()
        return DiskJSONCache.hash_key("emb_v2", model_id, str(dim_default), digest)

    # Dedupe: same text should be embedded once, then fanned out.
    # This reduces cache lookups and embedding work substantially.
    text_to_indices: dict[str, List[int]] = {}
    unique_texts: List[str] = []
    for idx, t in enumerate(texts):
        t2 = str(t or "")
        if t2 in text_to_indices:
            text_to_indices[t2].append(idx)
        else:
            text_to_indices[t2] = [idx]
            unique_texts.append(t2)

    unique_keys = [_cache_key_for_text(t) for t in unique_texts]
    backend = _get_cache_backend()

    # 1) Fetch cached vectors for unique texts
    key_to_vec: dict[str, np.ndarray] = {}
    if backend == "sqlite":
        key_to_vec = _get_sqlite_cache().get_many(unique_keys)

    # Back-compat JSON lookups (also used when backend is forced to json)
    if backend != "sqlite":
        for k, t in zip(unique_keys, unique_texts):
            cached = _JSON_CACHE.get(k)
            if cached is None:
                cached = _JSON_CACHE.get(t)
            if cached is not None:
                try:
                    key_to_vec[k] = np.asarray(cached, dtype=np.float32).reshape(-1)
                except Exception:
                    pass

    # 2) Compute missing embeddings (unique)
    missing_texts: List[str] = []
    missing_keys: List[str] = []
    for t, k in zip(unique_texts, unique_keys):
        if k not in key_to_vec:
            missing_texts.append(t)
            missing_keys.append(k)

    if missing_texts:
        t_start = time.perf_counter()
        if model is None:
            new_embs = _hash_embed(missing_texts, dim=dim_default)
        else:
            batch_size = int(os.getenv("KG_EMBEDDING_BATCH_SIZE", "64") or 64)
            show_progress = (os.getenv("KG_EMBEDDING_SHOW_PROGRESS", "0") or "0").strip() == "1"
            # These knobs do not change embedding values; they only affect throughput.
            new_embs = model.encode(
                missing_texts,
                convert_to_numpy=True,
                batch_size=max(1, batch_size),
                show_progress_bar=show_progress,
            )

        # Persist missing vectors in chosen backend
        if backend == "sqlite":
            _get_sqlite_cache().set_many([(k, np.asarray(v, dtype=np.float32)) for k, v in zip(missing_keys, new_embs)])
        else:
            for k, v in zip(missing_keys, new_embs):
                _JSON_CACHE.set(k, np.asarray(v, dtype=float).tolist())

        for k, v in zip(missing_keys, new_embs):
            key_to_vec[k] = np.asarray(v, dtype=np.float32).reshape(-1)
        
        t_end = time.perf_counter()
        logging.info(f"[Embed] Computed {len(missing_texts)} new embeddings in {t_end - t_start:.2f}s")

    # 3) Assemble into final numpy array in correct order
    # Determine dim from any available vector.
    any_vec = next(iter(key_to_vec.values()), None)
    dim = int(any_vec.size) if any_vec is not None else dim_default
    mat = np.zeros((len(texts), dim), dtype=np.float32)
    for t, k in zip(unique_texts, unique_keys):
        vec = key_to_vec.get(k)
        if vec is None:
            continue
        for idx in text_to_indices.get(t, []):
            mat[idx, :] = vec
    return mat


def compute_cosine_similarity(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query embedding and document embeddings.
    
    Parameters
    ----------
    query_emb : np.ndarray
        Query embedding vector of shape [dim] or [1, dim]
    doc_embs : np.ndarray
        Document embeddings matrix of shape [n_docs, dim]
    
    Returns
    -------
    np.ndarray
        Cosine similarity scores of shape [n_docs]
    """
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    
    # Normalize
    query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-9)
    doc_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-9)
    
    # Compute cosine similarity
    similarities = (doc_norms @ query_norm.T).flatten()
    return similarities
