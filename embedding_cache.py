# embedding_cache.py
"""
Centralized embedding cache module.
Reuses the existing DiskJSONCache infrastructure for storing and retrieving embeddings.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
from cache_utils import DiskJSONCache
import hashlib
import os

# Global instances
_EMB_MODEL: Optional["SentenceTransformer"] = None  # type: ignore[name-defined]
_EMB_CACHE = DiskJSONCache("cache_embeddings.json")


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
    cached_vectors: List[tuple[int, List[float]]] = []
    missing_indices: List[int] = []
    missing_texts: List[str] = []

    model_id = os.getenv("KG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    dim_default = int(os.getenv("KG_EMBEDDING_DIM", "384") or 384)

    def _cache_key_for_text(t: str) -> str:
        # Namespaced key avoids mixing vectors from different models/dims.
        digest = hashlib.sha256((t or "").encode("utf-8", errors="ignore")).hexdigest()
        return DiskJSONCache.hash_key("emb_v2", model_id, str(dim_default), digest)

    # 1) Check cache for each text
    for idx, text in enumerate(texts):
        key = _cache_key_for_text(text)
        cached = _EMB_CACHE.get(key)
        if cached is None:
            # Back-compat: old cache used raw text as key.
            cached = _EMB_CACHE.get(text)
        if cached is not None:
            cached_vectors.append((idx, cached))
        else:
            missing_indices.append(idx)
            missing_texts.append(text)

    # 2) Compute embeddings for missing texts
    if missing_texts:
        if model is None:
            new_embs = _hash_embed(missing_texts, dim=dim_default)
        else:
            new_embs = model.encode(missing_texts, convert_to_numpy=True)
        for i, vec in enumerate(new_embs):
            idx = missing_indices[i]
            vec_list = np.asarray(vec, dtype=float).tolist()
            key = _cache_key_for_text(texts[idx])
            _EMB_CACHE.set(key, vec_list)
            cached_vectors.append((idx, vec_list))

    # 3) Assemble into final numpy array in correct order
    dim = len(cached_vectors[0][1]) if cached_vectors else dim_default
    mat = np.zeros((len(texts), dim), dtype=float)

    for idx, vec_list in cached_vectors:
        mat[idx, :] = np.array(vec_list, dtype=float)

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
