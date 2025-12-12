# embedding_cache.py
"""
Centralized embedding cache module.
Reuses the existing DiskJSONCache infrastructure for storing and retrieving embeddings.
"""
from __future__ import annotations
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from cache_utils import DiskJSONCache

# Global instances
_EMB_MODEL: SentenceTransformer | None = None
_EMB_CACHE = DiskJSONCache("cache_embeddings.json")


def get_embedding_model() -> SentenceTransformer:
    """
    Get or initialize the global embedding model.
    Uses sentence-transformers/all-MiniLM-L6-v2 (fast, accurate).
    """
    global _EMB_MODEL
    if _EMB_MODEL is None:
        _EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMB_MODEL


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
        return np.array([]).reshape(0, 384)  # MiniLM-L6-v2 has 384 dimensions
    
    model = get_embedding_model()
    cached_vectors: List[tuple[int, List[float]]] = []
    missing_indices: List[int] = []
    missing_texts: List[str] = []

    # 1) Check cache for each text
    for idx, text in enumerate(texts):
        cached = _EMB_CACHE.get(text)
        if cached is not None:
            cached_vectors.append((idx, cached))
        else:
            missing_indices.append(idx)
            missing_texts.append(text)

    # 2) Compute embeddings for missing texts
    if missing_texts:
        new_embs = model.encode(missing_texts, convert_to_numpy=True)
        for i, vec in enumerate(new_embs):
            idx = missing_indices[i]
            vec_list = vec.astype(float).tolist()
            _EMB_CACHE.set(texts[idx], vec_list)
            cached_vectors.append((idx, vec_list))

    # 3) Assemble into final numpy array in correct order
    dim = len(cached_vectors[0][1]) if cached_vectors else 384
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
