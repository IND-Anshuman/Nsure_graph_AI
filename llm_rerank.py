# llm_rerank.py
"""
LLM-based reranking of retrieval candidates.
Uses an LLM to rank candidates by relevance to the query with caching.
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
import json
import os

from genai_compat import generate_text as genai_generate_text, is_available as genai_is_available

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

from hybrid_search import RetrievalCandidate
from cache_utils import DiskJSONCache

try:
    from sentence_transformers import CrossEncoder as _CrossEncoderRuntime  # type: ignore
except Exception:
    _CrossEncoderRuntime = None  # type: ignore

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder as CrossEncoder  # type: ignore
else:
    CrossEncoder = Any  # type: ignore


# Global cache for reranking results
_RERANK_CACHE = DiskJSONCache("cache_reranking_v3.json")
_CROSS_ENCODER: Optional[CrossEncoder] = None
_CROSS_ENCODER_NAME: Optional[str] = None


RERANK_PROMPT = """
You are an expert at ranking evidence for question answering.

Evidence item fields:
- id
- type: "SENTENCE", "ENTITY_CONTEXT", "ENTITY", "COMMUNITY", "CHUNK", etc.
- text
- semantic_score, graph_score, cross_score (higher is better)

Instructions:
- Read the question and evidence list.
- Select and order only the evidence that is clearly useful to answer the question.
- Strongly prefer items whose type is "SENTENCE" or "ENTITY_CONTEXT".
- Use "COMMUNITY" or "CHUNK" items only when they add essential context not already present in sentence-level evidence.
- When two items express similar content, keep the more specific / shorter sentence-level one.
- If an item is redundant or off-topic, exclude it.
- Keep the list concise (no more than the requested top_k).

Scoring rubric (highest first):
- relevance to question
- degree of direct factual overlap with the query intent
- specificity and concreteness
- novelty (penalize near-duplicates)

Output JSON only (no prose):
{
    "ranked_evidence_ids": ["id1", "id2", ...],
    "rationale": "1-2 sentences on why the top items were chosen"
}
"""


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _extract_json_text(raw: str) -> str:
    raw = (raw or "").strip()

    # Handle fenced blocks like ```json ... ```
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2 and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    # If there's extra text, try to extract the first JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1].strip()

    return raw


def _hybrid_rank(candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
    return sorted(
        candidates,
        key=lambda c: _safe_float(getattr(c, "hybrid_score", 0.0), 0.0),
        reverse=True,
    )


def _get_cross_encoder(model_name: str) -> CrossEncoder:
    global _CROSS_ENCODER, _CROSS_ENCODER_NAME
    if _CrossEncoderRuntime is None:
        raise RuntimeError("sentence-transformers is not installed (CrossEncoder unavailable)")
    if _CROSS_ENCODER is None or _CROSS_ENCODER_NAME != model_name:
        _CROSS_ENCODER = _CrossEncoderRuntime(model_name, device="cpu")
        _CROSS_ENCODER_NAME = model_name
    return _CROSS_ENCODER


def _score_with_cross_encoder(query: str, candidates: List[RetrievalCandidate], model_name: str, top_k: int) -> List[RetrievalCandidate]:
    if not candidates:
        return []
    model = _get_cross_encoder(model_name)
    pairs = [(query, cand.text) for cand in candidates]
    scores = model.predict(pairs)
    for cand, score in zip(candidates, scores):
        cand.cross_score = float(score)
    ranked = sorted(candidates, key=lambda c: c.cross_score, reverse=True)
    return ranked[:top_k]


def _call_llm_with_fallback(prompt: str,
                            temperature: float,
                            prefer_model: str,
                            fallback_model: str,
                            verbose: bool) -> Tuple[str, str, str]:
    """Call Gemini first; if unavailable/fails, fall back to OpenAI chat completion.
    Returns (text, provider, model).
    """
    last_error = None

    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key and genai_is_available():
        try:
            text = genai_generate_text(prefer_model, prompt, temperature=temperature)
            return (text or "").strip(), "google", prefer_model
        except Exception as e:
            last_error = e
            if verbose:
                print(f"[Rerank] Gemini failed, will try OpenAI fallback: {e}")

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and OpenAI is not None:
        try:
            client = OpenAI()
            resp = client.chat.completions.create(
                model=fallback_model,
                messages=[
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip(), "openai", fallback_model
        except Exception as e:
            last_error = e
            if verbose:
                print(f"[Rerank] OpenAI fallback failed: {e}")

    raise last_error or RuntimeError("No LLM provider configured (set GOOGLE_API_KEY or OPENAI_API_KEY)")


def llm_rerank_candidates(
    query: str,
    candidates: List[RetrievalCandidate],
    top_k: int = 12,
    rerank_mode: str = "llm",
    model_name: str = "gemini-2.0-flash",
    fallback_openai_model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    use_cross_encoder: bool = True,
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    cross_top_k: int = 60,
    use_cache: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Rerank candidates using an LLM.
    
    Parameters
    ----------
    query : str
        User query
    candidates : List[RetrievalCandidate]
        Candidates to rerank
    top_k : int
        Number of top items to return (default 12)
    model_name : str
        Gemini model to use
    temperature : float
        Temperature for LLM (lower = more deterministic)
    use_cache : bool
        Whether to use cached results
    verbose : bool
        Whether to print debug info
    
    Returns
    -------
    Dict with keys:
        - ranked_candidates: List[RetrievalCandidate] (top-k reranked)
        - ranked_evidence_ids: List[str] (ordered evidence ids)
        - rationale: str (LLM's ranking explanation)
    """
    if not candidates:
        return {
            "ranked_candidates": [],
            "ranked_evidence_ids": [],
            "rationale": "No candidates provided"
        }

    mode = (rerank_mode or "llm").strip().lower()
    if mode not in {"llm", "cross", "none"}:
        mode = "llm"

    # Fast path: no LLM calls.
    if mode == "none":
        ranked = _hybrid_rank(candidates)[:top_k]
        return {
            "ranked_candidates": ranked,
            "ranked_evidence_ids": [c.id for c in ranked],
            "rationale": "No rerank (hybrid_score order)"
        }

    if mode == "cross":
        try:
            ranked = _score_with_cross_encoder(query, candidates, cross_encoder_model, top_k)
            return {
                "ranked_candidates": ranked,
                "ranked_evidence_ids": [c.id for c in ranked],
                "rationale": "Cross-encoder rerank (no LLM)"
            }
        except Exception as e:
            if verbose:
                print(f"[Rerank] Cross-only mode failed, falling back to no rerank: {e}")
            ranked = _hybrid_rank(candidates)[:top_k]
            return {
                "ranked_candidates": ranked,
                "ranked_evidence_ids": [c.id for c in ranked],
                "rationale": "Cross-only failed; used hybrid_score order"
            }

    # Build cache key (LLM mode)
    candidate_ids = sorted([c.id for c in candidates])
    cache_key = DiskJSONCache.hash_key(
        "rerank_v4",
        query,
        json.dumps(candidate_ids, ensure_ascii=False),
        "mode=llm",
        model_name,
        fallback_openai_model,
        cross_encoder_model if use_cross_encoder else "no_cross",
    )
    
    # Check cache
    if use_cache:
        cached = _RERANK_CACHE.get(cache_key)
        if isinstance(cached, dict) and cached.get("ranked_evidence_ids"):
            if verbose:
                print("[Rerank] Using cached reranking result")
            # Reconstruct ranked candidates from cached ids
            cached_ids = cached.get("ranked_evidence_ids", [])
            # Ensure we still return a full top_k (cached results may be from older runs).
            seen = set()
            cached_ids = [cid for cid in cached_ids if isinstance(cid, str) and not (cid in seen or seen.add(cid))]
            if len(cached_ids) < top_k:
                filler = [c.id for c in _hybrid_rank(candidates) if c.id not in set(cached_ids)]
                cached_ids.extend(filler[: max(0, top_k - len(cached_ids))])
            cached_ids = cached_ids[:top_k]
            id_to_cand = {c.id: c for c in candidates}
            ranked_cands = [id_to_cand[cid] for cid in cached_ids if cid in id_to_cand]
            return {
                "ranked_candidates": ranked_cands,
                "ranked_evidence_ids": cached_ids,
                "rationale": cached.get("rationale", "")
            }
    
    # Stage 1: cross-encoder rerank to trim candidate list
    stage1_list = candidates
    if use_cross_encoder:
        try:
            top_limit = max(top_k * 3, cross_top_k)
            stage1_list = _score_with_cross_encoder(query, candidates, cross_encoder_model, top_limit)
            if verbose:
                print(f"[Rerank] Cross-encoder trimmed {len(candidates)} -> {len(stage1_list)}")
        except Exception as e:
            if verbose:
                print(f"[Rerank] Cross-encoder unavailable, skipping: {e}")
            stage1_list = candidates

    # Prepare evidence items for LLM
    evidence_items = []
    for i, cand in enumerate(stage1_list):
        evidence_items.append({
            "id": cand.id,
            "type": getattr(cand, "item_type", None),
            "text": (getattr(cand, "text", "") or "")[:500],  # Limit text length to avoid token overflow
            "semantic_score": round(_safe_float(getattr(cand, "semantic_score", 0.0), 0.0), 3),
            "graph_score": round(_safe_float(getattr(cand, "graph_score", 0.0), 0.0), 3),
            "cross_score": round(_safe_float(getattr(cand, "cross_score", 0.0), 0.0), 3),
        })
    
    # Build prompt
    user_content = {
        "question": query,
        "evidence_items": evidence_items,
        "top_k": top_k
    }
    
    prompt = f"{RERANK_PROMPT}\n\nInput:\n{json.dumps(user_content, ensure_ascii=False)}"
    
    if verbose:
        print(f"[Rerank] Calling LLM to rerank {len(candidates)} candidates...")
    
    try:
        raw, provider_used, model_used = _call_llm_with_fallback(
            prompt=prompt,
            temperature=temperature,
            prefer_model=model_name,
            fallback_model=fallback_openai_model,
            verbose=verbose,
        )
        
        raw = _extract_json_text(raw)
        result = json.loads(raw)
        
        # Validate result
        ranked_ids = result.get("ranked_evidence_ids", [])
        rationale = result.get("rationale", "")

        # Normalize: unique, preserve order, keep only strings
        seen = set()
        ranked_ids = [rid for rid in ranked_ids if isinstance(rid, str) and not (rid in seen or seen.add(rid))]

        # If model returns too few ids, fill deterministically from hybrid order
        if len(ranked_ids) < top_k:
            filler = [c.id for c in _hybrid_rank(candidates) if c.id not in set(ranked_ids)]
            ranked_ids.extend(filler[: max(0, top_k - len(ranked_ids))])

        # Limit to top_k
        ranked_ids = ranked_ids[:top_k]

        # Fallback if model returned nothing
        if not ranked_ids:
            fallback_ids = [c.id for c in _hybrid_rank(candidates)[:top_k]]
            if verbose:
                print("[Rerank] No ranked ids returned by LLM; falling back to hybrid order")
            ranked_ids = fallback_ids
            rationale = rationale or "Fallback to hybrid scores due to empty rerank result"

        # Cache result
        cache_result = {
            "ranked_evidence_ids": ranked_ids,
            "rationale": rationale,
            "provider": provider_used,
            "model": model_used,
        }
        if use_cache:
            _RERANK_CACHE.set(cache_key, cache_result)

        # Build ranked candidates list
        id_to_cand = {c.id: c for c in candidates}
        id_to_cand.update({c.id: c for c in stage1_list})
        ranked_candidates = [id_to_cand[cid] for cid in ranked_ids if cid in id_to_cand]

        if verbose:
            print(f"[Rerank] Reranked to top {len(ranked_candidates)} items")
            print(f"[Rerank] Rationale: {rationale[:200]}...")

        return {
            "ranked_candidates": ranked_candidates,
            "ranked_evidence_ids": ranked_ids,
            "rationale": rationale,
            "provider": provider_used,
            "model": model_used,
        }
    
    except Exception as e:
        if verbose:
            print(f"[Rerank] LLM reranking failed: {e}")
        
        # Fallback: return original order (by hybrid score)
        base_list = stage1_list if stage1_list else candidates
        base_list = _hybrid_rank(base_list)
        fallback_ids = [c.id for c in base_list[:top_k]]
        return {
            "ranked_candidates": base_list[:top_k],
            "ranked_evidence_ids": fallback_ids,
            "rationale": f"Reranking failed ({str(e)}), using hybrid scores"
        }
