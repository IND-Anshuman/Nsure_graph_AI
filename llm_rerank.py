# llm_rerank.py
"""
LLM-based reranking of retrieval candidates.
Uses an LLM to rank candidates by relevance to the query with caching.
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import json
import os
import google.generativeai as genai
from openai import OpenAI
from hybrid_search import RetrievalCandidate
from cache_utils import DiskJSONCache
from sentence_transformers import CrossEncoder


# Global cache for reranking results
_RERANK_CACHE = DiskJSONCache("cache_reranking.json")
_CROSS_ENCODER: CrossEncoder | None = None
_CROSS_ENCODER_NAME: str | None = None


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


def _get_cross_encoder(model_name: str) -> CrossEncoder:
    global _CROSS_ENCODER, _CROSS_ENCODER_NAME
    if _CROSS_ENCODER is None or _CROSS_ENCODER_NAME != model_name:
        _CROSS_ENCODER = CrossEncoder(model_name, device="cpu")
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

    if os.getenv("GOOGLE_API_KEY"):
        try:
            model = genai.GenerativeModel(prefer_model)
            resp = model.generate_content(prompt, generation_config={"temperature": temperature})
            return resp.text.strip(), "google", prefer_model
        except Exception as e:
            last_error = e
            if verbose:
                print(f"[Rerank] Gemini failed, will try OpenAI fallback: {e}")

    if os.getenv("OPENAI_API_KEY"):
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
            return resp.choices[0].message.content.strip(), "openai", fallback_model
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
        ranked = sorted(candidates, key=lambda c: float(getattr(c, "hybrid_score", 0.0) or 0.0), reverse=True)
        ranked = ranked[:top_k]
        return {
            "ranked_candidates": ranked,
            "ranked_evidence_ids": [c.id for c in ranked],
            "rationale": "No rerank (hybrid_score order)"
        }

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
            ranked = sorted(candidates, key=lambda c: float(getattr(c, "hybrid_score", 0.0) or 0.0), reverse=True)
            ranked = ranked[:top_k]
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
                ranked = sorted(candidates, key=lambda c: float(getattr(c, "hybrid_score", 0.0) or 0.0), reverse=True)[:top_k]
                return {
                    "ranked_candidates": ranked,
                    "ranked_evidence_ids": [c.id for c in ranked],
                    "rationale": "Cross-only failed; used hybrid_score order"
                }
                "rationale": "Cross-encoder rerank (no LLM)"
            }
        candidate_ids = [c.id for c in candidates]
        # Make cache key order-insensitive: candidate ordering can vary slightly due to dedup/expansion.
        candidate_ids_sorted = sorted(candidate_ids)
        cache_key = DiskJSONCache.hash_key(
            query,
            json.dumps(candidate_ids_sorted, ensure_ascii=False),
            model_name,
            fallback_openai_model,
            cross_encoder_model if use_cross_encoder else "no_cross",
        )
                print(f"[Rerank] Cross-only mode failed, falling back to no rerank: {e}")
            ranked = sorted(candidates, key=lambda c: float(getattr(c, "hybrid_score", 0.0) or 0.0), reverse=True)[:top_k]
            return {
                "ranked_candidates": ranked,
                "ranked_evidence_ids": [c.id for c in ranked],
                "rationale": "Cross-only failed; used hybrid_score order"
            }
    
    # Build cache key
    candidate_ids = [c.id for c in candidates]
    cache_key = DiskJSONCache.hash_key(
        query,
        json.dumps(candidate_ids, sort_keys=True),
        "mode=llm",
        model_name,
        fallback_openai_model,
        cross_encoder_model if use_cross_encoder else "no_cross",
    )
    
    # Check cache
    if use_cache:
        cached = _RERANK_CACHE.get(cache_key)
        if cached:
            if verbose:
                print("[Rerank] Using cached reranking result")
            # Reconstruct ranked candidates from cached ids
            id_to_cand = {c.id: c for c in candidates}
            ranked_cands = [id_to_cand[cid] for cid in cached["ranked_evidence_ids"] if cid in id_to_cand]
            return {
                "ranked_candidates": ranked_cands,
                "ranked_evidence_ids": cached["ranked_evidence_ids"],
                "rationale": cached["rationale"]
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
            "type": cand.item_type,
            "text": cand.text[:500],  # Limit text length to avoid token overflow
            "semantic_score": round(cand.semantic_score, 3),
            "graph_score": round(cand.graph_score, 3),
            "cross_score": round(getattr(cand, "cross_score", 0.0), 3),
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
        
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        
        result = json.loads(raw)
        
        # Validate result
        ranked_ids = result.get("ranked_evidence_ids", [])
        rationale = result.get("rationale", "")

        # Limit to top_k
        ranked_ids = ranked_ids[:top_k]

        # Fallback if model returned nothing
        if not ranked_ids:
            fallback_ids = [c.id for c in candidates[:top_k]]
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
        fallback_ids = [c.id for c in base_list[:top_k]]
        return {
            "ranked_candidates": base_list[:top_k],
            "ranked_evidence_ids": fallback_ids,
            "rationale": f"Reranking failed ({str(e)}), using hybrid scores"
        }
