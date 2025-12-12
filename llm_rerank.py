# llm_rerank.py
"""
LLM-based reranking of retrieval candidates.
Uses an LLM to rank candidates by relevance to the query with caching.
"""
from __future__ import annotations
from typing import List, Dict, Any
import json
import google.generativeai as genai
from hybrid_search import RetrievalCandidate
from cache_utils import DiskJSONCache


# Global cache for reranking results
_RERANK_CACHE = DiskJSONCache("cache_reranking.json")


RERANK_PROMPT = """
You are an expert at evaluating the relevance of evidence to answer questions.

Task:
- You will receive a question and a list of evidence items (sentences, entities, community summaries).
- Rank these evidence items by their usefulness in answering the question.
- Only include items that are factually relevant and useful.
- Return the top items (up to the specified limit) ranked from most to least useful.

Output format (JSON only, no extra text):
{
  "ranked_evidence_ids": ["id1", "id2", "id3", ...],
  "rationale": "Brief explanation of ranking criteria and top selections"
}

Guidelines:
- Prioritize items that directly answer the question
- Prefer specific facts over general summaries
- Consider semantic relevance and factual content
- Include diverse evidence types when possible
"""


def llm_rerank_candidates(
    query: str,
    candidates: List[RetrievalCandidate],
    top_k: int = 12,
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.1,
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
    
    # Build cache key
    candidate_ids = [c.id for c in candidates]
    cache_key = DiskJSONCache.hash_key(query, json.dumps(candidate_ids, sort_keys=True))
    
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
    
    # Prepare evidence items for LLM
    evidence_items = []
    for i, cand in enumerate(candidates):
        evidence_items.append({
            "id": cand.id,
            "type": cand.item_type,
            "text": cand.text[:500],  # Limit text length to avoid token overflow
            "semantic_score": round(cand.semantic_score, 3),
            "graph_score": round(cand.graph_score, 3)
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
    
    # Call LLM
    model = genai.GenerativeModel(model_name)
    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": temperature}
        )
        
        raw = response.text.strip()
        
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
        
        # Cache result
        cache_result = {
            "ranked_evidence_ids": ranked_ids,
            "rationale": rationale
        }
        if use_cache:
            _RERANK_CACHE.set(cache_key, cache_result)
        
        # Build ranked candidates list
        id_to_cand = {c.id: c for c in candidates}
        ranked_candidates = [id_to_cand[cid] for cid in ranked_ids if cid in id_to_cand]
        
        if verbose:
            print(f"[Rerank] Reranked to top {len(ranked_candidates)} items")
            print(f"[Rerank] Rationale: {rationale[:200]}...")
        
        return {
            "ranked_candidates": ranked_candidates,
            "ranked_evidence_ids": ranked_ids,
            "rationale": rationale
        }
    
    except Exception as e:
        if verbose:
            print(f"[Rerank] LLM reranking failed: {e}")
        
        # Fallback: return original order (by hybrid score)
        fallback_ids = [c.id for c in candidates[:top_k]]
        return {
            "ranked_candidates": candidates[:top_k],
            "ranked_evidence_ids": fallback_ids,
            "rationale": f"Reranking failed ({str(e)}), using hybrid scores"
        }
