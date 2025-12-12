# llm_synthesis.py
"""
LLM-grounded synthesis with evidence citation.
Generates answers using only provided evidence and cites sources.
"""
from __future__ import annotations
from typing import List, Dict, Any
import json
import google.generativeai as genai
from hybrid_search import RetrievalCandidate
from cache_utils import DiskJSONCache


# Global cache for synthesis results
_SYNTHESIS_CACHE = DiskJSONCache("cache_synthesis.json")


SYNTHESIS_PROMPT = """
You are an expert at answering questions using only provided evidence.

CRITICAL RULES:
1. Use ONLY the evidence provided below. Do NOT use external knowledge.
2. If the evidence is insufficient to answer the question, explicitly state this.
3. For EVERY factual claim in your answer, cite the evidence ID(s) that support it.
4. Output ONLY valid JSON in the exact format specified below.

Evidence Format:
Each evidence item has:
- id: unique identifier
- type: SENTENCE, ENTITY, COMMUNITY, etc.
- text: the content
- metadata: additional context

Output Format (JSON only, no extra text):
{
  "answer": "Your comprehensive answer here. Cite evidence like [evidence_id] after each claim.",
  "used_evidence": ["evidence_id1", "evidence_id2", ...],
  "extracted_facts": [
    {
      "fact": "A specific factual statement",
      "evidence_ids": ["id1", "id2"]
    },
    ...
  ],
  "confidence": "high|medium|low",
  "insufficiency_note": "Optional: if evidence is insufficient, explain what's missing"
}

Guidelines:
- Be precise and factual
- Cite evidence IDs in square brackets [id] immediately after claims
- Only list evidence IDs you actually used in "used_evidence"
- Extract 3-7 key facts from your answer for "extracted_facts"
- Set confidence based on evidence quality and coverage
"""


def llm_synthesize_answer(
    query: str,
    evidence_candidates: List[RetrievalCandidate],
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.2,
    use_cache: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Generate a grounded answer from evidence using an LLM.
    
    Parameters
    ----------
    query : str
        User query
    evidence_candidates : List[RetrievalCandidate]
        Evidence items to use for answering
    model_name : str
        Gemini model to use
    temperature : float
        Temperature for LLM (0.2 for factual consistency)
    use_cache : bool
        Whether to use cached results
    verbose : bool
        Whether to print debug info
    
    Returns
    -------
    Dict with keys:
        - answer: str (the generated answer with citations)
        - used_evidence: List[str] (evidence IDs actually used)
        - extracted_facts: List[Dict] (key facts with evidence citations)
        - confidence: str (high/medium/low)
        - insufficiency_note: str (optional, if evidence insufficient)
    """
    if not evidence_candidates:
        return {
            "answer": "No evidence available to answer this question.",
            "used_evidence": [],
            "extracted_facts": [],
            "confidence": "low",
            "insufficiency_note": "No evidence provided"
        }
    
    # Build cache key
    evidence_ids = [e.id for e in evidence_candidates]
    cache_key = DiskJSONCache.hash_key(query, json.dumps(evidence_ids, sort_keys=True))
    
    # Check cache
    if use_cache:
        cached = _SYNTHESIS_CACHE.get(cache_key)
        if cached:
            if verbose:
                print("[Synthesis] Using cached synthesis result")
            return cached
    
    # Prepare evidence for LLM
    evidence_items = []
    for i, cand in enumerate(evidence_candidates):
        evidence_items.append({
            "id": cand.id,
            "type": cand.item_type,
            "text": cand.text,
            "metadata": cand.metadata
        })
    
    # Build prompt
    user_content = {
        "question": query,
        "evidence": evidence_items
    }
    
    prompt = f"{SYNTHESIS_PROMPT}\n\nInput:\n{json.dumps(user_content, ensure_ascii=False)}"
    
    if verbose:
        print(f"[Synthesis] Calling LLM to synthesize answer from {len(evidence_candidates)} evidence items...")
    
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
        
        # Validate and extract fields
        answer = result.get("answer", "Unable to generate answer")
        used_evidence = result.get("used_evidence", [])
        extracted_facts = result.get("extracted_facts", [])
        confidence = result.get("confidence", "medium")
        insufficiency_note = result.get("insufficiency_note")
        
        synthesis_result = {
            "answer": answer,
            "used_evidence": used_evidence,
            "extracted_facts": extracted_facts,
            "confidence": confidence,
            "insufficiency_note": insufficiency_note
        }
        
        # Cache result
        if use_cache:
            _SYNTHESIS_CACHE.set(cache_key, synthesis_result)
        
        if verbose:
            print(f"[Synthesis] Generated answer with {len(used_evidence)} evidence citations")
            print(f"[Synthesis] Confidence: {confidence}")
            if insufficiency_note:
                print(f"[Synthesis] Note: {insufficiency_note}")
        
        return synthesis_result
    
    except Exception as e:
        if verbose:
            print(f"[Synthesis] LLM synthesis failed: {e}")
        
        # Fallback: simple concatenation
        fallback_answer = f"Error generating answer: {str(e)}\n\nAvailable evidence:\n"
        for cand in evidence_candidates[:3]:
            fallback_answer += f"- [{cand.id}] {cand.text[:200]}...\n"
        
        return {
            "answer": fallback_answer,
            "used_evidence": [],
            "extracted_facts": [],
            "confidence": "low",
            "insufficiency_note": f"Synthesis failed: {str(e)}"
        }


def format_answer_output(synthesis_result: Dict[str, Any], 
                        include_facts: bool = True,
                        include_evidence_list: bool = True) -> str:
    """
    Format the synthesis result for display.
    
    Parameters
    ----------
    synthesis_result : Dict
        Result from llm_synthesize_answer
    include_facts : bool
        Whether to include extracted facts section
    include_evidence_list : bool
        Whether to include list of used evidence IDs
    
    Returns
    -------
    str
        Formatted answer string
    """
    output_parts = []
    
    # Main answer
    output_parts.append("=" * 80)
    output_parts.append("ANSWER")
    output_parts.append("=" * 80)
    output_parts.append(synthesis_result["answer"])
    output_parts.append("")
    
    # Confidence
    confidence = synthesis_result.get("confidence", "medium")
    output_parts.append(f"Confidence: {confidence.upper()}")
    
    # Insufficiency note if present
    if synthesis_result.get("insufficiency_note"):
        output_parts.append(f"Note: {synthesis_result['insufficiency_note']}")
    
    output_parts.append("")
    
    # Evidence used
    if include_evidence_list:
        used = synthesis_result.get("used_evidence", [])
        output_parts.append(f"Evidence Used ({len(used)} items):")
        for eid in used:
            output_parts.append(f"  - {eid}")
        output_parts.append("")
    
    # Extracted facts
    if include_facts:
        facts = synthesis_result.get("extracted_facts", [])
        if facts:
            output_parts.append("Key Facts Extracted:")
            for i, fact_obj in enumerate(facts, 1):
                fact_text = fact_obj.get("fact", "")
                evidence_ids = fact_obj.get("evidence_ids", [])
                output_parts.append(f"  {i}. {fact_text}")
                output_parts.append(f"     Sources: {', '.join(evidence_ids)}")
            output_parts.append("")
    
    output_parts.append("=" * 80)
    
    return "\n".join(output_parts)
