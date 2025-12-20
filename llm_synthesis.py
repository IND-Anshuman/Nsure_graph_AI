# llm_synthesis.py
"""
LLM-grounded synthesis with evidence citation.
Generates answers using only provided evidence and cites sources.
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import json
import os
import google.generativeai as genai
from openai import OpenAI
from hybrid_search import RetrievalCandidate
from data_corpus import KnowledgeGraph
from cache_utils import DiskJSONCache


# Global cache for synthesis results
_SYNTHESIS_CACHE = DiskJSONCache("cache_synthesis.json")


SYNTHESIS_PROMPT = """
You are an expert at answering questions using only provided evidence.

Evidence item fields:
- id
- type: "SENTENCE", "ENTITY_CONTEXT", "ENTITY", "COMMUNITY", "CHUNK", etc.
- text
- metadata

CRITICAL RULES:
1) Use ONLY the evidence below. No external knowledge.
2) Answer concisely in <=120 words.
3) Every claim must cite evidence IDs in square brackets immediately after the claim.
4) Favor claims supported by >=2 evidence IDs when possible; otherwise cite the single strongest.
5) If evidence is insufficient, say so explicitly and return a cautious answer.
6) Prefer sentence-level evidence (type="SENTENCE" or "ENTITY_CONTEXT") as the primary basis for claims.
7) Treat "COMMUNITY" or "CHUNK" items as secondary background; do not make claims that contradict the specific sentences.
8) Output valid JSON only in the format below. No prose outside JSON.

Output JSON (no extra text):
{
    "answer": "Concise answer with [evidence_id] citations",
    "used_evidence": ["id1", "id2", ...],
    "extracted_facts": [
        {"fact": "Factual statement", "evidence_ids": ["id1", "id2"]},
        ...
    ],
    "confidence": "high|medium|low",
    "insufficiency_note": "Optional explanation when evidence is insufficient"
}

Guidance:
- Prefer specific, high-signal sentence or entity-context evidence; drop redundant items.
- Avoid generic, high-level paraphrases that are not clearly anchored in the wording of the evidence.
- If the evidence is mostly high-level community summaries, make this clear and lower confidence.
- Keep style declarative and grounded; no hedging fillers.
- 3–7 extracted facts are expected.
"""


def _sort_evidence_by_type_and_score(evidence_candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
    """Sort evidence so that sentence-level items come first.

    Priority order (best to worst):
    SENTENCE / ENTITY_CONTEXT -> ENTITY -> CHUNK -> COMMUNITY / other.
    Within the same type, fall back to semantic_score when available.
    """

    def _type_priority(t: str) -> int:
        t = (t or "").upper()
        if t in {"SENTENCE", "ENTITY_CONTEXT"}:
            return 0
        if t == "ENTITY":
            return 1
        if t == "CHUNK":
            return 2
        if t.startswith("COMMUNITY"):
            return 3
        return 4

    def _key(cand: RetrievalCandidate) -> Any:
        item_type = getattr(cand, "item_type", None)
        priority = _type_priority(item_type or "")
        sem_score = getattr(cand, "semantic_score", 0.0) or 0.0
        # negative semantic score so higher scores come first within same priority
        return (priority, -float(sem_score))

    try:
        return sorted(evidence_candidates, key=_key)
    except Exception:
        # In case of mixed types or dicts, fall back to original order
        return evidence_candidates


def _call_llm_with_fallback(prompt: str,
                            temperature: float,
                            prefer_model: str,
                            fallback_model: str,
                            verbose: bool) -> Tuple[str, str, str]:
    """Call Gemini first; fall back to OpenAI chat completion. Returns (text, provider, model)."""
    last_error = None

    if os.getenv("GOOGLE_API_KEY"):
        try:
            model = genai.GenerativeModel(prefer_model)
            resp = model.generate_content(prompt, generation_config={"temperature": temperature})
            return resp.text.strip(), "google", prefer_model
        except Exception as e:
            last_error = e
            if verbose:
                print(f"[Synthesis] Gemini failed, will try OpenAI fallback: {e}")

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
                print(f"[Synthesis] OpenAI fallback failed: {e}")

    raise last_error or RuntimeError("No LLM provider configured (set GOOGLE_API_KEY or OPENAI_API_KEY)")


def _is_structural_query(query: str) -> bool:
    """Deterministic rule-based structural query detector.

    Structural queries ask about graph organization: communities/clusters/grouping/hierarchy.
    """
    q = (query or "").lower()
    # Keep this conservative: only trigger when the user is explicitly asking about structure.
    keywords = [
        "community",
        "communities",
        "cluster",
        "clusters",
        "grouped",
        "grouping",
        "groupings",
        "organized into",
        "organisation",
        "organization",
        "structure",
        "hierarchy",
        "hierarchical",
        "related domains",
        "related topics",
        "related concepts",
        "how are",
        "how is",
        "partition",
    ]
    return any(k in q for k in keywords)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _community_ids_from_evidence(evidence_candidates: List[RetrievalCandidate], top_k: int) -> List[str]:
    comm_ids: List[str] = []
    for cand in (evidence_candidates or [])[: max(1, top_k)]:
        t = (getattr(cand, "item_type", "") or "").upper()
        if t.startswith("COMMUNITY"):
            cid = getattr(cand, "id", None)
            if cid and cid not in comm_ids:
                comm_ids.append(cid)
    return comm_ids


def _community_members_from_graph(graph: KnowledgeGraph, community_id: str) -> List[str]:
    """Return member node ids for a community using MEMBER_OF edges (entity/domain -> community)."""
    members: List[str] = []
    for e in graph.edges:
        if e.type == "MEMBER_OF" and e.target == community_id:
            if e.source in graph.nodes:
                members.append(e.source)
    # stable ordering: prefer higher mention_count if present, else lexical
    def _member_key(node_id: str) -> Any:
        node = graph.nodes.get(node_id)
        props = node.properties if node else {}
        mention_count = _safe_int(props.get("mention_count"), 0)
        canonical = str(props.get("canonical") or props.get("text") or node_id)
        return (-mention_count, canonical.lower())

    return sorted(set(members), key=_member_key)


def _parent_communities_from_graph(graph: KnowledgeGraph, community_id: str) -> List[str]:
    parents: List[str] = []
    for e in graph.edges:
        if e.type == "PART_OF" and e.source == community_id and e.target in graph.nodes:
            parents.append(e.target)
    return parents


def _graph_derived_structural_answer(
    query: str,
    graph: KnowledgeGraph,
    evidence_candidates: List[RetrievalCandidate],
    *,
    top_k_for_structure: int = 12,
    community_members_min: int = 5,
    coherence_min: float = 0.2,
    max_communities: int = 6,
    max_members_per_community: int = 10,
) -> Optional[Dict[str, Any]]:
    """Build a graph-derived grouping answer for structural queries.

    Returns None if conditions for structural override are not met.
    """
    if not graph or not _is_structural_query(query):
        return None

    community_ids = _community_ids_from_evidence(evidence_candidates, top_k=top_k_for_structure)
    if not community_ids:
        return None

    eligible: List[Tuple[str, int, float]] = []
    for cid in community_ids:
        node = graph.nodes.get(cid)
        if not node or node.label != "COMMUNITY":
            continue
        members_count = _safe_int(node.properties.get("members_count"), 0)
        coherence = _safe_float(node.properties.get("coherence"), 0.0)
        if members_count > community_members_min and coherence >= coherence_min:
            eligible.append((cid, members_count, coherence))

    if not eligible:
        return None

    # Prefer higher coherence, then larger communities.
    eligible.sort(key=lambda x: (x[2], x[1]), reverse=True)
    eligible = eligible[: max(1, max_communities)]

    lines: List[str] = []
    used_evidence: List[str] = []
    extracted_facts: List[Dict[str, Any]] = []

    lines.append("Graph-derived community groupings (inferred from KG community detection):")

    for cid, members_count, coherence in eligible:
        comm_node = graph.nodes[cid]
        title = (comm_node.properties.get("title") or comm_node.properties.get("micro_summary") or "").strip()
        level = comm_node.properties.get("level")
        members = _community_members_from_graph(graph, cid)

        # Only show ENTITY/DOMAIN-like members
        shown: List[str] = []
        for mid in members:
            mnode = graph.nodes.get(mid)
            if not mnode:
                continue
            if mnode.label not in {"ENTITY", "DOMAIN"}:
                continue
            canonical = (mnode.properties.get("canonical") or mnode.properties.get("text") or "").strip()
            if canonical:
                shown.append(canonical)
            if len(shown) >= max_members_per_community:
                break

        parent_ids = _parent_communities_from_graph(graph, cid)
        parent_note = f"; part_of={parent_ids[0]}" if parent_ids else ""

        header_bits = []
        if title:
            header_bits.append(title)
        header_bits.append(cid)
        if level is not None:
            header_bits.append(f"level={level}")
        header_bits.append(f"members={members_count}")
        header_bits.append(f"coherence={coherence:.2f}")
        header = " (" + ", ".join(header_bits) + parent_note + ")"

        if shown:
            lines.append(f"- {', '.join(shown)} {header} [{cid}]")
        else:
            lines.append(f"- (members not surfaced in text index) {header} [{cid}]")

        used_evidence.append(cid)
        extracted_facts.append({
            "fact": f"Community {cid} groups {min(len(shown), max_members_per_community)} top members from the KG.",
            "evidence_ids": [cid],
        })

    # Add supporting sentence evidence ids (supporting only)
    support_sentence_ids: List[str] = []
    for cand in evidence_candidates[: max(1, top_k_for_structure)]:
        if (getattr(cand, "item_type", "") or "").upper() == "SENTENCE":
            sid = getattr(cand, "id", None)
            if sid:
                support_sentence_ids.append(sid)
    support_sentence_ids = support_sentence_ids[:5]
    if support_sentence_ids:
        used_evidence.extend([sid for sid in support_sentence_ids if sid not in used_evidence])

    answer_text = "\n".join(lines)
    insufficiency_note = (
        "These groupings are inferred from graph-based community detection and MEMBER_OF/PART_OF edges; "
        "they are not explicitly stated in the source text."
    )

    return {
        "answer": answer_text,
        "used_evidence": used_evidence,
        "extracted_facts": extracted_facts[:7],
        "confidence": "medium",
        "insufficiency_note": insufficiency_note,
    }


def llm_synthesize_answer(
    query: str,
    evidence_candidates: List[RetrievalCandidate],
    graph: Optional[KnowledgeGraph] = None,
    model_name: str = "gemini-2.0-flash",
    fallback_openai_model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    evidence_only: bool = False,
    max_evidence_only: int = 8,
    max_evidence_chars: int = 650,
    keep_metadata_keys: Optional[List[str]] = None,
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

    # Structural override: for community/grouping questions, community structure is valid evidence.
    # This path is deterministic and does not ask an LLM to invent groupings.
    structural_result = None
    try:
        structural_result = _graph_derived_structural_answer(query=query, graph=graph, evidence_candidates=evidence_candidates)
    except Exception as e:
        if verbose:
            print(f"[Synthesis] Structural override failed, falling back to text synthesis: {e}")
        structural_result = None

    if structural_result is not None:
        return structural_result

    # Reorder evidence to put sentence-level items first so the
    # model sees the most concrete, grounded evidence up front.
    if evidence_candidates:
        evidence_candidates = _sort_evidence_by_type_and_score(evidence_candidates)

    if evidence_only:
        trimmed = evidence_candidates[:max_evidence_only]
        return {
            "answer": "Evidence-only mode; no abstractive synthesis performed.",
            "used_evidence": [c.id for c in trimmed],
            "extracted_facts": [{"fact": c.text[:240], "evidence_ids": [c.id]} for c in trimmed],
            "confidence": "medium",
            "insufficiency_note": "Synthesis disabled (evidence-only mode)"
        }
    
    # Build cache key
    evidence_ids = [e.id for e in evidence_candidates]
    # Make cache key order-insensitive: evidence ordering can vary slightly due to upstream ranking.
    evidence_ids_sorted = sorted(evidence_ids)
    cache_key = DiskJSONCache.hash_key(
        query,
        json.dumps(evidence_ids_sorted, ensure_ascii=False),
        model_name,
        fallback_openai_model,
        f"max_evidence_chars={int(max_evidence_chars)}",
        f"keep_metadata_keys={','.join(keep_metadata_keys or [])}",
    )
    
    # Check cache
    if use_cache:
        cached = _SYNTHESIS_CACHE.get(cache_key)
        if cached:
            if verbose:
                print("[Synthesis] Using cached synthesis result")
            return cached
    
    # Prepare evidence for LLM
    if keep_metadata_keys is None:
        keep_metadata_keys = [
            "doc_id",
            "index",
            "canonical",
            "type",
            "entity_id",
            "sent_id",
            "level",
            "coherence",
            "members_count",
            "sample_entities",
        ]

    evidence_items = []
    max_chars = max(50, int(max_evidence_chars))
    for i, cand in enumerate(evidence_candidates):
        md: Any = cand.metadata
        safe_md: Any = None
        if isinstance(md, dict):
            safe_md = {k: md.get(k) for k in keep_metadata_keys if k in md}
        evidence_items.append({
            "id": cand.id,
            "type": cand.item_type,
            "text": (cand.text or "")[:max_chars],
            "metadata": safe_md
        })
    
    # Build prompt
    user_content = {
        "question": query,
        "evidence": evidence_items
    }
    
    prompt = f"{SYNTHESIS_PROMPT}\n\nInput:\n{json.dumps(user_content, ensure_ascii=False)}"
    
    if verbose:
        print(f"[Synthesis] Calling LLM to synthesize answer from {len(evidence_candidates)} evidence items...")
    
    # Call LLM (Gemini first, OpenAI fallback)
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
            _SYNTHESIS_CACHE.set(cache_key, {**synthesis_result, "provider": provider_used, "model": model_used})
        
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
