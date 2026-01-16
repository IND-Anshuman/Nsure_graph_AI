# llm_synthesis.py
"""
LLM-grounded synthesis with evidence citation.
Generates answers using only provided evidence and cites sources.
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, Set
import json
import os
import re
import unicodedata
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
from answer_synthesis.hybrid_search import RetrievalCandidate
try:
    from graph_maker.data_corpus import KnowledgeGraph  # type: ignore
except Exception:  # pragma: no cover
    # Fallback stub to keep this module importable when graph_maker isn't available.
    from dataclasses import dataclass, field

    @dataclass
    class _KGNode:
        label: str = ""
        properties: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class _KGEdge:
        source: str = ""
        target: str = ""
        type: str = ""

    @dataclass
    class KnowledgeGraph:  # type: ignore
        nodes: Dict[str, _KGNode] = field(default_factory=dict)
        edges: List[_KGEdge] = field(default_factory=list)

from utils.cache_utils import DiskJSONCache

from utils.genai_compat import generate_text as genai_generate_text


# Global cache for synthesis results; bump version to invalidate stale results
_SYNTHESIS_CACHE = DiskJSONCache("cache_synthesis_v3.json")


SYNTHESIS_PROMPT = """
You are an expert analyst synthesizing comprehensive, well-structured answers from provided evidence.
Your goal: provide a thorough, direct, and actionable answer that is grounded in evidence and formatted appropriately for the query type.

Evidence item fields: id, type ("SENTENCE", "ENTITY_CONTEXT", "ENTITY", "COMMUNITY", etc.), text, metadata.

CRITICAL RULES:
1) Ground EVERY claim strictly in the provided evidence. Zero external knowledge.
2) BE CONFIDENT AND DIRECT. State facts clearly without hedging or uncertainty unless evidence is truly ambiguous.
   - Don't use phrases like "it appears that", "it seems", "might be", "possibly" unless truly uncertain
   - Start with the answer, then provide supporting details
   - Be assertive: "X is Y" not "X appears to be Y"
3) ADAPT FORMAT TO QUERY TYPE:
   
   **Definitional queries** ("What is X?", "Define Y"):
   - Start with a clear, concise definition (1-2 sentences)
   - Follow with key characteristics in bullet points
   - Include relevant examples if available
   
   **How-to/Process queries** ("How does X work?", "How to Y"):
   - Use numbered steps or clear bullet points
   - Each step should be actionable and specific
   - Include conditions or requirements where relevant
   
   **Comparison queries** ("Compare X and Y", "Difference between"):
   - Use structured comparison format
   - Bullet points highlighting key differences/similarities
   - Include a summary statement
   
   **List/Enumeration queries** ("What are the types of X?", "List all Y"):
   - Use clear bullet points or numbered lists
   - Each item should be concise and informative
   - Group related items if applicable
   
   **Explanation queries** ("Why does X?", "Explain Y"):
   - Start with direct answer to WHY
   - Use bullet points for multiple reasons/factors
   - Provide specific examples
   
   **Relationship queries** ("How does X relate to Y?"):
   - State the relationship type upfront
   - Use bullet points to explain:
     • Direct connections
     • Mechanisms of interaction
     • Conditions or constraints
     • Practical implications
   
   **Example-based queries** ("Give example of X"):
   - Provide concrete examples first
   - Use bullet format for multiple examples
   - Include context for each example
   
   **General/Open queries**:
   - Use mixed format: opening statement + structured bullets
   - Organize information logically
   - Use subheadings or categories if needed

4) DO NOT include evidence IDs like [id1, id2] in the answer text. Write naturally without citations.
   Instead, track which evidence you use internally and list the IDs in the "used_evidence" field.

5) STRUCTURE AND CLARITY:
   - Lead with the most important information
   - Use bullet points (•) or numbered lists (1., 2., 3.) liberally
   - Keep individual bullets concise but complete
   - Use bold concepts sparingly for emphasis (e.g., **Key Point:** description)
   - Break complex information into digestible chunks
   - Add line breaks between sections for readability

6) CONTENT QUALITY:
   - Provide context and background where relevant
   - Explain WHY things work the way they do
   - Include specific details, numbers, dates from evidence
   - Use concrete examples from evidence to illustrate points
   - Connect related concepts and show relationships
   - For legal/technical content: cite specific provisions, scope, interactions, and implications

7) Prefer sentence and entity-context evidence for concrete details; use community/chunk summaries for broader context.

8) Return strict JSON only; no prose outside JSON.

Output JSON Format:
{
    "answer": "Format this based on query type:\n\n**For definitions:** Start with clear definition, then bullet key points.\n\n**For processes:** Use numbered steps (1., 2., 3.) or clear workflow.\n\n**For lists:** Use bullet points (•) for each item.\n\n**For explanations:** Lead with direct answer, then use bullets for supporting points.\n\n**For comparisons:** Use structured bullets showing contrasts.\n\n**General:** Opening statement + organized bullets with relevant subheadings.\n\nBe confident, direct, and well-organized. No hedging unless truly uncertain.",
    "used_evidence": ["id1", "id2", "id3", ...],
    "extracted_facts": [
        {"fact": "Specific factual statement from evidence", "evidence_ids": ["id1", "id2"]},
        {"fact": "Another key fact with supporting evidence", "evidence_ids": ["id3"]},
        ...
    ],
    "confidence": "high|medium|low",
    "insufficiency_note": "Only if evidence is incomplete: explain what specific information is missing"
}

Quality Expectations:
- Answer is formatted appropriately for the query type
- Information is presented confidently and directly
- Structure uses bullets, numbers, or mixed format as appropriate
- Each point is clear, specific, and actionable
- Answer is scannable and easy to digest
- 5-10 extracted facts for substantive questions
- Used evidence includes all IDs that informed your answer
"""


SUMMARY_PROMPT = """
You are an expert policy/document analyst. The user is asking for a DOCUMENT SUMMARY.

Evidence item fields: id, type ("SENTENCE", "ENTITY_CONTEXT", "ENTITY", "COMMUNITY", etc.), text, metadata.

CRITICAL RULES:
1) Use ONLY the provided evidence. Zero external knowledge.
2) Write a DETAILED, SCANNABLE summary. Do not be brief.
3) If the evidence is incomplete for a good summary, say exactly what is missing in "insufficiency_note".
4) Do NOT include evidence IDs in the answer text; list IDs only in "used_evidence".
5) Return strict JSON only; no prose outside JSON.

OUTPUT REQUIREMENTS (write the answer in this structure):
- Executive Summary (6–10 bullets)
- What the document helps the reader do (3–6 bullets)
- Key Terms / Coverage / Scope (bullets; include numbers/limits/dates if present)
- Exclusions & Constraints (bullets)
- Procedures / How to use it (numbered steps: claims, notifications, timelines, dispute steps)
- Dispute / Grievance / Escalation (bullets; arbitration/ombudsman/contacts if present)
- Checklist for the reader (5–10 bullets: what to confirm in their copy)

Output JSON Format:
{
    "answer": "...",
    "used_evidence": ["id1", "id2"],
    "extracted_facts": [
        {"fact": "...", "evidence_ids": ["id1"]}
    ],
    "confidence": "high|medium|low",
    "insufficiency_note": "Only if evidence is incomplete: explain what specific info is missing"
}
"""


def _is_summary_query(query: str) -> bool:
    """Detect summarization intent (PDF/document summary requests)."""
    q = (query or "").strip().lower()
    if not q:
        return False
    triggers = [
        "summarize",
        "summarise",
        "summary",
        "summarisation",
        "summarization",
        "tldr",
        "tl;dr",
        "overview",
        "give me a summary",
        "summarize this",
        "summarise this",
        "summarize the pdf",
        "summarise the pdf",
        "summarize this pdf",
        "summarise this pdf",
        "summarize the document",
        "summarise the document",
    ]
    return any(t in q for t in triggers)


_SUMMARY_SECTION_ORDER = [
    "Executive Summary",
    "What the document helps the reader do",
    "Key Terms / Coverage / Scope",
    "Exclusions & Constraints",
    "Procedures / How to use it",
    "Dispute / Grievance / Escalation",
    "Checklist for the reader",
]


def _format_section_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        lines: List[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                txt = item.strip()
                if txt:
                    lines.append(f"• {txt}")
            else:
                lines.append(f"• {json.dumps(item, ensure_ascii=False)}")
        return "\n".join(lines).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value).strip()


def _coerce_answer_to_string(answer: Any) -> str:
    """Coerce LLM `answer` field into a renderable string.

    Some models return a structured object for summaries (sections as keys).
    The UI expects a string; this guarantees that.
    """
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer

    if isinstance(answer, dict):
        sections: List[str] = []
        used: Set[str] = set()

        for key in _SUMMARY_SECTION_ORDER:
            if key in answer and key not in used:
                used.add(key)
                body = _format_section_value(answer.get(key))
                if body:
                    sections.append(f"{key}:\n{body}")

        for key in sorted([k for k in answer.keys() if k not in used], key=lambda s: str(s).lower()):
            body = _format_section_value(answer.get(key))
            if body:
                sections.append(f"{key}:\n{body}")

        if sections:
            return "\n\n".join(sections).strip()

        return json.dumps(answer, ensure_ascii=False, indent=2)

    if isinstance(answer, list):
        body = _format_section_value(answer)
        return body if body else json.dumps(answer, ensure_ascii=False)

    return str(answer)


def _sanitize_evidence_text(text: str) -> str:
    """Light cleanup for common PDF/OCR artifacts before synthesis.

    This reduces repeated-letter noise like 'SSSSBBBBIIII' -> 'SBI'.
    It's intentionally conservative: only collapses 3+ repeated ASCII letters.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"([A-Za-z])\1{2,}", r"\1", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _sort_evidence_by_type_and_score(evidence_candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
    """Sort evidence so that sentence-level items come first.

    Priority order (best to worst):
    SENTENCE / ENTITY_CONTEXT -> ENTITY -> CHUNK -> COMMUNITY / other.
    Within the same type, fall back to semantic_score when available.
    """

    def _type_priority(t: str) -> int:
        t = (t or "").upper()
        if t in {"SENTENCE", "ENTITY_CONTEXT", "RELATION"}:
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
            text = genai_generate_text(prefer_model, prompt, temperature=temperature, purpose="QA")
            return (text or "").strip(), "google", prefer_model
        except Exception as e:
            last_error = e
            if verbose:
                print(f"[Synthesis] Gemini failed, will try OpenAI fallback: {e}")

    if os.getenv("OPENAI_API_KEY") and OpenAI is not None:
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


def _expand_letter_range(a: str, b: str) -> List[str]:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b or len(a) != 1 or len(b) != 1:
        return []
    if not a.isalpha() or not b.isalpha():
        return []
    start = ord(a)
    end = ord(b)
    if start > end:
        start, end = end, start
    return [chr(c) for c in range(start, end + 1)]


def _query_article_node_ids(query: str) -> Set[str]:
    """Extract article node ids from the query (art:3, art:31a, ...)."""
    q = query or ""
    ids: Set[str] = set()

    for m in re.finditer(r"\bArticle\s+(\d{1,3})([A-Za-z]?)\b", q, flags=re.IGNORECASE):
        num = m.group(1)
        suf = (m.group(2) or "").lower()
        ids.add(f"art:{num}{suf}")

    for m in re.finditer(r"\b(\d{1,3})([A-Za-z])\s*[\u2013\u2014\-]\s*(\d{1,3})([A-Za-z])\b", q):
        n1, a1, n2, a2 = m.group(1), m.group(2), m.group(3), m.group(4)
        if n1 == n2:
            for letter in _expand_letter_range(a1, a2):
                ids.add(f"art:{n1}{letter}")
        else:
            ids.add(f"art:{n1}{a1.lower()}")
            ids.add(f"art:{n2}{a2.lower()}")

    return ids


def _required_relation_types_for_query(query: str) -> Set[str]:
    q = (query or "").lower()
    req: Set[str] = set()

    if "article 3" in q and ("safeguard" in q or "views" in q or "consider" in q or "recommend" in q or "consult" in q):
        req |= {"REQUIRES_RECOMMENDATION_FROM", "REQUIRES_CONSULTATION_WITH", "CONSIDERS_VIEWS_OF"}

    if "article 13" in q and ("31a" in q or "31b" in q or "31c" in q or "31a" in q.replace(" ", "")):
        req |= {"OVERRIDES", "SAVES_LAWS_FROM_INVALIDATION", "SUBJECT_TO", "LIMITS"}

    return req


def _graph_has_required_relations(graph: KnowledgeGraph, node_ids: Set[str], required_types: Set[str]) -> Tuple[bool, Dict[str, int]]:
    if not graph or not node_ids or not required_types:
        return True, {}
    counts: Dict[str, int] = {t: 0 for t in required_types}
    for e in graph.edges:
        if e.type not in required_types:
            continue
        if e.source in node_ids or e.target in node_ids:
            counts[e.type] = counts.get(e.type, 0) + 1
    ok = all(counts.get(t, 0) > 0 for t in required_types)
    return ok, counts


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
    max_evidence_only: int = 20,  # increased from 8 to capture more nuanced relationships
    max_evidence_chars: int = 1200,  # increased from 650 for richer legal context
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

    summary_mode = _is_summary_query(query)

    # In summary mode, we want broader coverage and less creativity.
    # Note: Evidence is still the limiting factor; this increases evidence budget.
    if summary_mode:
        max_evidence_chars = max(int(max_evidence_chars), 2200)
        temperature = min(float(temperature), 0.15)

    if evidence_only:
        trimmed = evidence_candidates[:max_evidence_only]
        return {
            "answer": "Evidence-only mode; no abstractive synthesis performed.",
            "used_evidence": [c.id for c in trimmed],
            "extracted_facts": [{"fact": _sanitize_evidence_text((c.text or "")[:240]), "evidence_ids": [c.id]} for c in trimmed],
            "confidence": "medium",
            "insufficiency_note": "Synthesis disabled (evidence-only mode)"
        }
    
    # Build cache key
    evidence_ids = [e.id for e in evidence_candidates]
    # Make cache key order-insensitive: evidence ordering can vary slightly due to upstream ranking.
    evidence_ids_sorted = sorted(evidence_ids)
    cache_key = DiskJSONCache.hash_key(
            "synth_v8",
        query,
        json.dumps(evidence_ids_sorted, ensure_ascii=False),
        model_name,
        fallback_openai_model,
        f"max_evidence_chars={int(max_evidence_chars)}",
        f"summary_mode={str(summary_mode)}",
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

    # Cap evidence items to keep prompt size bounded.
    # Summary mode includes more items to improve coverage.
    max_items = 40 if summary_mode else 25

    evidence_items = []
    max_chars = max(50, int(max_evidence_chars))
    for i, cand in enumerate((evidence_candidates or [])[:max_items]):
        md: Any = cand.metadata
        safe_md: Any = None
        if isinstance(md, dict):
            safe_md = {k: md.get(k) for k in keep_metadata_keys if k in md}
        evidence_items.append({
            "id": cand.id,
            "type": cand.item_type,
            "text": _sanitize_evidence_text(cand.text or "")[:max_chars],
            "metadata": safe_md
        })
    
    # Build prompt
    user_content = {
        "question": query,
        "task": "summarize_document" if summary_mode else "answer_question",
        "evidence": evidence_items,
    }

    active_prompt = SUMMARY_PROMPT if summary_mode else SYNTHESIS_PROMPT
    prompt = f"{active_prompt}\n\nInput:\n{json.dumps(user_content, ensure_ascii=False)}"
    
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
        
        if not raw or not raw.strip():
            raise ValueError("LLM returned empty response")
        
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        
        if not raw or not raw.strip():
            raise ValueError("No JSON content found in LLM response after cleaning")
            
        result = json.loads(raw)
        
        # Validate and extract fields
        answer = _coerce_answer_to_string(result.get("answer", "Unable to generate answer"))
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

        # Post-check: if question requires graph relations but KG lacks them, downgrade.
        try:
            if graph is not None:
                node_ids = _query_article_node_ids(query)
                required_types = _required_relation_types_for_query(query)
                ok, counts = _graph_has_required_relations(graph, node_ids, required_types)
                if not ok:
                    synthesis_result["confidence"] = "low"
                    missing = [t for t in sorted(required_types) if counts.get(t, 0) <= 0]
                    note_bits = []
                    note_bits.append("Graph lacks required legal relation edges for this question")
                    if missing:
                        note_bits.append("missing=" + ",".join(missing))
                    if node_ids:
                        note_bits.append("entities=" + ",".join(sorted(node_ids)))
                    extra = "; ".join(note_bits)
                    prior = synthesis_result.get("insufficiency_note")
                    synthesis_result["insufficiency_note"] = (str(prior) + " | " + extra) if prior else extra
        except Exception:
            pass
        
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
