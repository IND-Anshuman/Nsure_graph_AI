# hybrid_search.py
"""
Hybrid retrieval with dense semantic search + graph-based expansion.
Combines semantic similarity scores with graph structure scores.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Any, Optional
from dataclasses import dataclass
import numpy as np
import re
import os
from graph_maker.data_corpus import KnowledgeGraph
from answer_synthesis.retrieval import IndexItem
from graph_maker.embedding_cache import get_embeddings_with_cache, compute_cosine_similarity
from graph_maker.relation_schema import load_relation_schema, relation_edge_types_for_retrieval


@dataclass
class RetrievalCandidate:
    """
    Represents a candidate item retrieved during hybrid search.
    """
    id: str                      # item id
    text: str                    # item text
    item_type: str               # "SENTENCE", "ENTITY", "COMMUNITY", etc.
    semantic_score: float        # cosine similarity to query
    graph_score: float           # score from graph expansion
    hybrid_score: float          # combined score
    cross_score: float = 0.0     # optional cross-encoder score
    metadata: Optional[Dict[str, Any]] = None     # additional metadata
    retrieval_path: str = ""          # how this was retrieved ("semantic", "expanded", "community")


def search_and_expand(
    query: str,
    graph: KnowledgeGraph,
    index_items: List[IndexItem],
    embeddings: np.ndarray,
    top_n_semantic: int = 20,
    top_k_final: int = 40,
    alpha: float = 0.7,
    beta: float = 0.3,
    expansion_hops: int = 1,
    include_community_expansion: bool = True,
    min_hybrid_score: float = 0.15,
    dedup_overlap_threshold: float = 0.9,
    semantic_dedup_threshold: float = 0.88,
    community_boost: float = 0.15,
    verbose: bool = False
) -> List[RetrievalCandidate]:
    """
    Hybrid retrieval: dense semantic retrieval + graph-based expansion.
    
    Parameters
    ----------
    query : str
        User query
    graph : KnowledgeGraph
        The knowledge graph
    index_items : List[IndexItem]
        Index items from build_retrieval_index_enhanced
    embeddings : np.ndarray
        Pre-computed embeddings for index items
    top_n_semantic : int
        Number of items to retrieve via semantic search
    top_k_final : int
        Final number of candidates to return after deduplication
    alpha : float
        Weight for semantic score (default 0.7)
    beta : float
        Weight for graph score (default 0.3)
    expansion_hops : int
        Number of hops for graph expansion (1 or 2)
    include_community_expansion : bool
        Whether to expand communities to their entities
    verbose : bool
        Whether to print debug info
    
    Returns
    -------
    List[RetrievalCandidate]
        Top-k candidates ranked by hybrid score
    """
    # 1) Dense semantic retrieval
    query_emb = get_embeddings_with_cache([query])[0]
    semantic_scores = compute_cosine_similarity(query_emb, embeddings)
    
    # Get top-N indices by semantic score
    top_indices = list(np.argsort(-semantic_scores)[:top_n_semantic])

    # 1b) Legal-citation lexical boost (critical for law PDFs)
    # If the query names an Article/Section/etc, force-include matching sentences/chunks.
    def _extract_citations(q: str) -> List[str]:
        q = (q or "")
        if not q:
            return []

        out: List[str] = []
        # Singular/plural keyword + number (+ optional subsections)
        citation_rx = re.compile(
            r"\b(articles?|sections?|clauses?|chapters?|parts?|schedules?|rules?|regulations?)\s+\d+[a-z]?(?:\([0-9a-z]+\))*\b",
            flags=re.IGNORECASE,
        )
        for m in citation_rx.finditer(q):
            out.append(m.group(0).strip().lower())

        # Expand ranges like '31A–31C' when context suggests articles.
        range_rx = re.compile(r"\b(\d+)([A-Za-z])\s*[\-\u2013\u2014]\s*(\d+)?([A-Za-z])\b")
        for m in range_rx.finditer(q):
            n1 = m.group(1)
            a1 = (m.group(2) or "").upper()
            n2 = m.group(3) or n1
            a2 = (m.group(4) or "").upper()
            if n1 != n2 or not (a1.isalpha() and a2.isalpha()):
                continue
            start = ord(a1)
            end = ord(a2)
            if start > end:
                start, end = end, start
            if end - start > 10:
                continue
            for code in range(start, end + 1):
                out.append(f"article {n1}{chr(code).lower()}")

        # Also capture bare tokens like '31A', '31B', '31C' and treat as article refs.
        # This helps when the query says 'Articles 13 and 31A–31C'.
        # IMPORTANT: be very careful to include single-digit articles like "Article 3"
        for m in re.finditer(r"\b(article|articles|section|sections)\s+(\d{1,3})(?:[a-z])?(?:\([0-9a-z\-\s]+\))?\b", q, re.IGNORECASE):
            ref = m.group(2)
            out.append(f"article {ref}")
            out.append(f"article {ref.lower()}")
        
        # Also match bare numbers in context like "Article 3 of the Constitution"
        for m in re.finditer(r"\b(\d{1,3})([A-Za-z]?)\b", q):
            num = m.group(1)
            letter = (m.group(2) or "").lower()
            if letter:
                out.append(f"article {num}{letter}")
            else:
                out.append(f"article {num}")

        # Normalize 'articles 13' -> also include 'article 13'
        normed: List[str] = []
        for c in out:
            c2 = c
            c2 = c2.replace("articles ", "article ")
            c2 = c2.replace("sections ", "section ")
            c2 = c2.replace("clauses ", "clause ")
            c2 = c2.replace("chapters ", "chapter ")
            c2 = c2.replace("parts ", "part ")
            c2 = c2.replace("schedules ", "schedule ")
            c2 = c2.replace("rules ", "rule ")
            c2 = c2.replace("regulations ", "regulation ")
            normed.append(c2)

        # Unique preserve order
        seen = set()
        uniq = []
        for c in normed:
            if c and c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    def _compile_citation_patterns(cits: List[str]) -> List[re.Pattern]:
        patterns: List[re.Pattern] = []
        for cit in cits:
            cit = (cit or "").strip().lower()
            if not cit:
                continue

            # Make matching robust to whitespace/underscore differences while staying boundary-safe.
            # This prevents false positives like "article 3" matching "article 32".
            parts = [p for p in re.split(r"\s+", cit) if p]
            if not parts:
                continue
            body = r"(?:\\s+|_)+".join(re.escape(p) for p in parts)
            patterns.append(re.compile(rf"(?<!\\w){body}(?!\\w)", flags=re.IGNORECASE))

            # Extra robustness for constitutions/PDFs that format headings as just:
            #   `3.` or `3.—` rather than `Article 3`.
            # If the citation is an article/section/clause with a number, also match
            # a line-start numbered heading. This is multiline and boundary-safe.
            m = re.match(r"^(article|section|clause|chapter|part|schedule|rule|regulation)\s+(\d{1,3})([a-z]?)$", cit)
            if m:
                num = m.group(2)
                suf = m.group(3) or ""
                # e.g. "3." / "3.—" / "3-" at the start of a line (allow small indentation)
                patterns.append(
                    re.compile(
                        rf"(?m)^(?:[ \t]{{0,6}}){re.escape(num + suf)}\s*(?:[\.\-\u2013\u2014]{{1,2}})\s*",
                        flags=re.IGNORECASE,
                    )
                )
                # Also match short forms like "Art. 3"
                patterns.append(re.compile(rf"(?<!\\w)art\.?\s*{re.escape(num + suf)}(?!\\w)", flags=re.IGNORECASE))
        return patterns

    citations = _extract_citations(query)
    citation_patterns = _compile_citation_patterns(citations)

    # 1c) Hard force-inclusion for provision text + relation facts.
    # Lexical matching helps, but in PDFs the actual Article text is often headed as
    # "3." rather than "Article 3", so the best evidence can be missed by pure lexical
    # match + semantic ranking. When the query cites Articles, inject:
    # - PROVISION_CONTEXT windows for those article entity_ids (art:*)
    # - RELATION items where source/target matches those article ids
    def _citation_to_article_id(cit: str) -> Optional[str]:
        cit = (cit or "").strip().lower()
        m = re.match(r"^article\s+(\d{1,3})([a-z]?)$", cit)
        if not m:
            return None
        num = m.group(1)
        suf = (m.group(2) or "").lower()
        return f"art:{num}{suf}".strip(":")

    cited_article_ids: List[str] = []
    seen_a: Set[str] = set()
    for c in citations:
        aid = _citation_to_article_id(c)
        if aid and aid not in seen_a:
            seen_a.add(aid)
            cited_article_ids.append(aid)

    if cited_article_ids:
        max_ctx_per_article = int(os.getenv("KG_FORCE_CTX_PER_ARTICLE", "6"))
        max_rel_per_article = int(os.getenv("KG_FORCE_REL_PER_ARTICLE", "20"))

        ctx_taken: Dict[str, int] = {a: 0 for a in cited_article_ids}
        rel_taken: Dict[str, int] = {a: 0 for a in cited_article_ids}

        forced_extra: List[int] = []
        for idx, it in enumerate(index_items):
            meta = it.metadata or {}
            if it.item_type == "PROVISION_CONTEXT":
                eid = str(meta.get("entity_id") or "")
                if eid in ctx_taken and ctx_taken[eid] < max_ctx_per_article:
                    forced_extra.append(idx)
                    ctx_taken[eid] += 1
            elif it.item_type == "RELATION":
                src = str(meta.get("source") or "")
                tgt = str(meta.get("target") or "")
                # Attribute relation evidence to any cited article it touches.
                for a in cited_article_ids:
                    if rel_taken[a] >= max_rel_per_article:
                        continue
                    if src == a or tgt == a:
                        forced_extra.append(idx)
                        rel_taken[a] += 1
                        break

        # Merge forced extras into top_indices and apply a score floor so they survive.
        for idx in forced_extra:
            if idx not in top_indices:
                top_indices.append(idx)
            it = index_items[idx]
            if it.item_type == "PROVISION_CONTEXT":
                semantic_scores[idx] = max(float(semantic_scores[idx]), 0.78)
            elif it.item_type == "RELATION":
                semantic_scores[idx] = max(float(semantic_scores[idx]), 0.72)
            elif it.item_type == "ENTITY":
                semantic_scores[idx] = max(float(semantic_scores[idx]), 0.62)
    if citation_patterns:
        id_to_index = {item.id: idx for idx, item in enumerate(index_items)}
        extra = []
        for item in index_items:
            if item.item_type not in {"SENTENCE", "CHUNK", "ENTITY", "ENTITY_CONTEXT", "PROVISION_CONTEXT"}:
                continue

            txt = (item.text or "").lower()
            meta = item.metadata or {}
            meta_canon = str(meta.get("canonical") or "").lower()
            meta_entity_id = str(meta.get("entity_id") or "").lower()
            haystack = " ".join([txt, meta_canon, meta_entity_id, str(item.id).lower()])

            if any(p.search(haystack) for p in citation_patterns):
                extra.append(id_to_index[item.id])
        # Merge extras into top_indices
        for idx in extra:
            if idx not in top_indices:
                top_indices.append(idx)
        # Also lightly boost semantic score for those extras
        for idx in extra:
            it = index_items[idx]
            floor = 0.65 if it.item_type == "PROVISION_CONTEXT" else 0.55
            semantic_scores[idx] = max(float(semantic_scores[idx]), floor)
    
    if verbose:
        print(f"[Hybrid Search] Retrieved {len(top_indices)} items via semantic search")
    
    def _community_level_weight(meta: Optional[Dict[str, Any]]) -> float:
        if not meta:
            return 1.0
        level_raw = meta.get("level")
        try:
            level = int(level_raw) if level_raw is not None else 0
        except Exception:
            level = 0
        # Prefer more concrete, lower-level communities.
        # level 0 -> 1.2, level 1 -> 1.0, level 2 -> 0.8, >=3 -> 0.8
        if level <= 0:
            return 1.2
        if level == 1:
            return 1.0
        return 0.8

    def _community_coherence_weight(meta: Optional[Dict[str, Any]]) -> float:
        if not meta:
            return 1.0
        try:
            coh = float(meta.get("coherence", 0.0) or 0.0)
        except Exception:
            coh = 0.0
        # map coherence in [0,1] to [0.8,1.2]
        return 0.8 + 0.4 * max(0.0, min(1.0, coh))

    def _base_graph_score_for_item(item: IndexItem, default_non_comm: float = 0.0) -> float:
        if item.item_type.startswith("COMMUNITY"):
            w_level = _community_level_weight(item.metadata)
            w_coh = _community_coherence_weight(item.metadata)
            return community_boost * w_level * w_coh
        return default_non_comm

    # 2) Build initial candidate set
    candidates: Dict[str, RetrievalCandidate] = {}
    id_to_index = {item.id: idx for idx, item in enumerate(index_items)}
    
    for idx in top_indices:
        item = index_items[idx]
        score = float(semantic_scores[idx])
        
        base_graph_score = _base_graph_score_for_item(item, default_non_comm=0.0)
        candidates[item.id] = RetrievalCandidate(
            id=item.id,
            text=item.text,
            item_type=item.item_type,
            semantic_score=score,
            graph_score=base_graph_score,
            hybrid_score=alpha * score + beta * base_graph_score,
            metadata=item.metadata,
            retrieval_path="semantic"
        )
    
    # 3) Graph expansion
    expansion_set: Set[str] = set()
    
    for item_id in list(candidates.keys()):
        item = candidates[item_id]
        
        # Expand based on item type
        if item.item_type == "ENTITY":
            # Add 1-hop neighbor entities
            neighbors = _get_entity_neighbors(graph, item_id, hops=expansion_hops)
            expansion_set.update(neighbors)
            
            # Add mentioning sentences
            mention_sents = _get_mentioning_sentences(graph, item_id)
            expansion_set.update(mention_sents)
            
            # Add parent communities
            parent_comms = _get_parent_communities(graph, item_id)
            expansion_set.update(parent_comms)
        
        elif item.item_type == "COMMUNITY" and include_community_expansion:
            # Add sample entities from community
            sample_entities = item.metadata.get("sample_entities", [])
            for ent_id in sample_entities:
                if ent_id in graph.nodes:
                    expansion_set.add(ent_id)
        
        elif item.item_type == "SENTENCE":
            # Add entities mentioned in this sentence
            sent_entities = _get_sentence_entities(graph, item_id)
            expansion_set.update(sent_entities)
    
    if verbose:
        print(f"[Hybrid Search] Expanded to {len(expansion_set)} additional items")
    
    # 4) Add expanded items to candidates
    # Build mapping from id to index item
    id_to_item = {item.id: item for item in index_items}

    for exp_id in expansion_set:
        if exp_id in candidates:
            # Already in candidates, boost graph score
            # Keep boosts modest so expansion doesn't drown semantic retrieval.
            # Entities/communities benefit more than raw sentences.
            if candidates[exp_id].item_type in {"ENTITY", "COMMUNITY", "COMMUNITY_MICRO", "COMMUNITY_BULLETS"}:
                candidates[exp_id].graph_score += 0.6
            else:
                candidates[exp_id].graph_score += 0.25
        else:
            # Add new candidate
            if exp_id in id_to_item:
                item = id_to_item[exp_id]
                # Get semantic score (already computed if in index)
                item_idx = id_to_index.get(exp_id)
                sem_score = float(semantic_scores[item_idx]) if item_idx is not None else 0.0
                base_graph_score = _base_graph_score_for_item(item, default_non_comm=1.0)
                
                candidates[exp_id] = RetrievalCandidate(
                    id=item.id,
                    text=item.text,
                    item_type=item.item_type,
                    semantic_score=sem_score,
                    graph_score=base_graph_score,
                    hybrid_score=alpha * sem_score + beta * base_graph_score,
                    metadata=item.metadata,
                    retrieval_path="expanded"
                )
            elif exp_id in graph.nodes:
                # Not in index but in graph - add with minimal info
                node = graph.nodes[exp_id]
                text = node.properties.get("text") or node.properties.get("summary") or node.properties.get("title") or ""
                
                candidates[exp_id] = RetrievalCandidate(
                    id=exp_id,
                    text=text,
                    item_type=node.label,
                    semantic_score=0.0,
                    graph_score=1.0,
                    hybrid_score=beta * 1.0,
                    metadata=node.properties,
                    retrieval_path="expanded"
                )
    
    # 5) Recompute hybrid scores
    for cand in candidates.values():
        cand.hybrid_score = alpha * cand.semantic_score + beta * cand.graph_score
    
    # 6) Sort, filter by min score, and deduplicate similar texts
    sorted_candidates = sorted(candidates.values(), key=lambda c: c.hybrid_score, reverse=True)

    def _is_duplicate(text: str, kept: List[str], threshold: float) -> bool:
        tokens = set(text.lower().split())
        if not tokens:
            return False
        for other in kept:
            other_tokens = set(other.lower().split())
            if not other_tokens:
                continue
            overlap = len(tokens & other_tokens) / max(1, len(tokens | other_tokens))
            if overlap >= threshold:
                return True
        return False

    def _semantic_duplicate(cand: RetrievalCandidate, kept_candidates: List[RetrievalCandidate], threshold: float) -> bool:
        if not kept_candidates:
            return False
        texts = [c.text for c in kept_candidates] + [cand.text]
        embs = get_embeddings_with_cache(texts)
        new_emb = embs[-1]
        prev_embs = embs[:-1]
        sims = compute_cosine_similarity(new_emb, prev_embs)
        return float(np.max(sims)) >= threshold if len(sims) > 0 else False

    final: List[RetrievalCandidate] = []
    kept_texts: List[str] = []

    for cand in sorted_candidates:
        if cand.hybrid_score < min_hybrid_score:
            continue
        # Do not dedupe ENTITY (and closely related) items: for legal QA, the entity
        # identifier itself (e.g., 'article 3') is a high-value anchor even when
        # the text overlaps with sentences/chunks.
        if cand.item_type not in {"ENTITY", "ENTITY_CONTEXT", "PROVISION_CONTEXT", "RELATION"}:
            if _is_duplicate(cand.text, kept_texts, dedup_overlap_threshold):
                continue
            if _semantic_duplicate(cand, final, semantic_dedup_threshold):
                continue
        final.append(cand)
        kept_texts.append(cand.text)
        if len(final) >= top_k_final:
            break

    if verbose:
        print(f"[Hybrid Search] Returning top {len(final)} candidates (from {len(sorted_candidates)} total)")
    
    return final


def _get_entity_neighbors(graph: KnowledgeGraph, entity_id: str, hops: int = 1) -> Set[str]:
    """
    Get neighboring entities connected via semantic relations or co-occurrence.
    """
    neighbors = set()
    
    # Get immediate neighbors.
    # Prefer a single, shared allowlist from relation_schema.json.
    try:
        schema = load_relation_schema()
        legal_types = set(relation_edge_types_for_retrieval(schema))
        # Keep PART_OF and common insurance/legal relations as reliable evidence.
        legal_types.update({
            "PART_OF", "MEMBER_OF", "ISSUED_BY", "INSURED_BY", "LIABLE_FOR", 
            "PROVIDES_FOR", "REQUIRES", "LIMITS", "AMENDS", "SUBJECT_TO"
        })
    except Exception:
        legal_types = {
            "DEFINES",
            "PROVIDES_FOR",
            "EMPOWERS",
            "REQUIRES",
            "REQUIRES_RECOMMENDATION_FROM",
            "REQUIRES_CONSULTATION_WITH",
            "PROHIBITS",
            "LIMITS",
            "AMENDS",
            "SUBJECT_TO",
            "NOTWITHSTANDING",
            "EXCEPTS",
            "APPLIES_TO",
            "BALANCES_WITH",
            "CONSIDERS_VIEWS_OF",
            "PROCEDURE_FOR",
            "INTERPRETS",
            "RELATED_TO",
            "PART_OF",
            "OVERRIDES",
            "SAVES_LAWS_FROM_INVALIDATION",
            "ISSUED_BY",
            "INSURED_BY",
            "LIABLE_FOR",
        }

    valid_labels = {"ENTITY", "DOMAIN"}

    for edge in graph.edges:
        # Check if the edge type is a reliable semantic relation or co-occurrence
        if edge.type in {
            "CO_OCCURS_WITH",
            "USED_WITH",
            "DEPLOYS_ON",
            "PROPOSED",
            "EVALUATED_ON",
            "PRESENTED_AT",
            "PUBLISHED_AT",
            "LOCATED_IN",
        } | legal_types:
            target_node = graph.nodes.get(edge.target)
            source_node = graph.nodes.get(edge.source)
            if edge.source == entity_id and target_node and target_node.label in valid_labels:
                neighbors.add(edge.target)
            elif edge.target == entity_id and source_node and source_node.label in valid_labels:
                neighbors.add(edge.source)
    
    # For 2-hop expansion
    if hops > 1:
        second_hop = set()
        for neighbor_id in list(neighbors):
            second_hop.update(_get_entity_neighbors(graph, neighbor_id, hops=1))
        neighbors.update(second_hop)
    
    return neighbors


def _get_mentioning_sentences(graph: KnowledgeGraph, entity_id: str) -> Set[str]:
    """
    Get sentences that mention this entity.
    """
    sentences = set()
    for edge in graph.edges:
        if edge.type == "MENTION_IN" and edge.source == entity_id:
            target_node = graph.nodes.get(edge.target)
            if target_node and target_node.label == "SENTENCE":
                sentences.add(edge.target)
    return sentences


def _get_parent_communities(graph: KnowledgeGraph, entity_id: str) -> Set[str]:
    """
    Get communities that this entity belongs to (via MEMBER_OF edges).
    """
    communities = set()
    for edge in graph.edges:
        if edge.type == "MEMBER_OF" and edge.source == entity_id:
            target_node = graph.nodes.get(edge.target)
            if target_node and target_node.label == "COMMUNITY":
                communities.add(edge.target)
    return communities


def _get_sentence_entities(graph: KnowledgeGraph, sent_id: str) -> Set[str]:
    """
    Get entities mentioned in this sentence.
    """
    entities = set()
    for edge in graph.edges:
        if edge.type == "MENTION_IN" and edge.target == sent_id:
            source_node = graph.nodes.get(edge.source)
            if source_node and source_node.label in {"ENTITY", "DOMAIN"}:
                entities.add(edge.source)
    return entities
