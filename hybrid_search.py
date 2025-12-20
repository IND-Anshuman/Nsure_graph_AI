# hybrid_search.py
"""
Hybrid retrieval with dense semantic search + graph-based expansion.
Combines semantic similarity scores with graph structure scores.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Any, Optional
from dataclasses import dataclass
import numpy as np
from data_corpus import KnowledgeGraph
from phase8_retrieval_enhanced import IndexItem
from embedding_cache import get_embeddings_with_cache, compute_cosine_similarity


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
    top_indices = np.argsort(-semantic_scores)[:top_n_semantic]
    
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
            candidates[exp_id].graph_score += 1.0
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
    
    # Get immediate neighbors
    for edge in graph.edges:
        if edge.type in ["CO_OCCURS_WITH", "USED_WITH", "DEPLOYS_ON", "PROPOSED", 
                        "EVALUATED_ON", "PRESENTED_AT", "PUBLISHED_AT", "LOCATED_IN"]:
            target_node = graph.nodes.get(edge.target)
            source_node = graph.nodes.get(edge.source)
            if edge.source == entity_id and target_node and target_node.label == "ENTITY":
                neighbors.add(edge.target)
            elif edge.target == entity_id and source_node and source_node.label == "ENTITY":
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
            if source_node and source_node.label == "ENTITY":
                entities.add(edge.source)
    return entities
