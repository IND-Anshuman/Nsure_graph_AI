# hybrid_search.py
"""
Hybrid retrieval with dense semantic search + graph-based expansion.
Combines semantic similarity scores with graph structure scores.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Set, Any
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
    metadata: Dict[str, Any]     # additional metadata
    retrieval_path: str          # how this was retrieved ("semantic", "expanded", "community")


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
    
    # 2) Build initial candidate set
    candidates: Dict[str, RetrievalCandidate] = {}
    
    for idx in top_indices:
        item = index_items[idx]
        score = float(semantic_scores[idx])
        
        candidates[item.id] = RetrievalCandidate(
            id=item.id,
            text=item.text,
            item_type=item.item_type,
            semantic_score=score,
            graph_score=0.0,
            hybrid_score=alpha * score,  # initially only semantic
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
                item_idx = next((i for i, it in enumerate(index_items) if it.id == exp_id), None)
                sem_score = float(semantic_scores[item_idx]) if item_idx is not None else 0.0
                
                candidates[exp_id] = RetrievalCandidate(
                    id=item.id,
                    text=item.text,
                    item_type=item.item_type,
                    semantic_score=sem_score,
                    graph_score=1.0,
                    hybrid_score=alpha * sem_score + beta * 1.0,
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
    
    # 6) Sort by hybrid score and return top-k
    sorted_candidates = sorted(candidates.values(), key=lambda c: c.hybrid_score, reverse=True)
    
    if verbose:
        print(f"[Hybrid Search] Returning top {top_k_final} candidates (from {len(sorted_candidates)} total)")
    
    return sorted_candidates[:top_k_final]


def _get_entity_neighbors(graph: KnowledgeGraph, entity_id: str, hops: int = 1) -> Set[str]:
    """
    Get neighboring entities connected via semantic relations or co-occurrence.
    """
    neighbors = set()
    
    # Get immediate neighbors
    for edge in graph.edges:
        if edge.type in ["CO_OCCURS_WITH", "USED_WITH", "DEPLOYS_ON", "PROPOSED", 
                        "EVALUATED_ON", "PRESENTED_AT", "PUBLISHED_AT", "LOCATED_IN"]:
            if edge.source == entity_id and graph.nodes.get(edge.target, {}).label == "ENTITY":
                neighbors.add(edge.target)
            elif edge.target == entity_id and graph.nodes.get(edge.source, {}).label == "ENTITY":
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
            if graph.nodes.get(edge.target, {}).label == "SENTENCE":
                sentences.add(edge.target)
    return sentences


def _get_parent_communities(graph: KnowledgeGraph, entity_id: str) -> Set[str]:
    """
    Get communities that this entity belongs to (via MEMBER_OF edges).
    """
    communities = set()
    for edge in graph.edges:
        if edge.type == "MEMBER_OF" and edge.source == entity_id:
            if graph.nodes.get(edge.target, {}).label == "COMMUNITY":
                communities.add(edge.target)
    return communities


def _get_sentence_entities(graph: KnowledgeGraph, sent_id: str) -> Set[str]:
    """
    Get entities mentioned in this sentence.
    """
    entities = set()
    for edge in graph.edges:
        if edge.type == "MENTION_IN" and edge.target == sent_id:
            if graph.nodes.get(edge.source, {}).label == "ENTITY":
                entities.add(edge.source)
    return entities
