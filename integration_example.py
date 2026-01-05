# integration_example.py
"""
Integration example showing how to use the enhanced hybrid retrieval system.

This module demonstrates:
1. Building the enhanced retrieval index
2. Performing hybrid search with graph expansion
3. Reranking candidates with LLM
4. Synthesizing grounded answers with evidence citation
"""
from typing import Dict, List
from data_corpus import KnowledgeGraph
from phase8_retrieval_enhanced import build_retrieval_index_enhanced
from hybrid_search import search_and_expand
from llm_rerank import llm_rerank_candidates
from llm_synthesis import llm_synthesize_answer, format_answer_output


def answer_query_with_hybrid_retrieval(
    query: str,
    graph: KnowledgeGraph,
    top_n_semantic: int = 20,
    top_k_final: int = 40,
    top_k_rerank: int = 12,
    alpha: float = 0.7,
    beta: float = 0.3,
    expansion_hops: int = 1,
    verbose: bool = True
) -> Dict:
    """
    Answer a query using the full hybrid retrieval pipeline.
    
    Parameters
    ----------
    query : str
        User question
    graph : KnowledgeGraph
        The populated knowledge graph
    top_n_semantic : int
        Number of items to retrieve via semantic search
    top_k_final : int
        Number of candidates after graph expansion
    top_k_rerank : int
        Number of items to keep after LLM reranking
    alpha : float
        Weight for semantic score (0-1)
    beta : float
        Weight for graph score (0-1)
    expansion_hops : int
        Number of hops for graph expansion (1 or 2)
    verbose : bool
        Whether to print progress
    
    Returns
    -------
    Dict containing:
        - answer: str
        - used_evidence: List[str]
        - extracted_facts: List[Dict]
        - confidence: str
        - rerank_rationale: str
        - all_candidates: List[RetrievalCandidate]
        - ranked_candidates: List[RetrievalCandidate]
    """
    # Step 1: Build retrieval index (if not already built)
    if verbose:
        print("=" * 80)
        print("STEP 1: Building Enhanced Retrieval Index")
        print("=" * 80)
    
    index_items, embeddings = build_retrieval_index_enhanced(
        graph,
        context_window_tokens=50,
        include_entity_contexts=True,
        verbose=verbose
    )
    
    if not index_items:
        return {
            "answer": "No index items available. Please build the knowledge graph first.",
            "used_evidence": [],
            "extracted_facts": [],
            "confidence": "low",
            "rerank_rationale": "",
            "all_candidates": [],
            "ranked_candidates": []
        }
    
    # Step 2: Hybrid search with graph expansion
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 2: Hybrid Search (Semantic + Graph Expansion)")
        print("=" * 80)
    
    candidates = search_and_expand(
        query=query,
        graph=graph,
        index_items=index_items,
        embeddings=embeddings,
        top_n_semantic=top_n_semantic,
        top_k_final=top_k_final,
        alpha=alpha,
        beta=beta,
        expansion_hops=expansion_hops,
        include_community_expansion=True,
        verbose=verbose
    )
    
    if verbose:
        print(f"\nRetrieved {len(candidates)} candidates")
        print("\nTop 5 candidates:")
        for i, cand in enumerate(candidates[:5], 1):
            print(f"  {i}. [{cand.id}] ({cand.item_type}) - hybrid_score={cand.hybrid_score:.3f}")
            print(f"     {cand.text[:100]}...")
    
    # Step 3: LLM Reranking
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 3: LLM-based Reranking")
        print("=" * 80)
    
    rerank_result = llm_rerank_candidates(
        query=query,
        candidates=candidates,
        top_k=top_k_rerank,
        use_cache=True,
        verbose=verbose
    )
    
    ranked_candidates = rerank_result["ranked_candidates"]
    rerank_rationale = rerank_result["rationale"]
    
    if verbose:
        print(f"\nReranked to top {len(ranked_candidates)} items")
        print(f"Rationale: {rerank_rationale[:200]}...")
    
    # Step 4: LLM Synthesis with Grounding
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 4: LLM Synthesis with Evidence Grounding")
        print("=" * 80)
    
    synthesis_result = llm_synthesize_answer(
        query=query,
        evidence_candidates=ranked_candidates,
        graph=graph,
        use_cache=True,
        verbose=verbose
    )
    
    # Combine results
    result = {
        **synthesis_result,
        "rerank_rationale": rerank_rationale,
        "all_candidates": candidates,
        "ranked_candidates": ranked_candidates
    }
    
    return result


def integration_example():
    """
    Example integration showing how to use the enhanced retrieval in main_pipeline.py
    """
    print("""
    # ===================================================================
    # INTEGRATION EXAMPLE: Enhanced Hybrid Retrieval
    # ===================================================================
    
    ## Step 1: Import modules in main_pipeline.py
    
    ```python
    from phase8_retrieval_enhanced import build_retrieval_index_enhanced
    from hybrid_search import search_and_expand
    from llm_rerank import llm_rerank_candidates
    from llm_synthesis import llm_synthesize_answer, format_answer_output
    ```
    
    ## Step 2: Build index after community detection
    
    ```python
    # After Phase 7 (community detection), add:
    
    print("[Phase 8] Building enhanced retrieval index...")
    index_items, embeddings = build_retrieval_index_enhanced(
        graph,
        context_window_tokens=50,
        include_entity_contexts=True,
        verbose=True
    )
    ```
    
    ## Step 3: Answer queries
    
    ```python
    # Query the knowledge graph
    query = "What are AI agents and how do they work?"
    
    # Hybrid search
    candidates = search_and_expand(
        query=query,
        graph=graph,
        index_items=index_items,
        embeddings=embeddings,
        top_n_semantic=20,
        top_k_final=40,
        alpha=0.7,  # semantic weight
        beta=0.3,   # graph weight
        expansion_hops=1,
        verbose=True
    )
    
    # Rerank with LLM
    rerank_result = llm_rerank_candidates(
        query=query,
        candidates=candidates,
        top_k=12,
        use_cache=True,
        verbose=True
    )
    
    # Synthesize grounded answer
    synthesis_result = llm_synthesize_answer(
        query=query,
        evidence_candidates=rerank_result["ranked_candidates"],
        use_cache=True,
        verbose=True
    )
    
    # Display answer
    print(format_answer_output(synthesis_result))
    
    # Access structured results
    print("Answer:", synthesis_result["answer"])
    print("Evidence used:", synthesis_result["used_evidence"])
    print("Facts:", synthesis_result["extracted_facts"])
    print("Confidence:", synthesis_result["confidence"])
    ```
    
    ## Step 4: Using the convenience wrapper
    
    ```python
    from integration_example import answer_query_with_hybrid_retrieval
    
    # One-line query answering
    result = answer_query_with_hybrid_retrieval(
        query="What are AI agents?",
        graph=graph,
        verbose=True
    )
    
    # Display formatted answer
    print(format_answer_output(result))
    ```
    
    # ===================================================================
    # CONFIGURATION OPTIONS
    # ===================================================================
    
    ## Tuning Parameters:
    
    - `alpha` (default 0.7): Weight for semantic similarity
      - Higher = more reliance on semantic matching
      - Lower = more reliance on graph structure
    
    - `beta` (default 0.3): Weight for graph-based score
      - Should satisfy: alpha + beta ≈ 1.0
    
    - `top_n_semantic` (default 20): Initial semantic retrieval count
      - Higher = broader initial search
      - Lower = faster but may miss relevant items
    
    - `top_k_final` (default 40): Candidates after graph expansion
      - Higher = more context for reranking
      - Lower = faster reranking
    
    - `top_k_rerank` (default 12): Final evidence items for synthesis
      - Higher = more comprehensive answers
      - Lower = more focused answers
    
    - `expansion_hops` (default 1): Graph expansion depth
      - 1 = immediate neighbors only
      - 2 = neighbors of neighbors (slower, broader)
    
    - `context_window_tokens` (default 50): Context around entity mentions
      - Higher = more context per entity
      - Lower = more focused entity contexts
    
    ## Caching:
    
    All LLM calls are cached using SHA256 hashes:
    - Cluster labeling: cached per cluster items
    - Relation extraction: cached per sentence + entities
    - Community summaries: cached per entity list
    - Reranking: cached per query + candidate IDs
    - Synthesis: cached per query + evidence IDs
    
    Cache files:
    - `cache_embeddings.json`: Embedding cache
    - `cache_community_summaries.json`: Community summary cache
    - `cache_reranking.json`: Reranking cache
    - `cache_synthesis.json`: Synthesis cache
    
    ## Performance Tips:
    
    1. Reuse index_items and embeddings across queries
    2. Enable caching (use_cache=True) for reproducibility
    3. Adjust alpha/beta based on your corpus:
       - Dense semantic corpus → higher alpha
       - Rich graph structure → higher beta
    4. Use verbose=True during development
    5. Use verbose=False in production
    
    # ===================================================================
    """)


if __name__ == "__main__":
    integration_example()
