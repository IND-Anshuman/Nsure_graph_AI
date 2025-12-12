# Enhanced GraphRAG Hybrid Retrieval System

## Overview

This implementation adds a production-ready, entity-aware hybrid retrieval system with LLM-powered reranking and grounded synthesis to the Knowledge Graph Agent project.

## New Modules

### 1. `embedding_cache.py`
Centralized embedding cache using SentenceTransformers with disk persistence.

**Key Functions:**
- `get_embedding_model()`: Get the global all-MiniLM-L6-v2 model
- `get_embeddings_with_cache(texts)`: Compute embeddings with disk caching
- `compute_cosine_similarity(query_emb, doc_embs)`: Efficient similarity computation

### 2. `phase8_retrieval_enhanced.py`
Enhanced retrieval index builder that creates a unified index from multiple sources.

**Key Functions:**
- `build_retrieval_index_enhanced(graph, ...)`: Build comprehensive retrieval index

**Index Components:**
- SENTENCE nodes: Full sentence text
- ENTITY nodes: Canonical name + first alias + first description  
- ENTITY_CONTEXT: 40-60 token windows around entity mentions
- COMMUNITY summaries: All hierarchical levels (L0, L1, L2...)

**Returns:**
- `index_items`: List of IndexItem objects with metadata
- `embeddings`: Pre-computed embedding matrix

### 3. `hybrid_search.py`
Hybrid retrieval combining semantic similarity with graph structure.

**Key Functions:**
- `search_and_expand(query, graph, index_items, embeddings, ...)`: Full hybrid search

**Process:**
1. **Semantic Retrieval**: Top-N items via cosine similarity
2. **Graph Expansion**: For each retrieved item:
   - ENTITY → 1/2-hop neighbors, mentioning sentences, parent communities
   - COMMUNITY → sample entities and descriptions
   - SENTENCE → parent chunk, co-located sentences
3. **Hybrid Scoring**: `score = alpha * semantic + beta * graph_score`
4. **Deduplication**: Return top-K unique candidates

**Tunable Parameters:**
- `alpha=0.7`: Semantic score weight
- `beta=0.3`: Graph score weight  
- `top_n_semantic=20`: Initial retrieval count
- `top_k_final=40`: Final candidates after expansion
- `expansion_hops=1`: Graph expansion depth (1 or 2)

### 4. `llm_rerank.py`
LLM-powered reranking with deterministic caching.

**Key Functions:**
- `llm_rerank_candidates(query, candidates, top_k=12, ...)`: Rerank via LLM

**Features:**
- Uses Gemini 2.0 Flash for fast reranking
- Cache key: SHA256(query + candidate_ids)
- Returns: ranked candidates + rationale
- Temperature: 0.1 (deterministic)

**Output:**
```json
{
  "ranked_candidates": [...],
  "ranked_evidence_ids": ["id1", "id2", ...],
  "rationale": "Explanation of ranking..."
}
```

### 5. `llm_synthesis.py`
Grounded answer synthesis with evidence citation.

**Key Functions:**
- `llm_synthesize_answer(query, evidence_candidates, ...)`: Generate answer
- `format_answer_output(result, ...)`: Pretty-print results

**Features:**
- Evidence-only constraint: Must cite sources for every claim
- Cache key: SHA256(query + evidence_ids)
- Temperature: 0.2 (factual consistency)
- Explicit insufficiency detection

**Output:**
```json
{
  "answer": "Answer text with [evidence_id] citations...",
  "used_evidence": ["id1", "id2", ...],
  "extracted_facts": [
    {"fact": "...", "evidence_ids": ["id1", ...]},
    ...
  ],
  "confidence": "high|medium|low",
  "insufficiency_note": "Optional explanation if evidence insufficient"
}
```

### 6. `integration_example.py`
Complete integration guide with examples.

**Key Functions:**
- `answer_query_with_hybrid_retrieval(query, graph, ...)`: One-line query answering
- `integration_example()`: Prints detailed integration instructions

## Integration with main_pipeline.py

The main pipeline has been updated to use the new modules in Phase 8:

```python
# Phase 8: Enhanced Retrieval Index
index_items, embeddings = build_retrieval_index_enhanced(
    graph,
    context_window_tokens=50,
    include_entity_contexts=True,
    verbose=True
)

# Query with hybrid search
candidates = search_and_expand(
    query=query,
    graph=graph,
    index_items=index_items,
    embeddings=embeddings,
    top_n_semantic=20,
    top_k_final=40,
    alpha=0.7,
    beta=0.3,
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

# Display formatted answer
print(format_answer_output(synthesis_result))
```

## Usage Example

```python
from integration_example import answer_query_with_hybrid_retrieval
from llm_synthesis import format_answer_output

# After building your knowledge graph...
result = answer_query_with_hybrid_retrieval(
    query="What are AI agents and how do they work?",
    graph=graph,
    verbose=True
)

# Display the answer
print(format_answer_output(result))

# Access structured data
print("Confidence:", result["confidence"])
print("Evidence used:", result["used_evidence"])
for fact in result["extracted_facts"]:
    print(f"- {fact['fact']} (sources: {fact['evidence_ids']})")
```

## Caching Strategy

All expensive operations are cached using deterministic SHA256 keys:

1. **Embeddings**: `cache_embeddings.json`
   - Key: text content
   - Shared across all modules

2. **Community Summaries**: `cache_community_summaries.json`
   - Key: sorted entity canonical names
   - Used in community detection

3. **Reranking**: `cache_reranking.json`
   - Key: SHA256(query + sorted_candidate_ids)
   - Ensures deterministic rankings

4. **Synthesis**: `cache_synthesis.json`
   - Key: SHA256(query + sorted_evidence_ids)
   - Ensures consistent answers

**Benefits:**
- Repeated queries hit cache (instant response)
- Deterministic results for testing
- Reduces API costs
- Thread-safe via DiskJSONCache

## Configuration Parameters

### Hybrid Search Tuning

**Semantic vs Graph Balance:**
```python
alpha=0.7  # Semantic similarity weight
beta=0.3   # Graph structure weight
```
- High alpha (0.8-0.9): Prefer semantic matching
- Balanced (0.6-0.7): Use both signals
- High beta (0.4-0.5): Prefer graph structure

**Retrieval Depth:**
```python
top_n_semantic=20    # Initial semantic retrieval
top_k_final=40       # After graph expansion
expansion_hops=1     # Graph expansion depth (1 or 2)
```
- More semantic items = broader initial search
- More final items = richer context for reranking
- 2-hop expansion = slower but more comprehensive

### Reranking & Synthesis

```python
top_k_rerank=12      # Evidence items for synthesis
temperature=0.1      # Reranking (deterministic)
temperature=0.2      # Synthesis (factual)
```

### Index Building

```python
context_window_tokens=50        # Context around entities
include_entity_contexts=True    # Add entity contexts to index
```

## Performance Optimization

1. **Reuse Index**: Build index once, query many times
2. **Enable Caching**: Set `use_cache=True` (default)
3. **Tune alpha/beta**: Based on your corpus characteristics
4. **Adjust top_k**: Balance quality vs speed
5. **Use verbose=False**: In production

## Multi-level Community Detection

The existing `Community_processing.py` already implements hierarchical communities:

- **Level 0**: Base entity communities (Louvain on entity graph)
- **Level 1+**: Communities of communities (iterative collapse)
- **Edges**: 
  - `MEMBER_OF`: Entity → Community (L0)
  - `PART_OF`: Community (Lk) → Community (Lk+1)

Configuration in `main_pipeline.py`:
```python
community_results = compute_multilevel_communities(
    graph,
    max_levels=2,
    min_comm_size=1,
    edge_types=["CO_OCCURS_WITH", "USED_WITH", "DEPLOYS_ON"],
    verbose=True
)
```

## Expected Behavior

### End-to-End Flow

1. **Ingest** → Build DOCUMENT + SENTENCE nodes
2. **Entity Discovery** → Cluster + LLM labeling → ENTITY nodes
3. **Graph Construction** → Add MENTION_IN, CO_OCCURS_WITH edges
4. **Semantic Relations** → LLM extraction → Typed edges
5. **Communities** → Multi-level Louvain → COMMUNITY nodes + hierarchical edges
6. **Indexing** → Unified retrieval index (sentences, entities, contexts, communities)
7. **Query → Hybrid Search** → Semantic + graph expansion
8. **Rerank** → LLM selects top evidence
9. **Synthesize** → Grounded answer with citations

### Quality Improvements

**Before (naive retrieval):**
- Generic sentence matches
- No entity awareness
- No graph structure utilization
- Vague answers without citations

**After (hybrid retrieval):**
- Entity-centric retrieval
- Graph expansion finds related context
- LLM reranking selects best evidence
- Grounded answers with explicit citations
- Extracted facts with source attribution

## Testing

The implementation includes `test_validation.py` for basic validation:

```bash
python test_validation.py
```

Tests:
- ✓ Embedding cache functionality
- ✓ Index building with multiple item types
- ✓ Hybrid search retrieval

**Note**: Full end-to-end tests require API keys and network access.

## Requirements

Added to existing requirements:
- sentence-transformers (embeddings)
- scikit-learn (clustering)
- networkx (graph algorithms)
- python-louvain (community detection)
- google-generativeai (LLM)

All dependencies are in `requirements.txt`.

## Files Modified/Created

**Created:**
- `embedding_cache.py` - Centralized embedding cache
- `phase8_retrieval_enhanced.py` - Enhanced index builder
- `hybrid_search.py` - Hybrid retrieval implementation
- `llm_rerank.py` - LLM reranking module
- `llm_synthesis.py` - Grounded synthesis module
- `integration_example.py` - Integration guide and helper
- `test_validation.py` - Basic validation tests
- `IMPLEMENTATION_GUIDE.md` - This file

**Modified:**
- `main_pipeline.py` - Integrated new modules in Phase 8
- `.gitignore` - Added cache files and graphml exclusions

**Unchanged:**
- `data_corpus.py` - Core data structures
- `cache_utils.py` - DiskJSONCache (reused)
- `ner.py` - Entity extraction
- `Community_processing.py` - Multi-level communities (already complete)
- `graph_save.py` - Persistence layer

## Future Enhancements

Potential improvements (not implemented):
1. **Async LLM calls**: Parallel reranking/synthesis
2. **Vector database**: Replace in-memory index with Milvus/Qdrant
3. **Custom rerankers**: Train specialized reranking models
4. **Query expansion**: Use LLM to expand user queries
5. **Relevance feedback**: User feedback to improve rankings
6. **Multi-modal**: Support image/video in knowledge graph

## License

Same as parent project.
