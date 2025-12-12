#!/usr/bin/env python
"""
Quick validation test for the enhanced retrieval modules.
Tests basic functionality without requiring API keys.
"""
import sys
import numpy as np
from data_corpus import KnowledgeGraph, KGNode, KGEdge, Entity
from phase8_retrieval_enhanced import build_retrieval_index_enhanced, IndexItem
from hybrid_search import search_and_expand, RetrievalCandidate
from embedding_cache import get_embeddings_with_cache


def create_test_graph():
    """Create a small test knowledge graph."""
    graph = KnowledgeGraph()
    
    # Add some ENTITY nodes
    graph.add_node(KGNode(
        id="ent:graphrag",
        label="ENTITY",
        properties={
            "canonical": "graphrag",
            "type": "FRAMEWORK",
            "aliases": ["GraphRAG", "Graph RAG"],
            "descriptions": ["A framework for retrieval-augmented generation using knowledge graphs"],
            "sources": ["doc1"]
        }
    ))
    
    graph.add_node(KGNode(
        id="ent:neo4j",
        label="ENTITY",
        properties={
            "canonical": "neo4j",
            "type": "DATABASE",
            "aliases": ["Neo4j", "Neo4j Graph Database"],
            "descriptions": ["A graph database management system"],
            "sources": ["doc1"]
        }
    ))
    
    # Add some SENTENCE nodes
    graph.add_node(KGNode(
        id="sent:doc1:0",
        label="SENTENCE",
        properties={
            "doc_id": "doc1",
            "index": 0,
            "text": "GraphRAG is a powerful framework that combines knowledge graphs with retrieval-augmented generation."
        }
    ))
    
    graph.add_node(KGNode(
        id="sent:doc1:1",
        label="SENTENCE",
        properties={
            "doc_id": "doc1",
            "index": 1,
            "text": "Neo4j is a popular graph database that can be used to store and query knowledge graphs."
        }
    ))
    
    # Add a COMMUNITY node
    graph.add_node(KGNode(
        id="comm_l0:0",
        label="COMMUNITY",
        properties={
            "level": 0,
            "comm_id": 0,
            "members_count": 2,
            "title": "Graph Technologies",
            "summary": "This community focuses on graph database technologies and frameworks for knowledge graph applications.",
            "sample_entities": ["ent:graphrag", "ent:neo4j"]
        }
    ))
    
    # Add edges
    graph.add_edge(KGEdge(
        id="e:0",
        source="ent:graphrag",
        target="sent:doc1:0",
        type="MENTION_IN",
        properties={"surface": "GraphRAG", "doc_id": "doc1"}
    ))
    
    graph.add_edge(KGEdge(
        id="e:1",
        source="ent:neo4j",
        target="sent:doc1:1",
        type="MENTION_IN",
        properties={"surface": "Neo4j", "doc_id": "doc1"}
    ))
    
    graph.add_edge(KGEdge(
        id="e:2",
        source="ent:graphrag",
        target="ent:neo4j",
        type="CO_OCCURS_WITH",
        properties={"sentence_id": "sent:doc1:0"}
    ))
    
    graph.add_edge(KGEdge(
        id="e:3",
        source="ent:graphrag",
        target="comm_l0:0",
        type="MEMBER_OF",
        properties={"level": 0, "comm_id": 0}
    ))
    
    graph.add_edge(KGEdge(
        id="e:4",
        source="ent:neo4j",
        target="comm_l0:0",
        type="MEMBER_OF",
        properties={"level": 0, "comm_id": 0}
    ))
    
    return graph


def test_embedding_cache():
    """Test embedding cache functionality."""
    print("\n" + "="*60)
    print("TEST 1: Embedding Cache")
    print("="*60)
    
    texts = ["test sentence 1", "test sentence 2", "test sentence 3"]
    
    # First call - should compute and cache
    embs1 = get_embeddings_with_cache(texts)
    print(f"✓ First call: got embeddings shape {embs1.shape}")
    
    # Second call - should use cache
    embs2 = get_embeddings_with_cache(texts)
    print(f"✓ Second call: got embeddings shape {embs2.shape}")
    
    # Check they're identical
    if np.allclose(embs1, embs2):
        print("✓ Cache working correctly (embeddings match)")
    else:
        print("✗ Cache issue (embeddings don't match)")
        return False
    
    return True


def test_index_building():
    """Test retrieval index building."""
    print("\n" + "="*60)
    print("TEST 2: Index Building")
    print("="*60)
    
    graph = create_test_graph()
    
    index_items, embeddings = build_retrieval_index_enhanced(
        graph,
        context_window_tokens=20,
        include_entity_contexts=True,
        verbose=False
    )
    
    print(f"✓ Built index with {len(index_items)} items")
    print(f"✓ Embeddings shape: {embeddings.shape}")
    
    # Check we got different item types
    item_types = set(item.item_type for item in index_items)
    print(f"✓ Item types found: {item_types}")
    
    expected_types = {"SENTENCE", "ENTITY", "COMMUNITY"}
    if expected_types.issubset(item_types):
        print("✓ All expected item types present")
    else:
        missing = expected_types - item_types
        print(f"✗ Missing item types: {missing}")
        return False
    
    return True


def test_hybrid_search():
    """Test hybrid search without LLM."""
    print("\n" + "="*60)
    print("TEST 3: Hybrid Search (without LLM)")
    print("="*60)
    
    graph = create_test_graph()
    index_items, embeddings = build_retrieval_index_enhanced(
        graph,
        context_window_tokens=20,
        include_entity_contexts=False,
        verbose=False
    )
    
    query = "What is GraphRAG?"
    
    candidates = search_and_expand(
        query=query,
        graph=graph,
        index_items=index_items,
        embeddings=embeddings,
        top_n_semantic=5,
        top_k_final=10,
        alpha=0.7,
        beta=0.3,
        expansion_hops=1,
        verbose=False
    )
    
    print(f"✓ Retrieved {len(candidates)} candidates for query: '{query}'")
    
    if len(candidates) > 0:
        print(f"✓ Top candidate: [{candidates[0].id}] (score={candidates[0].hybrid_score:.3f})")
        print(f"  Type: {candidates[0].item_type}")
        print(f"  Text: {candidates[0].text[:80]}...")
        return True
    else:
        print("✗ No candidates retrieved")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("VALIDATION TESTS FOR ENHANCED RETRIEVAL MODULES")
    print("="*60)
    
    results = []
    
    try:
        results.append(("Embedding Cache", test_embedding_cache()))
    except Exception as e:
        print(f"✗ Embedding Cache test failed: {e}")
        results.append(("Embedding Cache", False))
    
    try:
        results.append(("Index Building", test_index_building()))
    except Exception as e:
        print(f"✗ Index Building test failed: {e}")
        results.append(("Index Building", False))
    
    try:
        results.append(("Hybrid Search", test_hybrid_search()))
    except Exception as e:
        print(f"✗ Hybrid Search test failed: {e}")
        results.append(("Hybrid Search", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
