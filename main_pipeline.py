# main_pipeline.py
from dotenv import load_dotenv
from typing import Dict, List
import google.generativeai as genai
from data_corpus import KnowledgeGraph, Entity
import os
# ---- pipeline phases ----
from data_corpus import (
    build_corpus_from_sources,
    add_document_and_sentence_nodes,
)
from ner import extract_semantic_entities_for_doc
from ner import build_entity_catalog, add_entity_nodes
from ner import add_mention_and_cooccurrence_edges
from ner import add_semantic_relation_edges
from Community_processing import compute_multilevel_communities, build_and_add_community_nodes
from Community_processing import build_retrieval_index, search_relevant_nodes

# ---- new persistence modules ----
from graph_save import save_kg_to_graphml
from graph_save import export_to_neo4j

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def run_pipeline():
    graph = KnowledgeGraph()

    # ========= Phase 1: Ingest text & build DOCUMENT + SENTENCE nodes =========
    # You can put local PDFs and/or URLs here:
    sources = [
        # examples:
        # "docs/my_paper.pdf",
        # "https://arxiv.org/pdf/2305.12345.pdf",
        # "https://some-blog-post.com/llm-graphrag",
        "https://www.oracle.com/ae/a/ocom/docs/applications/the-rise-of-ai-agents-unleashing-productivity-and-innovation-ae.pdf"
    ]
    if not sources:
        # Fallback tiny demo text if no sources given
        print("[WARN] No sources specified. Using fallback demo corpus.")
        from data_corpus import _clean_whitespace

        corpus: Dict[str, str] = {
            "demo_doc": _clean_whitespace(
                "In late 2025, Anshuman Singh proposed AutoGraphRAG, "
                "a framework combining Neo4j and Milvus. "
                "AutoGraphRAG uses Gemini 1.5 Pro and OpenAI models "
                "to build semantic knowledge graphs from NeurIPS papers."
            )
        }
    else:
        corpus = build_corpus_from_sources(sources)

    sent_index = add_document_and_sentence_nodes(graph, corpus)

    # ========= Phase 2: AI-based entity discovery =========
    all_entities_per_doc: Dict[str, List[Entity]] = {}
    for doc_id, text in corpus.items():
        print(f"[Phase 2] Extracting entities for doc: {doc_id}")
        ents = extract_semantic_entities_for_doc(doc_id, text)
        all_entities_per_doc[doc_id] = ents
        print(f"  -> found {len(ents)} entities")

    # ========= Phase 3: catalog + ENTITY nodes =========
    print("[Phase 3] Building entity catalog & nodes...")
    catalog = build_entity_catalog(all_entities_per_doc)
    add_entity_nodes(graph, catalog)
    print(f"  -> catalog size: {len(catalog)} entities")

    # ========= Phase 4: mention + co-occurrence edges =========
    print("[Phase 4] Adding mention & co-occurrence edges...")
    add_mention_and_cooccurrence_edges(graph, all_entities_per_doc, sent_index)
    print(f"  -> total edges so far: {len(graph.edges)}")

    # ========= Phase 5: semantic relation edges =========
    print("[Phase 5] Extracting semantic relations via LLM...")
    add_semantic_relation_edges(graph, all_entities_per_doc, sent_index)
    print(f"  -> total edges after relations: {len(graph.edges)}")

    # ========= Phase 6: community detection =========
    print("[Phase 6] Detecting entity communities...")
    community_results = compute_multilevel_communities(
    graph,
    max_levels=2,
    min_comm_size=1,
    edge_types=["CO_OCCURS_WITH", "USED_WITH", "DEPLOYS_ON"],
    verbose=True
)

    # ========= Phase 7: community summaries =========
    print("[Phase 7] Creating community nodes & summaries...")
    comm_map = build_and_add_community_nodes(
    graph,
    community_results,          # or None to skip LLM summarization
    create_member_edges=True,
    create_partof_edges=True,
    include_entity_sample=6,
    verbose=True
)

    # ========= Phase 8: retrieval index build (optional) =========
    print("[Phase 8] Building retrieval index (sentences + communities)...")
    ids, embs = build_retrieval_index(graph)

    # Small test query
    if ids:
        query = "How does AutoGraphRAG build knowledge graphs and which tools does it use?"
        print(f"\n[Query test] {query}")
        results = search_relevant_nodes(query, ids, embs, top_k=5)
        for nid, score in results:
            node = graph.nodes[nid]
            text = node.properties.get("text") or node.properties.get("summary")
            print(f"- {nid} ({node.label}) score={score:.3f}")
            print(f"  text: {text[:200]!r}")
    else:
        print("[Phase 8] No nodes with text/summary to index.")

    # ========= Persistence: GraphML + Neo4j =========
    print("\n[Persistence] Saving KG to GraphML and Neo4j...")

    # 1) GraphML
    graphml_path = "knowledge_graph.graphml"
    save_kg_to_graphml(graph, graphml_path)

    # 2) Neo4j (optional - only if you have Neo4j running)
    # Update credentials in .env or skip if Neo4j is not available
    import os
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if neo4j_password:
        try:
            export_to_neo4j(
                graph,
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
                clear_existing=True,
            )
        except Exception as e:
            print(f"[WARN] Neo4j export failed: {e}")
            print("[INFO] Skipping Neo4j export. GraphML file was saved successfully.")
    else:
        print("[INFO] NEO4J_PASSWORD not set. Skipping Neo4j export.")

    print("[Done] Pipeline finished.")
    return graph, ids, embs


if __name__ == "__main__":
    run_pipeline()
