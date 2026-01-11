# main_pipeline.py
# -*- coding: utf-8 -*-
from dotenv import load_dotenv
from typing import Dict, List
from graph_maker.data_corpus import KnowledgeGraph, Entity, prune_graph
import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
# ---- pipeline phases ----
from graph_maker.data_corpus import (
    build_corpus_from_sources,
    add_document_and_sentence_nodes,
)
from graph_maker.ner import extract_semantic_entities_for_doc
from graph_maker.ner import build_entity_catalog, add_entity_nodes
from graph_maker.ner import merge_similar_catalog_entries
from graph_maker.ner import add_mention_and_cooccurrence_edges
from graph_maker.ner import add_semantic_relation_edges
from graph_maker.Community_processing import (
    compute_multilevel_communities,
    build_and_add_community_nodes,
    classify_query,
    multi_hop_traversal,
    handle_abstention,
)

# ---- new enhanced retrieval modules ----
from answer_synthesis.retrieval import build_retrieval_index_enhanced
from answer_synthesis.hybrid_search import search_and_expand
from answer_synthesis.llm_rerank import llm_rerank_candidates
from answer_synthesis.llm_synthesis import llm_synthesize_answer, format_answer_output

# ---- new persistence modules ----
from graph_maker.graph_save import save_kg_to_graphml
from graph_maker.graph_save import export_to_neo4j
from utils.kg_validate import validate_graph
from graph_maker.relation_schema import load_relation_schema, community_edge_types_from_schema

load_dotenv()



def run_pipeline(queries: List[str] | None = None, verbose: bool = False):
    graph = KnowledgeGraph()

    # ========= Phase 1: Ingest text & build DOCUMENT + SENTENCE nodes =========
    # You can put local PDFs and/or URLs here:
    sources = [
        # examples:
        # "docs/my_paper.pdf",
        # "https://arxiv.org/pdf/2305.12345.pdf",
        # "https://some-blog-post.com/llm-graphrag",
        #"https://hcmimphal.nic.in/documents/constitutionofindiaacts.pdf"
        #"https://www.ijrti.org/papers/IJRTI2304061.pdf"
        "./IRDAN144RP0003V02201819_GEN3615.pdf"
    ]
    if not sources:
        # Fallback tiny demo text if no sources given
        if verbose:
            print("[WARN] No sources specified. Using fallback demo corpus.")
        from graph_maker.data_corpus import _clean_whitespace

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

    sent_index = add_document_and_sentence_nodes(graph, corpus, min_sentence_chars=30, dedup_overlap_threshold=0.9)

    # ========= Phase 2: AI-based entity discovery =========
    all_entities_per_doc: Dict[str, List[Entity]] = {}
    doc_workers = int(os.getenv("KG_DOC_WORKERS", "0") or 0)
    if doc_workers <= 0:
        # Good default for mostly-CPU regex + some I/O; keep modest to avoid oversubscription.
        doc_workers = min(8, (os.cpu_count() or 4))

    if len(corpus) <= 1 or doc_workers <= 1:
        for doc_id, text in corpus.items():
            if verbose:
                print(f"[Phase 2] Extracting entities for doc: {doc_id}")
            ents = extract_semantic_entities_for_doc(doc_id, text)
            all_entities_per_doc[doc_id] = ents
            if verbose:
                print(f"  -> found {len(ents)} entities")
    else:
        if verbose:
            print(f"[Phase 2] Extracting entities in parallel (workers={doc_workers})")
        with ThreadPoolExecutor(max_workers=doc_workers) as ex:
            futs = {ex.submit(extract_semantic_entities_for_doc, doc_id, text): doc_id for doc_id, text in corpus.items()}
            for fut in as_completed(futs):
                doc_id = futs[fut]
                if verbose:
                    print(f"[Phase 2] Completed entities for doc: {doc_id}")
                ents = fut.result()
                all_entities_per_doc[doc_id] = ents
                if verbose:
                    print(f"  -> found {len(ents)} entities")

    # ========= Phase 3: catalog + ENTITY nodes =========
    if verbose:
        print("[Phase 3] Building entity catalog & nodes...")
    catalog = build_entity_catalog(all_entities_per_doc)
    if (os.getenv("KG_DISABLE_ENTITY_MERGE", "0") or "0").strip() != "1":
        merge_thr = float(os.getenv("KG_ENTITY_MERGE_THRESHOLD", "0.99") or 0.99)
        catalog = merge_similar_catalog_entries(catalog, similarity_threshold=merge_thr)
    add_entity_nodes(graph, catalog)
    if verbose:
        print(f"  -> catalog size: {len(catalog)} entities")

    # ========= Phase 4: mention + co-occurrence edges =========
    if verbose:
        print("[Phase 4] Adding mention & co-occurrence edges...")
    add_mention_and_cooccurrence_edges(graph, all_entities_per_doc, sent_index)
    if verbose:
        print(f"  -> total edges so far: {len(graph.edges)}")

    # ========= Phase 5: semantic relation edges =========
    if verbose:
        print("[Phase 5] Extracting typed semantic relations (legal-aware LLM)...")
    created_edges = add_semantic_relation_edges(graph, all_entities_per_doc, sent_index)
    if verbose:
        print(f"  -> created {len(created_edges)} typed semantic edges; total edges now: {len(graph.edges)}")

    # ========= Phase 6: community detection =========
    if verbose:
        print("[Phase 6] Detecting entity communities...")
    # Safer defaults to avoid community explosion; override via env if needed.
    min_comm_size = int(os.getenv("KG_COMMUNITY_MIN_SIZE", "5") or 5)
    max_levels_env = (os.getenv("KG_COMMUNITY_MAX_LEVELS", "") or "").strip().lower()
    max_levels = None
    if max_levels_env:
        try:
            max_levels = int(max_levels_env)
        except Exception:
            max_levels = None
    community_results = compute_multilevel_communities(
        graph,
        max_levels=max_levels,
        min_comm_size=min_comm_size,
        edge_types=community_edge_types_from_schema(load_relation_schema()),
        verbose=verbose,
    )

    # ========= Phase 7: community summaries =========
    if verbose:
        print("[Phase 7] Creating community nodes & summaries...")
    comm_map = build_and_add_community_nodes(
        graph,
        community_results,          # or None to skip LLM summarization
        create_member_edges=True,
        create_partof_edges=True,
        include_entity_sample=6,
        verbose=verbose,
    )

    # ========= Phase 8: Enhanced retrieval index build =========
    if verbose:
        print("[Phase 8] Building enhanced retrieval index...")
    index_items, embeddings = build_retrieval_index_enhanced(
        graph,
        context_window_tokens=50,
        include_entity_contexts=True,
        verbose=verbose
    )

    # Run retrieval/synthesis for one or more queries
    if index_items:
        # Default single demo query if none provided
        if not queries:
            queries = [
                "Which clauses should be modeled as terminal “no-coverage” nodes in the knowledge graph?"
                ]

        all_summaries = []
        for i, query in enumerate(queries, start=1):
            if verbose:
                print(f"\n[Query {i} - Hybrid Search] {query}")

            # classify query to route retrieval strategy
            qtype = classify_query(query)
            if qtype == "definition":
                community_boost = 0.05
            elif qtype == "overview":
                community_boost = 0.45
            elif qtype == "comparison":
                community_boost = 0.2
            elif qtype == "howwhy":
                community_boost = 0.25
            elif qtype == "missing":
                community_boost = 0.05
            else:
                community_boost = 0.15

            # Hybrid search
            candidates = search_and_expand(
                query=query,
                graph=graph,
                index_items=index_items,
                embeddings=embeddings,
                top_n_semantic=20,
                top_k_final=40,
                alpha=0.7,
                beta=0.3,
                community_boost=community_boost,
                expansion_hops=1,
                verbose=verbose,
            )

            # Debug: show top pre-rerank candidates with hierarchy metadata
            if verbose:
                debug_top = candidates[:10]
                print("[Debug] Top pre-rerank candidates:")
                for c in debug_top:
                    meta = c.metadata or {}
                    lvl = meta.get("level")
                    coh = meta.get("coherence")
                    print(
                        f"  id={c.id} type={c.item_type} level={lvl} "
                        f"coh={coh} sem={c.semantic_score:.3f} graph={c.graph_score:.3f} "
                        f"hybrid={c.hybrid_score:.3f} path={c.retrieval_path}"
                    )

            # Rerank
            rerank_result = llm_rerank_candidates(
                query=query,
                candidates=candidates,
                top_k=12,
                use_cache=True,
                verbose=verbose,
            )

            # Debug: show top post-rerank items (as returned by reranker)
            ranked = rerank_result.get("ranked_candidates", []) if isinstance(rerank_result, dict) else []
            if verbose:
                print("[Debug] Top post-rerank candidates:")
                for c in ranked[:10]:
                    # c is typically a dict with id/item_type/metadata
                    cid = c.get("id") if isinstance(c, dict) else getattr(c, "id", None)
                    ctype = c.get("item_type") if isinstance(c, dict) else getattr(c, "item_type", None)
                    meta = c.get("metadata") if isinstance(c, dict) else getattr(c, "metadata", {})
                    lvl = (meta or {}).get("level")
                    coh = (meta or {}).get("coherence")
                    print(f"  id={cid} type={ctype} level={lvl} coh={coh}")

            # If no ranked candidates, handle structured abstention
            if not ranked:
                abst_id = handle_abstention(graph, query)
                if verbose:
                    print(f"[Abstention] No evidence found for query. Created abstention node {abst_id}")
                synthesis_result = {"query": query, "answer": None, "abstention_node": abst_id}
            else:
                # Synthesize answer
                synthesis_result = llm_synthesize_answer(
                    query=query,
                    evidence_candidates=ranked,
                    graph=graph,
                    use_cache=True,
                    verbose=verbose,
                )
                # For how/why queries, add multi-hop traversal evidence to explanation
                if qtype == "howwhy":
                    # pick top entity ids from ranked candidates
                    top_entity_ids = [
                        (c.get("id") if isinstance(c, dict) else getattr(c, "id", None))
                        for c in ranked
                        if (
                            (c.get("item_type") if isinstance(c, dict) else getattr(c, "item_type", None))
                            == "ENTITY"
                        )
                    ][:6]
                    traversal = []
                    if top_entity_ids:
                        traversal = multi_hop_traversal(graph, top_entity_ids, hops=3)
                    # attach traversal to synthesis result when possible
                    try:
                        if isinstance(synthesis_result, dict):
                            synthesis_result["traversal_paths"] = traversal
                    except Exception:
                        pass

            # Display answer
            output = format_answer_output(synthesis_result)
            print("\n" + output)
            all_summaries.append({"query": query, "answer": synthesis_result})

    else:
        if verbose:
            print("[Phase 8] No nodes with text/summary to index.")

    # ========= Graph maintenance =========
    prune_graph(graph, min_degree=1, preserve_labels={"DOCUMENT", "SENTENCE", "COMMUNITY"})

    # ========= Integrity checks =========
    try:
        dangling, missing_refs, bad_member, bad_partof = validate_graph(graph)
        if verbose:
            print(
                f"[Validate] dangling_edges={dangling} missing_refs={missing_refs} "
                f"bad_MEMBER_OF={bad_member} bad_PART_OF={bad_partof}"
            )
    except Exception as e:
        if verbose:
            print(f"[Validate] skipped due to error: {e}")

    # ========= Simple graph statistics =========
    entity_count = sum(1 for n in graph.nodes.values() if n.label == "ENTITY")
    community_count = sum(1 for n in graph.nodes.values() if n.label == "COMMUNITY")
    cooccur_edges = sum(1 for e in graph.edges if e.type == "CO_OCCURS_WITH")
    non_typed = {
        "MENTION_IN",
        "CO_OCCURS_WITH",
        "HAS_SENTENCE",
        "MEMBER_OF",
        "PART_OF",
    }
    typed_rel_edges = sum(1 for e in graph.edges if e.type not in non_typed)

    if verbose:
        print("\n[Stats] ENTITY nodes:", entity_count)
        print("[Stats] COMMUNITY nodes:", community_count)
        print("[Stats] CO_OCCURS_WITH edges:", cooccur_edges)
        print("[Stats] Typed relation edges:", typed_rel_edges)

    # ========= Persistence: GraphML + Neo4j =========
    if verbose:
        print("\n[Persistence] Saving KG to GraphML and Neo4j...")

    # 1) GraphML
    graphml_path = "knowledge_graph.graphml"
    save_kg_to_graphml(graph, graphml_path)

    # 2) Neo4j (optional - only if you have Neo4j running)
    # Update credentials in .env or skip if Neo4j is not available

    # For local single-instance Neo4j, bolt:// is typically correct.
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
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
            if verbose:
                print("[INFO] Skipping Neo4j export. GraphML file was saved successfully.")
    else:
        if verbose:
            print("[INFO] NEO4J_PASSWORD not set. Skipping Neo4j export.")

    if verbose:
        print("[Done] Pipeline finished.")
    return graph, index_items, embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run knowledge-graph pipeline and answer queries.")
    parser.add_argument("--query", "-q", action="append", help="A query to run (can specify multiple)")
    parser.add_argument("--queries-file", "-f", type=Path, help="Path to a text file with one query per line")
    parser.add_argument("--verbose", action="store_true", help="Print pipeline logs (default: only answers)")
    args = parser.parse_args()

    queries: List[str] | None = None
    if args.queries_file:
        if args.queries_file.exists():
            queries = [line.strip() for line in args.queries_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            print(f"[WARN] queries file not found: {args.queries_file}")
    if args.query:
        queries = (queries or []) + args.query

    run_pipeline(queries=queries, verbose=args.verbose)
