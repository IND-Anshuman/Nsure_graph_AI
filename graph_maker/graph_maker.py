# graph_maker.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from graph_maker.data_corpus import (
    Entity,
    KnowledgeGraph,
    add_document_and_sentence_nodes,
    build_corpus_from_sources,
    prune_graph,
)
from graph_maker.ner import (
    add_entity_nodes,
    add_mention_and_cooccurrence_edges,
    add_semantic_relation_edges,
    build_entity_catalog,
    extract_semantic_entities_for_doc,
    merge_similar_catalog_entries,
)
from graph_maker.Community_processing import compute_multilevel_communities, build_and_add_community_nodes
from graph_maker.graph_save import export_to_neo4j, save_kg_to_graphml
from utils.kg_validate import validate_graph

load_dotenv()


def build_knowledge_graph(
    *,
    sources: List[str],
    verbose: bool = False,
    skip_relations: bool = False,
    skip_communities: bool = False,
    skip_community_summaries: bool = False,
) -> KnowledgeGraph:
    """Build a KnowledgeGraph from sources.

    Pipeline:
      1) DOCUMENT + SENTENCE nodes
      2) Entity extraction per doc
      3) Entity catalog + ENTITY nodes
      4) Mention + co-occurrence edges
      5) Typed semantic relation edges
      6) Community detection
      7) Community nodes + summaries

    Returns:
      KnowledgeGraph
    """

    prev_disable_comm_summaries = os.getenv("KG_DISABLE_COMMUNITY_SUMMARIES")
    if skip_community_summaries:
        os.environ["KG_DISABLE_COMMUNITY_SUMMARIES"] = "1"

    graph = KnowledgeGraph()

    t0 = time.perf_counter()

    # -------- Phase 1: ingest corpus + sentence nodes --------
    if not sources:
        raise ValueError("No sources provided")

    corpus = build_corpus_from_sources(sources)
    sent_index = add_document_and_sentence_nodes(
        graph,
        corpus,
        min_sentence_chars=int(os.getenv("KG_MIN_SENTENCE_CHARS", "30") or 30),
        dedup_overlap_threshold=float(os.getenv("KG_SENTENCE_DEDUP_OVERLAP", "0.9") or 0.9),
    )
    if verbose:
        print(f"[Timing] Phase 1 (ingest+sentences): {time.perf_counter() - t0:.2f}s")
    t1 = time.perf_counter()

    # -------- Phase 2: entity extraction --------
    all_entities_per_doc: Dict[str, List[Entity]] = {}
    doc_workers = int(os.getenv("KG_DOC_WORKERS", "0") or 0)
    if doc_workers <= 0:
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
            futs = {
                ex.submit(extract_semantic_entities_for_doc, doc_id, text): doc_id
                for doc_id, text in corpus.items()
            }
            for fut in as_completed(futs):
                doc_id = futs[fut]
                ents = fut.result()
                all_entities_per_doc[doc_id] = ents
                if verbose:
                    print(f"[Phase 2] Completed entities for doc: {doc_id} -> {len(ents)}")

    if verbose:
        print(f"[Timing] Phase 2 (entities): {time.perf_counter() - t1:.2f}s")
    t2 = time.perf_counter()

    # -------- Phase 3: catalog + ENTITY nodes --------
    if verbose:
        print("[Phase 3] Building entity catalog & nodes...")
    catalog = build_entity_catalog(all_entities_per_doc)
    if (os.getenv("KG_DISABLE_ENTITY_MERGE", "0") or "0").strip() != "1":
        merge_thr = float(os.getenv("KG_ENTITY_MERGE_THRESHOLD", "0.99") or 0.99)
        catalog = merge_similar_catalog_entries(catalog, similarity_threshold=merge_thr)
    add_entity_nodes(graph, catalog)
    if verbose:
        print(f"  -> catalog size: {len(catalog)} entities")

    if verbose:
        print(f"[Timing] Phase 3 (catalog+ENTITY nodes): {time.perf_counter() - t2:.2f}s")
    t3 = time.perf_counter()

    # -------- Phase 4: mention + co-occurrence edges --------
    if verbose:
        print("[Phase 4] Adding mention & co-occurrence edges...")
    add_mention_and_cooccurrence_edges(graph, all_entities_per_doc, sent_index)
    if verbose:
        print(f"  -> total edges so far: {len(graph.edges)}")

    if verbose:
        print(f"[Timing] Phase 4 (mentions+cooccurrence): {time.perf_counter() - t3:.2f}s")
    t4 = time.perf_counter()

    # -------- Phase 5: typed semantic relation edges --------
    if skip_relations:
        if verbose:
            print("[Phase 5] Skipping semantic relation extraction (skip_relations)")
    else:
        if verbose:
            print("[Phase 5] Extracting typed semantic relations...")
        created_edges = add_semantic_relation_edges(graph, all_entities_per_doc, sent_index)
        if verbose:
            try:
                print(f"  -> created {len(created_edges)} typed semantic edges")
            except Exception:
                pass
            print(f"  -> total edges now: {len(graph.edges)}")

    if verbose:
        print(f"[Timing] Phase 5 (relations): {time.perf_counter() - t4:.2f}s")
    t5 = time.perf_counter()

    # -------- Phase 6/7: communities --------
    if skip_communities:
        if verbose:
            print("[Phase 6/7] Skipping community detection (skip_communities)")
    else:
        if verbose:
            print("[Phase 6] Detecting entity communities...")

        min_comm_size = int(os.getenv("KG_COMMUNITY_MIN_SIZE", "5") or 5)
        max_levels_env = (os.getenv("KG_COMMUNITY_MAX_LEVELS", "") or "").strip().lower()
        max_levels: Optional[int] = None
        if max_levels_env:
            try:
                max_levels = int(max_levels_env)
            except Exception:
                max_levels = None

        # If you want dynamic control over edge types, pass edge_types=None and configure
        # Community_processing._build_entity_graph defaults, or set explicit edge_types here.
        community_results = compute_multilevel_communities(
            graph,
            max_levels=max_levels,
            min_comm_size=min_comm_size,
            edge_types=None,
            verbose=verbose,
        )

        # -------- Phase 7: community nodes + summaries --------
        if verbose:
            print("[Phase 7] Creating community nodes & summaries...")
        # Community_processing already supports disabling summaries via env.
        build_and_add_community_nodes(
            graph,
            community_results,
            create_member_edges=True,
            create_partof_edges=True,
            include_entity_sample=int(os.getenv("KG_COMMUNITY_ENTITY_SAMPLE", "6") or 6),
            verbose=verbose,
        )

    if verbose:
        print(f"[Timing] Phase 6/7 (communities): {time.perf_counter() - t5:.2f}s")

    # -------- Graph maintenance + validation --------
    prune_graph(graph, min_degree=1, preserve_labels={"DOCUMENT", "SENTENCE", "COMMUNITY"})

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

    if verbose:
        entity_count = sum(1 for n in graph.nodes.values() if n.label == "ENTITY")
        community_count = sum(1 for n in graph.nodes.values() if n.label == "COMMUNITY")
        print(f"[Stats] ENTITY nodes: {entity_count}")
        print(f"[Stats] COMMUNITY nodes: {community_count}")
        print(f"[Stats] Total nodes: {len(graph.nodes)}")
        print(f"[Stats] Total edges: {len(graph.edges)}")

    if skip_community_summaries:
        if prev_disable_comm_summaries is None:
            os.environ.pop("KG_DISABLE_COMMUNITY_SUMMARIES", None)
        else:
            os.environ["KG_DISABLE_COMMUNITY_SUMMARIES"] = prev_disable_comm_summaries

    return graph


def persist_graph(
    *,
    graph: KnowledgeGraph,
    graphml_path: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: Optional[str],
    clear_existing: bool = True,
    verbose: bool = False,
) -> None:
    """Persist KG to GraphML and optionally Neo4j."""

    save_kg_to_graphml(graph, graphml_path)

    if neo4j_password:
        try:
            export_to_neo4j(
                graph,
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
                clear_existing=clear_existing,
            )
        except Exception as e:
            print(f"[WARN] Neo4j export failed: {e}")
            if verbose:
                print("[INFO] Skipping Neo4j export. GraphML file was saved successfully.")
    else:
        if verbose:
            print("[INFO] NEO4J_PASSWORD not set. Skipping Neo4j export.")


def _read_sources_from_file(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a knowledge graph from sources and persist it.")
    parser.add_argument(
        "--source",
        "-s",
        action="append",
        default=[],
        help="Source path/URL (repeatable). Example: -s docs/file.pdf -s https://example.com/page",
    )
    parser.add_argument(
        "--sources-file",
        "-f",
        type=Path,
        help="Text file with one source per line (paths/URLs). Lines starting with # are ignored.",
    )
    parser.add_argument(
        "--graphml",
        default="knowledge_graph.graphml",
        help="Output GraphML path (default: knowledge_graph.graphml)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print pipeline logs")

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Safe fast mode: skips community summaries and Neo4j export (does not change the core graph structure)",
    )
    parser.add_argument("--skip-relations", action="store_true", help="Skip semantic relation extraction")
    parser.add_argument("--skip-communities", action="store_true", help="Skip community detection + nodes")
    parser.add_argument(
        "--skip-community-summaries",
        action="store_true",
        help="Skip LLM community summaries (faster). Still creates communities if supported.",
    )
    parser.add_argument("--skip-neo4j", action="store_true", help="Skip Neo4j export even if password is set")
    parser.add_argument(
        "--relation-max-sentences",
        type=int,
        default=0,
        help="Cap number of sentences for relation extraction (0 = no cap)",
    )
    parser.add_argument(
        "--disable-llm-relations",
        action="store_true",
        help="Disable LLM relation extraction (keeps any rule-based relations)",
    )

    parser.add_argument(
        "--doc-workers",
        type=int,
        default=0,
        help="Entity extraction workers (0 = default). Sets KG_DOC_WORKERS.",
    )
    parser.add_argument(
        "--relation-workers",
        type=int,
        default=0,
        help="Relation extraction workers (0 = default). Sets KG_RELATION_WORKERS.",
    )
    parser.add_argument(
        "--spacy-n-process",
        type=int,
        default=0,
        help="spaCy n_process for sentence splitting across multiple documents (0/1 = no multiprocessing). Sets KG_SPACY_N_PROCESS.",
    )
    parser.add_argument(
        "--spacy-batch-size",
        type=int,
        default=0,
        help="spaCy batch_size for nlp.pipe (0 = default). Sets KG_SPACY_BATCH_SIZE.",
    )

    # Neo4j options (optional)
    parser.add_argument(
        "--neo4j-uri",
        default=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
        help="Neo4j Bolt URI (default from NEO4J_URI or bolt://127.0.0.1:7687)",
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Neo4j username (default from NEO4J_USER or neo4j)",
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.getenv("NEO4J_PASSWORD"),
        help="Neo4j password (default from NEO4J_PASSWORD). If unset, Neo4j export is skipped.",
    )
    parser.add_argument(
        "--neo4j-clear",
        action="store_true",
        help="Clear existing graph in Neo4j before importing",
    )

    args = parser.parse_args(argv)

    # Safe fast-mode knobs: skip expensive summaries and database export.
    if args.fast:
        os.environ["KG_DISABLE_COMMUNITY_SUMMARIES"] = "1"

    if args.skip_community_summaries:
        os.environ["KG_DISABLE_COMMUNITY_SUMMARIES"] = "1"

    # Worker tuning knobs (safe: does not change graph definition, only speed).
    if args.doc_workers and int(args.doc_workers) > 0:
        os.environ["KG_DOC_WORKERS"] = str(int(args.doc_workers))
    if args.relation_workers and int(args.relation_workers) > 0:
        os.environ["KG_RELATION_WORKERS"] = str(int(args.relation_workers))
    if args.spacy_n_process and int(args.spacy_n_process) > 0:
        os.environ["KG_SPACY_N_PROCESS"] = str(int(args.spacy_n_process))
    if args.spacy_batch_size and int(args.spacy_batch_size) > 0:
        os.environ["KG_SPACY_BATCH_SIZE"] = str(int(args.spacy_batch_size))

    # Apply speed knobs via env so deeper modules can read them.
    if args.relation_max_sentences and int(args.relation_max_sentences) > 0:
        os.environ["KG_RELATION_MAX_SENTENCES"] = str(int(args.relation_max_sentences))
    if args.disable_llm_relations:
        os.environ["KG_DISABLE_LLM_RELATIONS"] = "1"

    sources: List[str] = []
    if args.sources_file:
        if args.sources_file.exists():
            sources.extend(_read_sources_from_file(args.sources_file))
        else:
            raise FileNotFoundError(f"sources file not found: {args.sources_file}")
    sources.extend(args.source or [])

    graph = build_knowledge_graph(
        sources=sources,
        verbose=args.verbose,
        skip_relations=bool(args.skip_relations),
        skip_communities=bool(args.skip_communities),
        skip_community_summaries=bool(args.skip_community_summaries or args.fast),
    )

    persist_graph(
        graph=graph,
        graphml_path=str(args.graphml),
        neo4j_uri=str(args.neo4j_uri),
        neo4j_user=str(args.neo4j_user),
        neo4j_password=None if args.skip_neo4j or args.fast else (str(args.neo4j_password) if args.neo4j_password else None),
        clear_existing=bool(args.neo4j_clear),
        verbose=args.verbose,
    )

    if args.verbose:
        print("[Done] graph_maker finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
