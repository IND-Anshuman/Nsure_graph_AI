r"""engine.py

Answer queries from an existing GraphML knowledge graph export.

- Loads `knowledge_graph.graphml` (produced by graph_save.save_kg_to_graphml)
- Rebuilds the enhanced retrieval index
- Runs hybrid retrieval + LLM reranking + grounded synthesis

Usage (PowerShell):
  python .\engine.py --graphml .\knowledge_graph.graphml -q "What is this document about?" --verbose
  python .\engine.py -q "How is AI applied in finance?" -q "Compare healthcare vs finance" --verbose
  python .\engine.py --queries-file .\queries.txt

Notes:
- Requires LLM API keys for reranking/synthesis: set GOOGLE_API_KEY and/or OPENAI_API_KEY.
- Does not modify the knowledge graph; it only loads and queries it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import argparse
import json
import os
import re
import time

import networkx as nx
from dotenv import load_dotenv

from data_corpus import KnowledgeGraph, KGNode, KGEdge
from phase8_retrieval_enhanced import build_retrieval_index_enhanced
from hybrid_search import search_and_expand
from llm_rerank import llm_rerank_candidates
from llm_synthesis import llm_synthesize_answer, format_answer_output
from Community_processing import classify_query, handle_abstention, multi_hop_traversal


_TAG_PATTERNS: Dict[str, re.Pattern] = {
    # Core interaction concepts
    "exclusion": re.compile(r"\b(exclusion|excluded|not\s+cover(?:ed|age)|shall\s+not\s+be\s+liable)\b", re.IGNORECASE),
    "exception": re.compile(r"\b(exception|exceptions|general\s+exceptions)\b", re.IGNORECASE),
    "third_party": re.compile(r"\b(third\s*[- ]\s*party)\b", re.IGNORECASE),
    "override": re.compile(
        r"\b(notwithstanding|override|overrides|statutor(?:y|ily)|compulsor(?:y|ily)|motor\s+vehicles?\s+act|mv\s*act|shall\s+pay)\b",
        re.IGNORECASE,
    ),
    "recovery": re.compile(
        r"\b(right\s+of\s+recovery|recover(?:y|able)?|reimbursement|repay(?:ment)?|entitled\s+to\s+recover|recover\s+from\s+the\s+insured)\b",
        re.IGNORECASE,
    ),
    # Common noise for liability/override questions
    "war_nuclear": re.compile(r"\b(war|nuclear|radioactive|ionis(?:ing|ing)|contamination)\b", re.IGNORECASE),
}


def _cand_text(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("text") or "")
    return str(getattr(c, "text", "") or "")


def _cand_type(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("item_type") or c.get("type") or "")
    return str(getattr(c, "item_type", "") or "")


def _cand_id(c: Any) -> Optional[str]:
    if isinstance(c, dict):
        v = c.get("id")
        return str(v) if v is not None else None
    v = getattr(c, "id", None)
    return str(v) if v is not None else None


def _tags_for_text(text: str) -> set[str]:
    t = text or ""
    out: set[str] = set()
    for name, rx in _TAG_PATTERNS.items():
        if rx.search(t):
            out.add(name)
    return out


def _is_interaction_query(query: str) -> bool:
    q = (query or "").lower()
    triggers = [
        "interact",
        "interaction",
        "override",
        "overrides",
        "statutory",
        "liability",
        "third party",
        "third-party",
        "exclusion",
        "exceptions",
        "general exceptions",
        "recovery",
        "recover",
        "reimburse",
        "repay",
    ]
    return any(t in q for t in triggers)


def _required_tags_for_query(query: str) -> set[str]:
    if not _is_interaction_query(query):
        return set()
    # Minimal interaction mechanics coverage: baseline exclusion/exception, statutory override, and consequence/recovery.
    return {"override", "recovery", "exclusion"}


def _coverage_from_ranked(ranked: List[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for c in ranked or []:
        tags = _tags_for_text(_cand_text(c))
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
    return counts


def _augment_query_for_missing(query: str, missing: set[str]) -> str:
    extra: List[str] = []
    if "recovery" in missing:
        extra.append("right of recovery reimbursement repay recover from insured")
    if "override" in missing:
        extra.append("statutory obligation notwithstanding motor vehicles act insurer shall pay third party")
    if "exclusion" in missing:
        extra.append("exclusion general exceptions not covered")
    if not extra:
        return query
    return f"{query}\n\nFocus: {'; '.join(extra)}"


def _should_filter_war_noise(query: str) -> bool:
    q = (query or "").lower()
    # Only apply to the statutory/third-party/exclusion interaction family.
    return any(k in q for k in ["third party", "third-party", "motor vehicles", "statutory", "liability", "exclusion", "exceptions"]) \
        and not any(k in q for k in ["war", "nuclear", "radioactive"]) 


def _filter_noise_candidates(candidates: List[Any], query: str) -> List[Any]:
    if not candidates:
        return []
    if not _should_filter_war_noise(query):
        return candidates
    out: List[Any] = []
    for c in candidates:
        txt = _cand_text(c)
        tags = _tags_for_text(txt)
        # Drop war/nuclear clauses unless they also contain core interaction tags.
        if "war_nuclear" in tags and not (tags & {"override", "recovery", "third_party", "exclusion"}):
            continue
        out.append(c)
    return out


def _merge_candidates(a: List[Any], b: List[Any]) -> List[Any]:
    """Merge candidate lists by id, keeping the best (max) scores per item."""
    by_id: Dict[str, Any] = {}
    for c in (a or []) + (b or []):
        cid = _cand_id(c)
        if not cid:
            continue
        prev = by_id.get(cid)
        if prev is None:
            by_id[cid] = c
            continue
        # Prefer the candidate with higher hybrid_score when possible.
        try:
            prev_h = float(getattr(prev, "hybrid_score", 0.0) if not isinstance(prev, dict) else prev.get("hybrid_score", 0.0) or 0.0)
            cur_h = float(getattr(c, "hybrid_score", 0.0) if not isinstance(c, dict) else c.get("hybrid_score", 0.0) or 0.0)
        except Exception:
            prev_h, cur_h = 0.0, 0.0
        if cur_h > prev_h:
            by_id[cid] = c
    return list(by_id.values())


def _calibrate_confidence_and_note(
    synthesis_result: Dict[str, Any],
    *,
    query: str,
    coverage_counts: Dict[str, int],
) -> Dict[str, Any]:
    required = _required_tags_for_query(query)
    if not required:
        return synthesis_result

    missing = {t for t in required if coverage_counts.get(t, 0) <= 0}
    if not missing:
        return synthesis_result

    # Confidence should be conservative when a required step is missing.
    synthesis_result["confidence"] = "medium" if synthesis_result.get("confidence") != "low" else "low"
    note = str(synthesis_result.get("insufficiency_note") or "").strip()
    add = (
        "Interaction mechanics are partially supported: missing direct evidence for "
        f"{', '.join(sorted(missing))}. "
        "For statutory override questions, include an explicit recovery/reimbursement clause when present."
    )
    synthesis_result["insufficiency_note"] = (note + " " + add).strip() if note else add
    return synthesis_result


def _hybrid_rank_fallback(cands: List[Any], *, top_k: int) -> List[Any]:
    def _score(c: Any) -> float:
        try:
            if isinstance(c, dict):
                return float(c.get("hybrid_score") or 0.0)
            return float(getattr(c, "hybrid_score", 0.0) or 0.0)
        except Exception:
            return 0.0

    return sorted(cands or [], key=_score, reverse=True)[: max(1, int(top_k))]


def _build_mentions_index(graph: KnowledgeGraph) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Return (sent_id->entity_ids, entity_id->sent_ids) from MENTION_IN edges."""
    sent_to_ents: Dict[str, Set[str]] = {}
    ent_to_sents: Dict[str, Set[str]] = {}
    for e in graph.edges:
        if e.type != "MENTION_IN":
            continue
        ent = str(e.source)
        sent = str(e.target)
        if sent:
            sent_to_ents.setdefault(sent, set()).add(ent)
        if ent:
            ent_to_sents.setdefault(ent, set()).add(sent)
    return sent_to_ents, ent_to_sents


def _candidate_entity_ids(c: Any) -> Set[str]:
    """Best-effort: map a candidate to underlying ENTITY ids in the KG."""
    out: Set[str] = set()
    if isinstance(c, dict):
        cid = c.get("id")
        ctype = str(c.get("item_type") or c.get("type") or "")
        meta = c.get("metadata") or {}
    else:
        cid = getattr(c, "id", None)
        ctype = str(getattr(c, "item_type", "") or "")
        meta = getattr(c, "metadata", None) or {}

    cid_s = str(cid) if cid is not None else ""
    t = (ctype or "").upper()

    if t == "ENTITY" and cid_s:
        out.add(cid_s)
    elif t in {"ENTITY_CONTEXT", "PROVISION_CONTEXT"}:
        eid = meta.get("entity_id")
        if eid:
            out.add(str(eid))
    elif t == "RELATION":
        src = meta.get("source")
        tgt = meta.get("target")
        if src:
            out.add(str(src))
        if tgt:
            out.add(str(tgt))

    return out


def _interaction_chain_prune(
    *,
    graph: KnowledgeGraph,
    ranked: List[Any],
    query: str,
    sent_to_ents: Dict[str, Set[str]],
    hops: int = 3,
    verbose: bool = False,
) -> List[Any]:
    """Keep only evidence that participates in trigger→override→consequence chain.

    Heuristic (query-time, deterministic):
    - Build anchor entity ids from ranked evidence that matches core tags.
    - Traverse ENTITY/COMMUNITY graph from those anchors.
    - Keep evidence that either:
        - is directly tagged as a required bucket item, OR
        - references an entity in the traversed subgraph.
    """
    if not ranked or not _is_interaction_query(query):
        return ranked

    required = _required_tags_for_query(query)
    if not required:
        return ranked

    # Collect anchor entities from high-signal evidence.
    anchors: Set[str] = set()
    for c in ranked:
        tags = _tags_for_text(_cand_text(c))
        if not (tags & {"override", "recovery", "exclusion", "third_party", "exception"}):
            continue
        anchors |= _candidate_entity_ids(c)

    # Also include entities mentioned by tagged sentences.
    for c in ranked:
        if _cand_type(c).upper() != "SENTENCE":
            continue
        sid = _cand_id(c)
        if not sid:
            continue
        tags = _tags_for_text(_cand_text(c))
        if tags & {"override", "recovery", "exclusion", "third_party", "exception"}:
            anchors |= set(sent_to_ents.get(sid, set()))

    anchors = {a for a in anchors if a in graph.nodes and (graph.nodes.get(a).label == "ENTITY")}
    if not anchors:
        return ranked

    # Traverse entity/community graph to define the interaction neighborhood.
    paths = multi_hop_traversal(graph, sorted(anchors), hops=int(hops), allowed_labels={"ENTITY", "COMMUNITY"})
    neighborhood: Set[str] = set(anchors)
    for p in paths:
        for n in p:
            neighborhood.add(str(n))

    # Prune: keep only items connected to neighborhood, or required-tag items.
    pruned: List[Any] = []
    for c in ranked:
        ctype = _cand_type(c).upper()
        txt = _cand_text(c)
        tags = _tags_for_text(txt)

        # Always drop war/nuclear noise unless explicitly asked.
        if "war_nuclear" in tags and _should_filter_war_noise(query):
            continue

        # Always keep if it satisfies a required bucket.
        if required & tags:
            pruned.append(c)
            continue

        # Prefer sentence/context/relation evidence; aggressively drop chunks/communities unless connected.
        if ctype.startswith("COMMUNITY") or ctype == "CHUNK":
            # Keep only if it directly includes core tags AND connects to neighborhood via entity ids.
            ent_ids = _candidate_entity_ids(c)
            if ent_ids and (ent_ids & neighborhood):
                pruned.append(c)
            continue

        if ctype == "SENTENCE":
            sid = _cand_id(c)
            if sid and (sent_to_ents.get(sid, set()) & neighborhood):
                pruned.append(c)
            continue

        ent_ids = _candidate_entity_ids(c)
        if ent_ids and (ent_ids & neighborhood):
            pruned.append(c)
            continue

    # Safety: if pruning removes required coverage, fall back to original.
    cov = _coverage_from_ranked(pruned)
    missing = {t for t in required if cov.get(t, 0) <= 0}
    if missing:
        if verbose:
            print(f"[ChainPrune] would drop required={sorted(missing)}; keeping unpruned ranked list")
        return ranked

    if verbose:
        print(f"[ChainPrune] ranked={len(ranked)} -> kept={len(pruned)}")
    return pruned


def _maybe_json_load(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return ""
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            return value
    return value


def import_graphml_to_kg(path: str) -> KnowledgeGraph:
    """Import GraphML written by graph_save.save_kg_to_graphml into data_corpus.KnowledgeGraph."""
    G = nx.read_graphml(path)

    kg = KnowledgeGraph()

    # Nodes
    for node_id in G.nodes():
        node_data = dict(G.nodes[node_id])
        label = node_data.pop("label", "Node")
        props: Dict[str, Any] = {k: _maybe_json_load(v) for k, v in node_data.items()}
        kg.add_node(KGNode(id=str(node_id), label=str(label), properties=props))

    # Edges
    edge_counter = 0
    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        for u, v, k, data in G.edges(keys=True, data=True):
            edge_counter += 1
            edge_type = data.get("type") or data.get("edge_type") or "RELATED_TO"
            edge_id = data.get("id") or str(k) or f"{u}__{v}__{edge_counter}"
            props: Dict[str, Any] = {}
            for kk, vv in dict(data).items():
                if kk in {"type", "edge_type", "id"}:
                    continue
                props[kk] = _maybe_json_load(vv)
            kg.add_edge(KGEdge(id=str(edge_id), source=str(u), target=str(v), type=str(edge_type), properties=props))
    else:
        for u, v, data in G.edges(data=True):
            edge_counter += 1
            edge_type = data.get("type") or data.get("edge_type") or "RELATED_TO"
            edge_id = data.get("id") or f"{u}__{v}__{edge_counter}"
            props: Dict[str, Any] = {}
            for kk, vv in dict(data).items():
                if kk in {"type", "edge_type", "id"}:
                    continue
                props[kk] = _maybe_json_load(vv)
            kg.add_edge(KGEdge(id=str(edge_id), source=str(u), target=str(v), type=str(edge_type), properties=props))

    return kg


def _community_boost_for_query(query: str) -> float:
    qtype = classify_query(query)
    if qtype == "definition":
        return 0.05
    if qtype == "overview":
        return 0.45
    if qtype == "comparison":
        return 0.2
    if qtype == "howwhy":
        return 0.25
    if qtype == "missing":
        return 0.05
    return 0.15


def answer_queries_from_graphml(
    *,
    graphml_path: str,
    queries: List[str],
    verbose: bool,
    top_n_semantic: int,
    top_k_final: int,
    rerank_top_k: int,
) -> None:
    t0 = time.perf_counter()
    graph = import_graphml_to_kg(graphml_path)
    if verbose:
        print(f"[Load] GraphML loaded: nodes={len(graph.nodes)} edges={len(graph.edges)} ({time.perf_counter()-t0:.2f}s)")

    sent_to_ents, _ent_to_sents = _build_mentions_index(graph)

    if verbose:
        print("[Index] Building enhanced retrieval index...")
    index_items, embeddings = build_retrieval_index_enhanced(
        graph,
        context_window_tokens=50,
        include_entity_contexts=True,
        verbose=verbose,
    )

    if not index_items:
        print("[Index] No items to index; cannot answer queries.")
        return

    for i, query in enumerate(queries, start=1):
        if verbose:
            print(f"\n[Query {i}] {query}")

        qtype = classify_query(query)
        community_boost = _community_boost_for_query(query)

        candidates = search_and_expand(
            query=query,
            graph=graph,
            index_items=index_items,
            embeddings=embeddings,
            top_n_semantic=top_n_semantic,
            top_k_final=top_k_final,
            alpha=0.7,
            beta=0.3,
            community_boost=community_boost,
            expansion_hops=1,
            verbose=verbose,
        )

        # Remove obvious noise for statutory-liability interaction questions (e.g., war/nuclear clauses).
        candidates = _filter_noise_candidates(candidates, query)

        ranked: List[Any] = []
        try:
            rerank_result = llm_rerank_candidates(
                query=query,
                candidates=candidates,
                top_k=rerank_top_k,
                use_cache=True,
                verbose=verbose,
            )
            ranked = rerank_result.get("ranked_candidates", []) if isinstance(rerank_result, dict) else []
        except Exception as e:
            if verbose:
                print(f"[Rerank] unavailable ({type(e).__name__}: {e}); using hybrid-score fallback")
            ranked = _hybrid_rank_fallback(candidates, top_k=rerank_top_k)

        # Enforce interaction coverage: if we have override evidence but no recovery/consequence evidence,
        # run a bounded second-pass retrieval with an augmented query and merge results.
        required = _required_tags_for_query(query)
        cov = _coverage_from_ranked(ranked)
        missing = {t for t in required if cov.get(t, 0) <= 0}
        if required and missing:
            aug_query = _augment_query_for_missing(query, missing)
            if verbose:
                print(f"[Coverage] missing={sorted(missing)} -> running targeted second-pass retrieval")
            more = search_and_expand(
                query=aug_query,
                graph=graph,
                index_items=index_items,
                embeddings=embeddings,
                top_n_semantic=max(int(top_n_semantic), 60),
                top_k_final=max(int(top_k_final), 80),
                alpha=0.7,
                beta=0.3,
                community_boost=community_boost,
                expansion_hops=1,
                verbose=verbose,
            )
            more = _filter_noise_candidates(more, query)
            merged = _merge_candidates(candidates, more)
            try:
                rerank_result = llm_rerank_candidates(
                    query=query,
                    candidates=merged,
                    top_k=rerank_top_k,
                    use_cache=True,
                    verbose=verbose,
                )
                ranked = rerank_result.get("ranked_candidates", []) if isinstance(rerank_result, dict) else []
            except Exception as e:
                if verbose:
                    print(f"[Rerank] unavailable ({type(e).__name__}: {e}); using hybrid-score fallback")
                ranked = _hybrid_rank_fallback(merged, top_k=rerank_top_k)
            cov = _coverage_from_ranked(ranked)
            if verbose:
                print(f"[Coverage] after second pass: { {k: cov.get(k,0) for k in sorted(required)} }")

        # Strict interaction-chain pruning: for interaction questions, keep only evidence
        # that participates in trigger → override → consequence (recovery) reasoning.
        ranked = _interaction_chain_prune(
            graph=graph,
            ranked=ranked,
            query=query,
            sent_to_ents=sent_to_ents,
            hops=3,
            verbose=verbose,
        )
        cov = _coverage_from_ranked(ranked)

        if not ranked:
            abst_id = handle_abstention(graph, query)
            if verbose:
                print(f"[Abstention] No evidence found; created {abst_id}")
            synthesis_result: Dict[str, Any] = {"query": query, "answer": None, "abstention_node": abst_id}
        else:
            try:
                synthesis_result = llm_synthesize_answer(
                    query=query,
                    evidence_candidates=ranked,
                    graph=graph,
                    use_cache=True,
                    verbose=verbose,
                )
            except Exception as e:
                if verbose:
                    print(f"[Synthesis] unavailable ({type(e).__name__}: {e}); using evidence-only fallback")
                synthesis_result = llm_synthesize_answer(
                    query=query,
                    evidence_candidates=ranked,
                    graph=graph,
                    evidence_only=True,
                    use_cache=False,
                    verbose=verbose,
                )

            # Calibrate confidence based on whether the required interaction steps are actually evidenced.
            if isinstance(synthesis_result, dict):
                synthesis_result = _calibrate_confidence_and_note(synthesis_result, query=query, coverage_counts=cov)

            # For how/why queries, attach multi-hop traversal paths (deterministic) like main_pipeline.py
            if qtype == "howwhy":
                top_entity_ids = [
                    (c.get("id") if isinstance(c, dict) else getattr(c, "id", None))
                    for c in ranked
                    if ((c.get("item_type") if isinstance(c, dict) else getattr(c, "item_type", None)) == "ENTITY")
                ][:6]
                traversal = []
                if top_entity_ids:
                    traversal = multi_hop_traversal(graph, top_entity_ids, hops=3)
                if isinstance(synthesis_result, dict):
                    synthesis_result["traversal_paths"] = traversal

        print("\n" + format_answer_output(synthesis_result))


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.read().splitlines() if ln.strip() and not ln.strip().startswith("#")]


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Answer queries from an existing knowledge_graph.graphml")
    parser.add_argument(
        "--graphml",
        default="knowledge_graph.graphml",
        help="Path to GraphML file (default: knowledge_graph.graphml)",
    )
    parser.add_argument("--query", "-q", action="append", help="Query to run (repeatable)")
    parser.add_argument("--queries-file", "-f", help="Text file with one query per line")
    parser.add_argument("--verbose", action="store_true", help="Verbose debug logs")

    parser.add_argument("--top-n-semantic", type=int, default=20, help="Top-N semantic retrieval candidates")
    parser.add_argument("--top-k-final", type=int, default=40, help="Top-K final candidates after expansion+dedup")
    parser.add_argument("--rerank-top-k", type=int, default=12, help="Top-K evidence passed to synthesis")

    args = parser.parse_args(argv)

    queries: List[str] = []
    if args.queries_file:
        if not os.path.exists(args.queries_file):
            raise FileNotFoundError(f"queries file not found: {args.queries_file}")
        queries.extend(_read_lines(args.queries_file))
    if args.query:
        queries.extend([q for q in args.query if q])

    if not queries:
        queries = [
            "How do the general exceptions interact with the liability to third parties section, and where does statutory obligation override policy exclusions?",
            "Which policy exclusions explicitly do not apply when meeting the requirements of the Motor Vehicles Act?",
            "How does the “Vehicle Laid Up” endorsement modify the insurer’s liability compared to the base policy?",
            "Does the policy anywhere state that exclusions are waived for humanitarian or equitable reasons?",
            "Is the insurer liable if the insured vehicle is driven by a person not permitted under the Driver’s Clause?",
        ]

    answer_queries_from_graphml(
        graphml_path=str(args.graphml),
        queries=queries,
        verbose=bool(args.verbose),
        top_n_semantic=int(args.top_n_semantic),
        top_k_final=int(args.top_k_final),
        rerank_top_k=int(args.rerank_top_k),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
