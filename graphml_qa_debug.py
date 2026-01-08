"""graphml_qa_debug.py

Run GraphRAG Q&A directly from an existing GraphML knowledge graph export.

- Loads `knowledge_graph.graphml` (produced by graph_save.save_kg_to_graphml)
- Rebuilds the enhanced retrieval index (sentences, chunks, entities, entity-contexts, communities)
- Runs hybrid retrieval + reranking + synthesis for a set of queries
- Prints detailed debug output and per-stage timing

Usage (PowerShell):
  python .\graphml_qa_debug.py
  python .\graphml_qa_debug.py --graphml .\knowledge_graph.graphml --verbose
  python .\graphml_qa_debug.py -q "How are application domains grouped into related communities?" --verbose

Notes:
- Requires the same API keys as main_pipeline for reranking/synthesis (GOOGLE_API_KEY and/or OPENAI_API_KEY).
- Does not modify NER/community/retrieval logic; it only loads and queries an existing graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable, Tuple
import argparse
from contextlib import contextmanager
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
import pickle
import time

import networkx as nx
from dotenv import load_dotenv
from openai import OpenAI

from data_corpus import KnowledgeGraph, KGNode, KGEdge
from phase8_retrieval_enhanced import build_retrieval_index_enhanced
from hybrid_search import search_and_expand
from llm_rerank import llm_rerank_candidates
from llm_synthesis import llm_synthesize_answer, format_answer_output
from cache_utils import DiskJSONCache
from Community_processing import classify_query, handle_abstention, multi_hop_traversal


@dataclass
class Timer:
    name: str
    start: float


class Timing:
    def __init__(self) -> None:
        self.events: List[Tuple[str, float]] = []

    def __enter__(self) -> "Timing":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    @contextmanager
    def time_block(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.events.append((name, dt))

    def print_summary(self) -> None:
        if not self.events:
            return
        print("\n[TIMING] Summary")
        for name, dt in self.events:
            print(f"  - {name}: {dt:.3f}s")


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
        props: Dict[str, Any] = {}
        for k, v in node_data.items():
            props[k] = _maybe_json_load(v)
        kg.add_node(KGNode(id=str(node_id), label=str(label), properties=props))

    # Edges
    # networkx can yield 3-tuples (u, v, data) or 4-tuples (u, v, key, data) depending on graph type
    edge_counter = 0
    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        iterator = G.edges(keys=True, data=True)
        for u, v, k, data in iterator:
            edge_counter += 1
            edge_type = data.get("type") or data.get("edge_type") or "RELATED_TO"
            edge_id = data.get("id") or str(k) or f"{u}__{v}__{edge_counter}"
            props: Dict[str, Any] = {}
            for kk, vv in data.items():
                if kk in {"type", "edge_type", "id"}:
                    continue
                props[kk] = _maybe_json_load(vv)
            kg.add_edge(KGEdge(id=str(edge_id), source=str(u), target=str(v), type=str(edge_type), properties=props))
    else:
        iterator2 = G.edges(data=True)
        for u, v, data in iterator2:
            edge_counter += 1
            edge_type = data.get("type") or data.get("edge_type") or "RELATED_TO"
            edge_id = data.get("id") or f"{u}__{v}__{edge_counter}"
            props: Dict[str, Any] = {}
            for kk, vv in data.items():
                if kk in {"type", "edge_type", "id"}:
                    continue
                props[kk] = _maybe_json_load(vv)
            kg.add_edge(KGEdge(id=str(edge_id), source=str(u), target=str(v), type=str(edge_type), properties=props))

    return kg


def _graphml_signature(path: str) -> Dict[str, Any]:
    st = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": int(st.st_size),
        "mtime": float(st.st_mtime),
    }


def _index_cache_key(graphml_path: str, params: Dict[str, Any]) -> str:
    sig = _graphml_signature(graphml_path)
    raw = json.dumps({"sig": sig, "params": params}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_or_build_index(
    *,
    graph: KnowledgeGraph,
    graphml_path: str,
    context_window_tokens: int,
    include_entity_contexts: bool,
    verbose: bool,
    use_index_cache: bool,
    index_cache_dir: str,
    rebuild_index: bool,
) -> Tuple[List[Any], Any, str]:
    params = {
        "context_window_tokens": int(context_window_tokens),
        "include_entity_contexts": bool(include_entity_contexts),
    }
    cache_key = _index_cache_key(graphml_path, params)
    os.makedirs(index_cache_dir, exist_ok=True)
    cache_path = os.path.join(index_cache_dir, f"index_cache_{cache_key}.pkl")

    if use_index_cache and (not rebuild_index) and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and payload.get("cache_key") == cache_key:
                if verbose:
                    print(f"[IndexCache] Loaded: {cache_path}")
                return payload["index_items"], payload["embeddings"], cache_path
        except Exception as e:
            if verbose:
                print(f"[IndexCache] Load failed, rebuilding: {e}")

    index_items, embeddings = build_retrieval_index_enhanced(
        graph,
        context_window_tokens=context_window_tokens,
        include_entity_contexts=include_entity_contexts,
        verbose=verbose,
    )

    if use_index_cache:
        try:
            payload = {
                "version": 1,
                "cache_key": cache_key,
                "graphml_signature": _graphml_signature(graphml_path),
                "params": params,
                "index_items": index_items,
                "embeddings": embeddings,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose:
                print(f"[IndexCache] Saved: {cache_path}")
        except Exception as e:
            if verbose:
                print(f"[IndexCache] Save failed (continuing without cache): {e}")

    return index_items, embeddings, cache_path


def _default_queries() -> List[str]:
    return [
        "Q1. What major application areas of Artificial Intelligence are discussed in the paper?",
        "Q2. According to the knowledge-graph structure built from the paper, how are AI application domains grouped into related communities?",
        "Q3. Which specific medical fields are mentioned in the paper where AI is applied, and what roles does AI play in those fields?",
        "Q4. How does the paper define Natural Language Processing (NLP), and what problem does it aim to solve?",
        "Q5. What real-world systems mentioned in the paper demonstrate the use of NLP, and what is their practical impact?",
        "Q6. How is Artificial Intelligence applied in the finance sector according to the paper?",
        "Q7. What roles does AI play in agriculture as described in the paper?",
        "Q8. Compare the use of AI in healthcare and finance as discussed in the paper.",
        "Q9. What relationships between Artificial Intelligence and its application domains can be inferred from the paper?",
        "Q10. What challenges and limitations of deploying AI systems in real-world environments are discussed in the paper?",
        "Q11. What areas of research and innovation related to AI are highlighted in the paper?",
        "Q12. Which AI application domain appears most emphasized in the paper, based on frequency and detail?",
    ]


def _compute_community_boost(query: str) -> float:
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


def _cand_id(c: Any) -> Optional[str]:
    if isinstance(c, dict):
        v = c.get("id")
        return str(v) if v is not None else None
    v = getattr(c, "id", None)
    return str(v) if v is not None else None


def _cand_type(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("item_type") or c.get("type") or "")
    return str(getattr(c, "item_type", "") or "")


def _cand_metadata(c: Any) -> Dict[str, Any]:
    if isinstance(c, dict):
        md = c.get("metadata")
        return md if isinstance(md, dict) else {}
    md = getattr(c, "metadata", None)
    return md if isinstance(md, dict) else {}


def _is_structural_query(query: str) -> bool:
    q = (query or "").lower()
    keywords = [
        "community",
        "communities",
        "cluster",
        "clusters",
        "grouped",
        "grouping",
        "organization",
        "organised",
        "organized",
        "hierarchy",
        "hierarchical",
        "structure",
        "partition",
    ]
    return any(k in q for k in keywords)


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_str(x: Any, default: str) -> str:
    try:
        s = str(x)
        return s if s else default
    except Exception:
        return default


def _extract_json_object(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1]
    return t


def _call_planner_llm(prompt: str, *, prefer_model: str, fallback_model: str, temperature: float, verbose: bool) -> str:
    last_error = None
    if os.getenv("GOOGLE_API_KEY"):
        try:
            from genai_compat import generate_text as genai_generate_text

            return (genai_generate_text(prefer_model, prompt, temperature=temperature) or "").strip()
        except Exception as e:
            last_error = e
            if verbose:
                print(f"[Planner] Gemini failed, trying OpenAI fallback: {e}")

    if os.getenv("OPENAI_API_KEY"):
        try:
            client = OpenAI()
            resp = client.chat.completions.create(
                model=fallback_model,
                messages=[
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_error = e
            if verbose:
                print(f"[Planner] OpenAI fallback failed: {e}")

    raise last_error or RuntimeError("No LLM provider configured (set GOOGLE_API_KEY or OPENAI_API_KEY)")


def _graph_stats(graph: KnowledgeGraph) -> Dict[str, Any]:
    try:
        nodes = list(graph.nodes.values())
    except Exception:
        nodes = []
    return {
        "nodes": len(getattr(graph, "nodes", {}) or {}),
        "edges": len(getattr(graph, "edges", []) or []),
        "community_nodes": sum(1 for n in nodes if getattr(n, "label", None) == "COMMUNITY"),
        "sentence_nodes": sum(1 for n in nodes if getattr(n, "label", None) == "SENTENCE"),
        "entity_domain_nodes": sum(1 for n in nodes if getattr(n, "label", None) in {"ENTITY", "DOMAIN"}),
    }


def plan_query_params(
    *,
    query: str,
    graph: KnowledgeGraph,
    graphml_path: str,
    defaults: Dict[str, Any],
    prefer_model: str,
    fallback_model: str,
    temperature: float,
    cache: DiskJSONCache,
    verbose: bool,
) -> Dict[str, Any]:
    """Agentic per-query parameter planner (LLM) with strict bounds and cache.

    Returns a dict with keys: rerank_mode, top_k_final, top_k_rerank, top_k_synthesis, max_evidence_chars.
    On any failure, returns defaults.
    """
    # Hard bounds (kept conservative)
    bounds = {
        "top_k_final": (10, 80),
        "top_k_rerank": (4, 24),
        "top_k_synthesis": (3, 12),
        "max_evidence_chars": (200, 1200),
    }
    allowed_modes = {"llm", "cross", "none"}

    sig = _graphml_signature(graphml_path)
    stats = _graph_stats(graph)
    qtype = classify_query(query)

    cache_key = DiskJSONCache.hash_key(
        query.strip(),
        json.dumps({"sig": sig, "stats": stats}, sort_keys=True),
        json.dumps(defaults, sort_keys=True),
        prefer_model,
        fallback_model,
    )

    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        return cached

    prompt = (
        "You are a GraphRAG parameter planner.\n"
        "Return VALID JSON ONLY with keys: rerank_mode, top_k_final, top_k_rerank, top_k_synthesis, max_evidence_chars.\n"
        "Allowed rerank_mode: [\"llm\",\"cross\",\"none\"].\n"
        f"Hard bounds: top_k_final in [{bounds['top_k_final'][0]},{bounds['top_k_final'][1]}], "
        f"top_k_rerank in [{bounds['top_k_rerank'][0]},{bounds['top_k_rerank'][1]}], "
        f"top_k_synthesis in [{bounds['top_k_synthesis'][0]},{bounds['top_k_synthesis'][1]}], "
        f"max_evidence_chars in [{bounds['max_evidence_chars'][0]},{bounds['max_evidence_chars'][1]}].\n"
        "Constraints: top_k_synthesis <= top_k_rerank <= top_k_final.\n"
        "Latency goal: prefer smaller values unless query is complex.\n"
        "If query is a definition/short factual, prefer rerank_mode=cross or none.\n"
        "If query is structural (communities/grouping/hierarchy), avoid too-small top_k_final; prefer rerank_mode=cross or llm.\n"
        "\n"
        f"Query type: {qtype}\n"
        f"Structural: {str(_is_structural_query(query))}\n"
        f"Graph stats: {json.dumps(stats, ensure_ascii=False)}\n"
        f"Defaults: {json.dumps(defaults, ensure_ascii=False)}\n"
        "\nUser query:\n"
        f"{query}\n"
    )

    try:
        raw = _call_planner_llm(
            prompt,
            prefer_model=prefer_model,
            fallback_model=fallback_model,
            temperature=float(temperature),
            verbose=verbose,
        )
        raw = _extract_json_object(raw)
        plan = json.loads(raw) if raw else {}
    except Exception as e:
        if verbose:
            print(f"[Planner] Failed; using defaults: {e}")
        return defaults

    rerank_mode = _safe_str(plan.get("rerank_mode"), defaults["rerank_mode"]).lower().strip()
    if rerank_mode not in allowed_modes:
        rerank_mode = defaults["rerank_mode"]

    def _clamp(name: str, value: Any) -> int:
        lo, hi = bounds[name]
        v = _safe_int(value, _safe_int(defaults.get(name), lo))
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    top_k_final = _clamp("top_k_final", plan.get("top_k_final"))
    top_k_rerank = _clamp("top_k_rerank", plan.get("top_k_rerank"))
    top_k_synthesis = _clamp("top_k_synthesis", plan.get("top_k_synthesis"))
    max_evidence_chars = _clamp("max_evidence_chars", plan.get("max_evidence_chars"))

    # Cross-field constraints
    top_k_rerank = min(top_k_rerank, top_k_final)
    top_k_synthesis = min(top_k_synthesis, top_k_rerank)

    # Structural safeguard: keep enough breadth to surface COMMUNITY evidence.
    if _is_structural_query(query):
        top_k_final = max(top_k_final, 25)
        top_k_rerank = max(min(top_k_rerank, top_k_final), 8)
        top_k_synthesis = max(min(top_k_synthesis, top_k_rerank), 6)
        if rerank_mode == "none":
            rerank_mode = "cross"

    final = {
        "rerank_mode": rerank_mode,
        "top_k_final": int(top_k_final),
        "top_k_rerank": int(top_k_rerank),
        "top_k_synthesis": int(top_k_synthesis),
        "max_evidence_chars": int(max_evidence_chars),
    }

    cache.set(cache_key, final)
    return final


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="GraphRAG Q&A from existing GraphML KG with debug + timing.")
    parser.add_argument("--graphml", default="knowledge_graph.graphml", help="Path to GraphML file")
    parser.add_argument("--queries-file", "-f", type=str, help="Text file with one query per line")
    parser.add_argument("--query", "-q", action="append", help="Query to run (can pass multiple)")
    parser.add_argument("--verbose", action="store_true", help="Print full debug logs")

    parser.add_argument("--top-n-semantic", type=int, default=20)
    parser.add_argument("--top-k-final", type=int, default=40)
    parser.add_argument("--top-k-rerank", type=int, default=12)
    parser.add_argument("--top-k-synthesis", type=int, default=10, help="How many reranked items to send to synthesis")
    parser.add_argument("--max-evidence-chars", type=int, default=650, help="Truncate evidence text for synthesis")
    parser.add_argument("--rerank-mode", choices=["llm", "cross", "none"], default="llm", help="Reranking strategy")
    parser.add_argument("--expansion-hops", type=int, default=1)

    parser.add_argument("--workers", type=int, default=1, help="Run queries concurrently (prints in original order)")
    parser.add_argument("--index-cache-dir", type=str, default="outputs", help="Directory for cached index/embeddings")
    parser.add_argument("--no-index-cache", action="store_true", help="Disable index/embedding cache")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuilding the index even if cache exists")

    parser.add_argument("--agentic", action="store_true", help="Use an LLM planner to choose per-query parameters")
    parser.add_argument("--planner-model", type=str, default="gemini-2.0-flash", help="Planner Gemini model")
    parser.add_argument("--planner-fallback-openai-model", type=str, default="gpt-4o-mini", help="Planner OpenAI fallback model")
    parser.add_argument("--planner-temperature", type=float, default=0.0, help="Planner temperature (recommend 0.0)")
    parser.add_argument("--planner-cache", type=str, default=os.path.join("outputs", "cache_agentic_plans.json"), help="Planner cache path")

    args = parser.parse_args()

    graphml_path = args.graphml
    if not os.path.exists(graphml_path):
        raise FileNotFoundError(f"GraphML not found: {graphml_path}")

    timing = Timing()

    with timing.time_block("load_graphml"):
        graph = import_graphml_to_kg(graphml_path)

    if args.verbose:
        print(f"[Load] GraphML: {graphml_path}")
        print(f"[Load] Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")

    with timing.time_block("build_retrieval_index_enhanced"):
        index_items, embeddings, cache_path = load_or_build_index(
            graph=graph,
            graphml_path=graphml_path,
            context_window_tokens=50,
            include_entity_contexts=True,
            verbose=args.verbose,
            use_index_cache=(not args.no_index_cache),
            index_cache_dir=args.index_cache_dir,
            rebuild_index=args.rebuild_index,
        )
    if args.verbose and (not args.no_index_cache):
        print(f"[IndexCache] Path: {cache_path}")

    queries: Optional[List[str]] = None
    if args.queries_file:
        if not os.path.exists(args.queries_file):
            raise FileNotFoundError(f"queries file not found: {args.queries_file}")
        with open(args.queries_file, "r", encoding="utf-8") as f:
            queries = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

    if args.query:
        queries = (queries or []) + args.query

    if not queries:
        queries = _default_queries()

    if not index_items:
        print("No index items available; cannot answer queries.")
        return

    planner_cache = DiskJSONCache(args.planner_cache)

    def _run_one(query_idx: int, query_text: str, verbose: bool) -> Dict[str, Any]:
        out_buf = io.StringIO()
        stage_times: Dict[str, float] = {}

        def _t(name: str, fn):
            t0 = time.perf_counter()
            try:
                return fn()
            finally:
                stage_times[name] = time.perf_counter() - t0

        with redirect_stdout(out_buf):
            plan = {
                "rerank_mode": args.rerank_mode,
                "top_k_final": int(args.top_k_final),
                "top_k_rerank": int(args.top_k_rerank),
                "top_k_synthesis": int(args.top_k_synthesis),
                "max_evidence_chars": int(args.max_evidence_chars),
            }
            if args.agentic:
                plan = _t(
                    "agentic_plan",
                    lambda: plan_query_params(
                        query=query_text,
                        graph=graph,
                        graphml_path=graphml_path,
                        defaults=plan,
                        prefer_model=args.planner_model,
                        fallback_model=args.planner_fallback_openai_model,
                        temperature=args.planner_temperature,
                        cache=planner_cache,
                        verbose=verbose,
                    ),
                )

            if verbose and args.agentic:
                print(f"[Planner] {json.dumps(plan, ensure_ascii=False)}")

            community_boost = _compute_community_boost(query_text)

            candidates = _t(
                "hybrid_search",
                lambda: search_and_expand(
                    query=query_text,
                    graph=graph,
                    index_items=index_items,
                    embeddings=embeddings,
                    top_n_semantic=args.top_n_semantic,
                    top_k_final=int(plan["top_k_final"]),
                    alpha=0.7,
                    beta=0.3,
                    community_boost=community_boost,
                    expansion_hops=args.expansion_hops,
                    verbose=verbose,
                ),
            )

            if verbose:
                print("[Debug] Top pre-rerank candidates:")
                for c in candidates[:10]:
                    meta = c.metadata or {}
                    lvl = meta.get("level")
                    coh = meta.get("coherence")
                    print(
                        f"  id={c.id} type={c.item_type} level={lvl} coh={coh} "
                        f"sem={c.semantic_score:.3f} graph={c.graph_score:.3f} hybrid={c.hybrid_score:.3f} path={c.retrieval_path}"
                    )

            rerank_result = _t(
                "rerank",
                lambda: llm_rerank_candidates(
                    query=query_text,
                    candidates=candidates,
                    top_k=int(plan["top_k_rerank"]),
                    rerank_mode=str(plan["rerank_mode"]),
                    use_cache=True,
                    verbose=verbose,
                ),
            )

            ranked = rerank_result.get("ranked_candidates", []) if isinstance(rerank_result, dict) else []

            if verbose:
                print("[Debug] Top post-rerank candidates:")
                for c in ranked[:10]:
                    cid = _cand_id(c)
                    ctype = _cand_type(c)
                    meta = _cand_metadata(c)
                    lvl = meta.get("level")
                    coh = meta.get("coherence")
                    print(f"  id={cid} type={ctype} level={lvl} coh={coh}")

            if not ranked:
                abst_id = handle_abstention(graph, query_text)
                synthesis_result = {"query": query_text, "answer": None, "abstention_node": abst_id}
            else:
                evidence_for_synthesis = ranked[: max(1, int(plan["top_k_synthesis"]))]
                synthesis_result = _t(
                    "synthesis",
                    lambda: llm_synthesize_answer(
                        query=query_text,
                        evidence_candidates=evidence_for_synthesis,
                        graph=graph,
                        use_cache=True,
                        verbose=verbose,
                        max_evidence_chars=int(plan["max_evidence_chars"]),
                    ),
                )

                if classify_query(query_text) == "howwhy":
                    top_entity_ids: List[str] = []
                    for cand in ranked:
                        if _cand_type(cand).upper() == "ENTITY":
                            cid = _cand_id(cand)
                            if cid:
                                top_entity_ids.append(cid)
                        if len(top_entity_ids) >= 6:
                            break
                    if top_entity_ids:
                        traversal = _t(
                            "multi_hop_traversal",
                            lambda: multi_hop_traversal(graph, top_entity_ids, hops=3),
                        )
                        try:
                            if isinstance(synthesis_result, dict):
                                synthesis_result["traversal_paths"] = traversal
                        except Exception:
                            pass

        return {
            "idx": query_idx,
            "query": query_text,
            "answer": format_answer_output(synthesis_result),
            "logs": out_buf.getvalue(),
            "stage_times": stage_times,
        }

    workers = max(1, int(args.workers))
    results: List[Dict[str, Any]] = []
    if workers == 1:
        for i, q in enumerate(queries, start=1):
            print(f"\n[Query {i}] {q}")
            r = _run_one(i, q, args.verbose)
            if args.verbose and r["logs"].strip():
                print(r["logs"].rstrip())
            print("\n" + r["answer"])
            results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_one, i, q, args.verbose) for i, q in enumerate(queries, start=1)]
            for f in futs:
                results.append(f.result())
        results.sort(key=lambda x: x["idx"])
        for r in results:
            print(f"\n[Query {r['idx']}] {r['query']}")
            if args.verbose and r["logs"].strip():
                print(r["logs"].rstrip())
            print("\n" + r["answer"])

    timing.print_summary()

    if results:
        totals: Dict[str, float] = {}
        for r in results:
            for k, v in (r.get("stage_times") or {}).items():
                totals[k] = totals.get(k, 0.0) + float(v)
        print("\n[TIMING] Per-query stage totals")
        for k in ["agentic_plan", "hybrid_search", "rerank", "synthesis", "multi_hop_traversal"]:
            if k in totals:
                print(f"  - {k}: {totals[k]:.3f}s")


if __name__ == "__main__":
    main()
