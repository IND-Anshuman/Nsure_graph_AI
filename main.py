"""main.py

Flask API wrapper around the Knowledge-Graph-Agent pipeline.

Endpoints:
- GET  /health
- POST /query

/query accepts either:
1) multipart/form-data with fields:
   - pdf: uploaded PDF file
   - questions: JSON array string OR a single string
   - options: (optional) JSON object string

2) application/json with fields:
   - pdf_url: URL to a PDF (or other supported source)
   - OR pdf_path: local path (disabled by default; see KG_ALLOW_LOCAL_PATH)
   - questions: list[str] OR string
   - options: optional object

It builds a KG for the provided source and answers each question.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from graph_maker import build_knowledge_graph
from graph_save import save_kg_to_graphml
from phase8_retrieval_enhanced import IndexItem, build_retrieval_index_enhanced
from hybrid_search import search_and_expand
from llm_rerank import llm_rerank_candidates
from llm_synthesis import llm_synthesize_answer
from Community_processing import classify_query, handle_abstention
from engine import import_graphml_to_kg


load_dotenv()

app = Flask(__name__)
app.url_map.strict_slashes = False

# Enable CORS for frontend communication
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://localhost:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
    }
})


_API_CACHE_DIR = Path(os.getenv("KG_API_CACHE_DIR", "outputs/api_cache"))
_API_CACHE_VERSION = 1


def _now_ms() -> int:
    return int(time.time() * 1000)


def _coerce_questions(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    s = str(value).strip()
    if not s:
        return []
    # If it's a JSON list, accept it.
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("\"") and s.endswith("\"")):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x).strip()]
            if isinstance(parsed, str) and parsed.strip():
                return [parsed.strip()]
        except Exception:
            pass
    return [s]


def _read_json_field(raw: Any, *, default: Any) -> Any:
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    s = str(raw).strip()
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


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


def _save_uploaded_pdf(upload_dir: Path) -> str:
    f = request.files.get("pdf")
    if f is None:
        raise ValueError("Missing multipart field 'pdf'")
    filename = secure_filename(f.filename or "document.pdf")
    if not filename.lower().endswith(".pdf"):
        # We keep this strict for safety: your corpus loader supports other types,
        # but the API contract here is "pdf".
        raise ValueError("Uploaded file must be a .pdf")

    upload_dir.mkdir(parents=True, exist_ok=True)
    out_path = upload_dir / f"{_now_ms()}_{filename}"
    f.save(str(out_path))
    return str(out_path)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()


def _canonical_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)


def _cache_key_for_build(*, source: str, build_opts: Dict[str, Any]) -> str:
    # For local files, hash bytes so uploads with different filenames still reuse cache.
    try:
        if os.path.exists(source):
            src_sig = f"file:{_sha256_file(source)}"
        else:
            src_sig = f"url:{_sha256_text(source)}"
    except Exception:
        src_sig = f"src:{_sha256_text(source)}"

    key_payload = {
        "v": _API_CACHE_VERSION,
        "src": src_sig,
        "build": build_opts,
    }
    return _sha256_text(_canonical_json(key_payload))


def _cache_key_for_index(*, graph_key: str, index_opts: Dict[str, Any]) -> str:
    # Include embedding config to avoid mixing vectors from different models.
    emb_model = os.getenv("KG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    emb_dim = int(os.getenv("KG_EMBEDDING_DIM", "384") or 384)
    key_payload = {
        "v": _API_CACHE_VERSION,
        "graph_key": graph_key,
        "index": index_opts,
        "emb": {"model": emb_model, "dim": emb_dim},
    }
    return _sha256_text(_canonical_json(key_payload))


def _graph_cache_paths(graph_key: str) -> Dict[str, Path]:
    base = _API_CACHE_DIR / "graphs" / graph_key
    return {
        "base": base,
        "graphml": base / "graph.graphml",
        "meta": base / "meta.json",
    }


def _index_cache_paths(index_key: str) -> Dict[str, Path]:
    base = _API_CACHE_DIR / "indexes" / index_key
    return {
        "base": base,
        "items": base / "index_items.json",
        "embeddings": base / "embeddings.npy",
        "meta": base / "meta.json",
    }


def _load_index_items(path: Path) -> List[IndexItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("index_items.json must be a list")
    items: List[IndexItem] = []
    for obj in payload:
        if not isinstance(obj, dict):
            continue
        items.append(
            IndexItem(
                id=str(obj.get("id")),
                text=str(obj.get("text")),
                item_type=str(obj.get("item_type")),
                metadata=dict(obj.get("metadata") or {}),
            )
        )
    return items


def _save_index_items(path: Path, items: List[IndexItem]) -> None:
    data = [
        {"id": it.id, "text": it.text, "item_type": it.item_type, "metadata": it.metadata}
        for it in items
    ]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _resolve_cache_options(options: Dict[str, Any]) -> Dict[str, Any]:
    cache_opts = options.get("cache", {}) if isinstance(options.get("cache"), dict) else {}
    enabled = bool(cache_opts.get("enabled", True))
    rebuild_graph = bool(cache_opts.get("rebuild_graph", False))
    rebuild_index = bool(cache_opts.get("rebuild_index", False))
    return {"enabled": enabled, "rebuild_graph": rebuild_graph, "rebuild_index": rebuild_index}


def _get_or_build_graph(
    *,
    source: str,
    verbose: bool,
    build_opts: Dict[str, Any],
    cache: Dict[str, Any],
) -> tuple[Any, str, str]:
    graph_key = _cache_key_for_build(source=source, build_opts=build_opts)
    paths = _graph_cache_paths(graph_key)
    graphml_path = str(paths["graphml"])

    if cache["enabled"] and (not cache["rebuild_graph"]) and paths["graphml"].exists():
        graph = import_graphml_to_kg(graphml_path)
        return graph, graph_key, graphml_path

    graph = build_knowledge_graph(
        sources=[source],
        verbose=verbose,
        skip_relations=bool(build_opts.get("skip_relations", False)),
        skip_communities=bool(build_opts.get("skip_communities", False)),
        skip_community_summaries=bool(build_opts.get("skip_community_summaries", True)),
    )

    if cache["enabled"]:
        paths["base"].mkdir(parents=True, exist_ok=True)
        save_kg_to_graphml(graph, graphml_path)
        paths["meta"].write_text(
            json.dumps(
                {
                    "version": _API_CACHE_VERSION,
                    "graph_key": graph_key,
                    "source": source,
                    "build_opts": build_opts,
                    "counts": {"nodes": len(graph.nodes), "edges": len(graph.edges)},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return graph, graph_key, graphml_path


def _get_or_build_index(
    *,
    graph: Any,
    graph_key: str,
    verbose: bool,
    index_opts: Dict[str, Any],
    cache: Dict[str, Any],
):
    import numpy as np

    index_key = _cache_key_for_index(graph_key=graph_key, index_opts=index_opts)
    paths = _index_cache_paths(index_key)

    if cache["enabled"] and (not cache["rebuild_index"]) and paths["items"].exists() and paths["embeddings"].exists():
        index_items = _load_index_items(paths["items"])
        embeddings = np.load(str(paths["embeddings"]), allow_pickle=False)
        return index_items, embeddings, index_key

    index_items, embeddings = build_retrieval_index_enhanced(
        graph,
        context_window_tokens=int(index_opts.get("context_window_tokens", 50) or 50),
        include_entity_contexts=bool(index_opts.get("include_entity_contexts", True)),
        include_provision_contexts=bool(index_opts.get("include_provision_contexts", True)),
        provision_window_sentences=int(index_opts.get("provision_window_sentences", 3) or 3),
        chunk_size_words=int(index_opts.get("chunk_size_words", 160) or 160),
        chunk_overlap_words=int(index_opts.get("chunk_overlap_words", 40) or 40),
        min_sentence_chars=int(index_opts.get("min_sentence_chars", 25) or 25),
        min_text_chars=int(index_opts.get("min_text_chars", 40) or 40),
        verbose=verbose,
    )

    if cache["enabled"]:
        paths["base"].mkdir(parents=True, exist_ok=True)
        _save_index_items(paths["items"], index_items)
        np.save(str(paths["embeddings"]), embeddings)
        paths["meta"].write_text(
            json.dumps(
                {
                    "version": _API_CACHE_VERSION,
                    "index_key": index_key,
                    "graph_key": graph_key,
                    "index_opts": index_opts,
                    "counts": {"items": len(index_items)},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return index_items, embeddings, index_key


def _safe_graphml_output_path(*, output_dir: Path, requested_name: Optional[str] = None) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    if requested_name:
        name = secure_filename(str(requested_name))
        if not name:
            name = f"knowledge_graph_{_now_ms()}.graphml"
        if not name.lower().endswith(".graphml"):
            name = f"{name}.graphml"
    else:
        name = f"knowledge_graph_{_now_ms()}.graphml"
    return str(output_dir / name)


def _resolve_source_from_request() -> str:
    # Prefer file upload if present.
    if request.files and request.files.get("pdf") is not None:
        base = Path(os.getenv("KG_API_UPLOAD_DIR", "outputs/api_uploads"))
        return _save_uploaded_pdf(base)

    # Else JSON body.
    payload = request.get_json(silent=True) or {}
    pdf_url = (payload.get("pdf_url") or payload.get("url") or "").strip()
    if pdf_url:
        return pdf_url

    pdf_path = (payload.get("pdf_path") or payload.get("path") or "").strip()
    if pdf_path:
        allow = (os.getenv("KG_ALLOW_LOCAL_PATH", "0") or "0").strip() == "1"
        if not allow:
            raise ValueError("Local paths are disabled. Set KG_ALLOW_LOCAL_PATH=1 to allow 'pdf_path'.")
        return pdf_path

    raise ValueError("Provide either an uploaded 'pdf' file, 'pdf_url', or (if enabled) 'pdf_path'.")


def _safe_int(x: Any, default: int) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _answer_one(
    *,
    q: str,
    graph: Any,
    index_items: List[IndexItem],
    embeddings: Any,
    verbose: bool,
    qa_opts: Dict[str, Any],
    top_n_semantic: int,
    top_k_final: int,
    rerank_top_k: int,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    q = str(q).strip()
    qtype = classify_query(q)
    community_boost = _safe_float(qa_opts.get("community_boost"), _community_boost_for_query(q))

    candidates = search_and_expand(
        query=q,
        graph=graph,
        index_items=index_items,
        embeddings=embeddings,
        top_n_semantic=top_n_semantic,
        top_k_final=top_k_final,
        alpha=_safe_float(qa_opts.get("alpha"), 0.7),
        beta=_safe_float(qa_opts.get("beta"), 0.3),
        community_boost=community_boost,
        expansion_hops=_safe_int(qa_opts.get("expansion_hops"), 1),
        verbose=verbose,
    )
    t_retr = time.perf_counter()

    ranked: List[Any] = []
    try:
        rr = llm_rerank_candidates(
            query=q,
            candidates=candidates,
            top_k=rerank_top_k,
            rerank_mode=str(qa_opts.get("rerank_mode", "llm") or "llm"),
            model_name=str(qa_opts.get("rerank_model", "gemini-2.0-flash") or "gemini-2.0-flash"),
            fallback_openai_model=str(qa_opts.get("rerank_fallback_openai_model", "gpt-4o-mini") or "gpt-4o-mini"),
            temperature=_safe_float(qa_opts.get("rerank_temperature"), 0.1),
            use_cross_encoder=bool(qa_opts.get("use_cross_encoder", True)),
            cross_encoder_model=str(qa_opts.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2") or "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            cross_top_k=_safe_int(qa_opts.get("cross_top_k"), 60),
            use_cache=True,
            verbose=verbose,
        )
        if isinstance(rr, dict):
            ranked = rr.get("ranked_candidates", []) or []
    except Exception:
        ranked = candidates[: max(1, rerank_top_k)]
    t_rerank = time.perf_counter()

    if not ranked:
        abst_id = handle_abstention(graph, q)
        return {
            "question": q,
            "query_type": qtype,
            "result": {"query": q, "answer": None, "abstention_node": abst_id},
            "timing_s": {"retrieval": round(t_retr - t0, 4), "rerank": round(t_rerank - t_retr, 4), "synthesis": 0.0, "total": round(t_rerank - t0, 4)},
        }

    try:
        result = llm_synthesize_answer(
            query=q,
            evidence_candidates=ranked,
            graph=graph,
            model_name=str(qa_opts.get("synthesis_model", "gemini-2.0-flash") or "gemini-2.0-flash"),
            fallback_openai_model=str(qa_opts.get("synthesis_fallback_openai_model", "gpt-4o-mini") or "gpt-4o-mini"),
            temperature=_safe_float(qa_opts.get("synthesis_temperature"), 0.2),
            max_evidence_chars=_safe_int(qa_opts.get("max_evidence_chars"), 1200),
            use_cache=True,
            verbose=verbose,
        )
    except Exception:
        result = llm_synthesize_answer(
            query=q,
            evidence_candidates=ranked,
            graph=graph,
            evidence_only=True,
            use_cache=False,
            verbose=verbose,
        )
    t_syn = time.perf_counter()

    return {
        "question": q,
        "query_type": qtype,
        "result": result,
        "timing_s": {"retrieval": round(t_retr - t0, 4), "rerank": round(t_rerank - t_retr, 4), "synthesis": round(t_syn - t_rerank, 4), "total": round(t_syn - t0, 4)},
    }


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.get("/routes")
def routes() -> Any:
    """List available routes (debug helper)."""
    rules = sorted({str(r) for r in app.url_map.iter_rules()})
    return jsonify({"routes": rules})


@app.post("/build")
def build() -> Any:
    """Build and persist a KG as GraphML.

    Input matches /query (multipart upload or JSON with pdf_url).
    Options:
      - options.build.* controls graph build (same as /query)
      - options.output.graphml_name optional filename (sanitized)
      - options.output.output_dir optional directory (default outputs/api_graphs)
    """
    t0 = time.perf_counter()

    source = _resolve_source_from_request()

    if request.files and request.files.get("pdf") is not None:
        options = _read_json_field(request.form.get("options"), default={})
    else:
        payload = request.get_json(silent=True) or {}
        options = _read_json_field(payload.get("options"), default={})

    verbose = bool(options.get("verbose", False))

    build_opts = options.get("build", {}) if isinstance(options.get("build"), dict) else {}
    cache = _resolve_cache_options(options)
    graph, graph_key, cached_graphml_path = _get_or_build_graph(
        source=source,
        verbose=verbose,
        build_opts=build_opts,
        cache=cache,
    )

    t_graph = time.perf_counter()

    out_opts = options.get("output", {}) if isinstance(options.get("output"), dict) else {}
    out_dir = Path(str(out_opts.get("output_dir") or os.getenv("KG_API_GRAPHS_DIR", "outputs/api_graphs")))
    graphml_name = out_opts.get("graphml_name")
    graphml_path = _safe_graphml_output_path(output_dir=out_dir, requested_name=str(graphml_name) if graphml_name else None)

    # If user requested a specific output file, write/copy it there.
    if os.path.abspath(cached_graphml_path) != os.path.abspath(graphml_path):
        out_dir = Path(os.path.dirname(graphml_path))
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached_graphml_path, graphml_path)
    t_done = time.perf_counter()

    return jsonify(
        {
            "source": source,
            "graph_key": graph_key,
            "graphml_path": graphml_path,
            "cached_graphml_path": cached_graphml_path,
            "counts": {"nodes": len(graph.nodes), "edges": len(graph.edges)},
            "timing_s": {"graph_build": round(t_graph - t0, 4), "persist": round(t_done - t_graph, 4), "total": round(t_done - t0, 4)},
        }
    )


@app.post("/query")
def query() -> Any:
    t0 = time.perf_counter()

    # Parse inputs
    source = _resolve_source_from_request()

    if request.files and request.files.get("pdf") is not None:
        questions = _coerce_questions(request.form.get("questions"))
        options = _read_json_field(request.form.get("options"), default={})
    else:
        payload = request.get_json(silent=True) or {}
        questions = _coerce_questions(payload.get("questions") or payload.get("question"))
        options = _read_json_field(payload.get("options"), default={})

    if not questions:
        return jsonify({"error": "No questions provided"}), 400

    verbose = bool(options.get("verbose", False))

    cache = _resolve_cache_options(options)

    # Build graph
    build_opts = options.get("build", {}) if isinstance(options.get("build"), dict) else {}
    graph, graph_key, graphml_path = _get_or_build_graph(
        source=source,
        verbose=verbose,
        build_opts=build_opts,
        cache=cache,
    )

    t_graph = time.perf_counter()

    # Build index
    index_opts = options.get("index", {}) if isinstance(options.get("index"), dict) else {}
    index_items, embeddings, index_key = _get_or_build_index(
        graph=graph,
        graph_key=graph_key,
        verbose=verbose,
        index_opts=index_opts,
        cache=cache,
    )

    if not index_items:
        return jsonify({"error": "No index items were built (graph may be empty)."}), 500

    t_index = time.perf_counter()

    # Retrieval/rerank/synthesis knobs
    qa_opts = options.get("qa", {}) if isinstance(options.get("qa"), dict) else {}
    top_n_semantic = int(qa_opts.get("top_n_semantic", 20) or 20)
    top_k_final = int(qa_opts.get("top_k_final", 40) or 40)
    rerank_top_k = int(qa_opts.get("rerank_top_k", 12) or 12)

    clean_questions = [str(q).strip() for q in questions if str(q).strip()]

    # Parallelize per-question work to reduce wall-clock time.
    # Keep concurrency modest to avoid LLM rate limits.
    env_workers = os.getenv("KG_QA_MAX_WORKERS") or os.getenv("KG_QA_WORKERS") or "2"
    requested_workers = _safe_int(qa_opts.get("max_workers") or qa_opts.get("workers"), int(env_workers or 2))
    max_workers = max(1, min(len(clean_questions), requested_workers))

    answers: List[Dict[str, Any]] = []
    if max_workers <= 1 or len(clean_questions) <= 1:
        for q in clean_questions:
            answers.append(
                _answer_one(
                    q=q,
                    graph=graph,
                    index_items=index_items,
                    embeddings=embeddings,
                    verbose=verbose,
                    qa_opts=qa_opts,
                    top_n_semantic=top_n_semantic,
                    top_k_final=top_k_final,
                    rerank_top_k=rerank_top_k,
                )
            )
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, q in enumerate(clean_questions):
                fut = ex.submit(
                    _answer_one,
                    q=q,
                    graph=graph,
                    index_items=index_items,
                    embeddings=embeddings,
                    verbose=verbose,
                    qa_opts=qa_opts,
                    top_n_semantic=top_n_semantic,
                    top_k_final=top_k_final,
                    rerank_top_k=rerank_top_k,
                )
                futures[fut] = i
            results: List[Tuple[int, Dict[str, Any]]] = []
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results.append((idx, fut.result()))
                except Exception as e:
                    results.append((idx, {"question": clean_questions[idx], "error": str(e)}))
            results.sort(key=lambda x: x[0])
            answers = [r for _, r in results]

    t_done = time.perf_counter()

    return jsonify(
        {
            "source": source,
            "graph_key": graph_key,
            "graphml_path": graphml_path,
            "index_key": index_key,
            "counts": {"nodes": len(graph.nodes), "edges": len(graph.edges)},
            "timing_s": {
                "graph_build": round(t_graph - t0, 4),
                "index_build": round(t_index - t_graph, 4),
                "qa_total": round(t_done - t_index, 4),
                "total": round(t_done - t0, 4),
            },
            "answers": answers,
        }
    )


@app.post("/query_graphml")
def query_graphml() -> Any:
    """Answer questions from an existing GraphML file without rebuilding the KG.

    JSON body:
      - graphml_path: str
      - questions: list[str] or str
      - options: same shape as /query (index/qa/cache/verbose)
    """
    t0 = time.perf_counter()
    payload = request.get_json(silent=True) or {}

    graphml_path = str(payload.get("graphml_path") or payload.get("graphml") or "").strip()
    if not graphml_path:
        return jsonify({"error": "Missing graphml_path"}), 400
    if not os.path.exists(graphml_path):
        return jsonify({"error": f"GraphML not found: {graphml_path}"}), 404

    questions = _coerce_questions(payload.get("questions") or payload.get("question"))
    if not questions:
        return jsonify({"error": "No questions provided"}), 400

    options = _read_json_field(payload.get("options"), default={})
    verbose = bool(options.get("verbose", False))
    cache = _resolve_cache_options(options)

    graph = import_graphml_to_kg(graphml_path)
    t_graph = time.perf_counter()

    # Index cache key is derived from graphml content signature + index options.
    # We approximate this by hashing file bytes (stable for a given GraphML).
    graph_key = _sha256_file(graphml_path)
    index_opts = options.get("index", {}) if isinstance(options.get("index"), dict) else {}
    index_items, embeddings, index_key = _get_or_build_index(
        graph=graph,
        graph_key=graph_key,
        verbose=verbose,
        index_opts=index_opts,
        cache=cache,
    )
    t_index = time.perf_counter()

    qa_opts = options.get("qa", {}) if isinstance(options.get("qa"), dict) else {}
    top_n_semantic = int(qa_opts.get("top_n_semantic", 20) or 20)
    top_k_final = int(qa_opts.get("top_k_final", 40) or 40)
    rerank_top_k = int(qa_opts.get("rerank_top_k", 12) or 12)

    clean_questions = [str(q).strip() for q in questions if str(q).strip()]
    env_workers = os.getenv("KG_QA_MAX_WORKERS") or os.getenv("KG_QA_WORKERS") or "2"
    requested_workers = _safe_int(qa_opts.get("max_workers") or qa_opts.get("workers"), int(env_workers or 2))
    max_workers = max(1, min(len(clean_questions), requested_workers))

    answers: List[Dict[str, Any]] = []
    if max_workers <= 1 or len(clean_questions) <= 1:
        for q in clean_questions:
            answers.append(
                _answer_one(
                    q=q,
                    graph=graph,
                    index_items=index_items,
                    embeddings=embeddings,
                    verbose=verbose,
                    qa_opts=qa_opts,
                    top_n_semantic=top_n_semantic,
                    top_k_final=top_k_final,
                    rerank_top_k=rerank_top_k,
                )
            )
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, q in enumerate(clean_questions):
                fut = ex.submit(
                    _answer_one,
                    q=q,
                    graph=graph,
                    index_items=index_items,
                    embeddings=embeddings,
                    verbose=verbose,
                    qa_opts=qa_opts,
                    top_n_semantic=top_n_semantic,
                    top_k_final=top_k_final,
                    rerank_top_k=rerank_top_k,
                )
                futures[fut] = i
            results: List[Tuple[int, Dict[str, Any]]] = []
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results.append((idx, fut.result()))
                except Exception as e:
                    results.append((idx, {"question": clean_questions[idx], "error": str(e)}))
            results.sort(key=lambda x: x[0])
            answers = [r for _, r in results]

    t_done = time.perf_counter()
    return jsonify(
        {
            "graphml_path": graphml_path,
            "graph_key": graph_key,
            "index_key": index_key,
            "counts": {"nodes": len(graph.nodes), "edges": len(graph.edges)},
            "timing_s": {
                "load_graphml": round(t_graph - t0, 4),
                "index_build": round(t_index - t_graph, 4),
                "qa_total": round(t_done - t_index, 4),
                "total": round(t_done - t0, 4),
            },
            "answers": answers,
        }
    )


if __name__ == "__main__":
    # Flask dev server (use a proper WSGI server in production).
    host = os.getenv("KG_API_HOST", "127.0.0.1")
    port = int(os.getenv("KG_API_PORT", "5000") or 5000)
    debug = (os.getenv("KG_API_DEBUG", "0") or "0").strip() == "1"
    app.run(host=host, port=port, debug=debug)
