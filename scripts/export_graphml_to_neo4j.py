"""Export a GraphML knowledge graph into Neo4j.
Reads a .graphml file via NetworkX and imports it into Neo4j using the official
Neo4j Python driver.

- Nodes are created with a stable `id` (the GraphML node id).
- Labels: always includes :KGNode, plus an optional label derived from the
  GraphML node attribute `label` (e.g. DOCUMENT) if present.
- Relationships: created with a type derived from GraphML edge attribute `type`
  if present, otherwise RELATED_TO. Each relationship also stores all edge
  attributes as properties.

Usage (PowerShell):
  python scripts/export_graphml_to_neo4j.py \
    --graphml outputs/api_cache/graphs/<hash>/graph.graphml \
    --uri bolt://localhost:7687 --user neo4j --password <pwd> --database neo4j

You can also set env vars:
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
"""

from __future__ import annotations

import argparse
import math
import os
import re
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx
from dotenv import find_dotenv, load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable


_LABEL_RE = re.compile(r"[^0-9A-Za-z_]")


def _sanitize_schema_name(name: str, *, prefix: str = "") -> str:
    """Make a Neo4j-safe label/type name.

    Neo4j identifiers for labels/types must be non-empty and typically use
    alphanumerics/underscore. We also ensure the name doesn't start with a digit.
    """

    name = (name or "").strip()
    if not name:
        return f"{prefix}UNKNOWN" if prefix else "UNKNOWN"

    name = _LABEL_RE.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        return f"{prefix}UNKNOWN" if prefix else "UNKNOWN"
    if name[0].isdigit():
        name = f"{prefix or 'X_'}{name}"
    return name


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _infer_node_name(node_id: str, attrs: Dict[str, Any]) -> str:
    """Best-effort display name for Neo4j Browser."""

    for key in ("canonical", "title", "text", "description"):
        value = attrs.get(key)
        if isinstance(value, str) and value.strip():
            value_1 = _one_line(value)
            # Keep the display name short-ish; full text remains in properties.
            return value_1[:160]

    label = attrs.get("label")
    if isinstance(label, str) and label.strip():
        doc_id = attrs.get("doc_id")
        if isinstance(doc_id, str) and doc_id.strip():
            return f"{_one_line(label)}:{_one_line(doc_id)}"
        return _one_line(label)

    return str(node_id)


def _clean_props(props: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None/NaN props; keep values Neo4j can store."""

    cleaned: Dict[str, Any] = {}
    for key, value in (props or {}).items():
        if key is None:
            continue
        key_str = str(key)
        if not key_str:
            continue

        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue

        # Neo4j supports primitives and lists of primitives.
        # GraphML often contains strings; we keep them as-is.
        if isinstance(value, (str, int, bool, float)):
            cleaned[key_str] = value
        elif isinstance(value, (list, tuple)):
            # Keep only primitive list items; coerce others to string.
            lst: List[Any] = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, float) and math.isnan(item):
                    continue
                if isinstance(item, (str, int, bool, float)):
                    lst.append(item)
                else:
                    lst.append(str(item))
            cleaned[key_str] = lst
        else:
            cleaned[key_str] = str(value)

    return cleaned


def _iter_batches(rows: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def _read_graphml(graphml_path: str) -> nx.Graph:
    # networkx will infer Graph / DiGraph / MultiDiGraph based on file.
    return nx.read_graphml(graphml_path)


def _prepare_nodes(graph: nx.Graph) -> Dict[Tuple[str, ...], List[Dict[str, Any]]]:
    """Group nodes by (label, type) label-combos for efficient import."""

    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for node_id, attrs in graph.nodes(data=True):
        attrs = dict(attrs or {})

        labels: List[str] = []
        raw_label = attrs.get("label")
        raw_type = attrs.get("type")

        if isinstance(raw_label, str) and raw_label.strip():
            labels.append(_sanitize_schema_name(raw_label, prefix="N_"))
        if isinstance(raw_type, str) and raw_type.strip():
            type_label = _sanitize_schema_name(raw_type, prefix="N_")
            if type_label not in labels:
                labels.append(type_label)

        label_combo = tuple(sorted(set(labels)))

        props = _clean_props(attrs)
        props.pop("id", None)
        props.pop("_id", None)
        props.setdefault("name", _infer_node_name(str(node_id), attrs))

        row = {"id": str(node_id), "props": props}
        groups.setdefault(label_combo, []).append(row)

    return groups


def _prepare_edges(graph: nx.Graph) -> Dict[str, List[Dict[str, Any]]]:
    """Group edges by sanitized relationship type for efficient import."""

    groups: Dict[str, List[Dict[str, Any]]] = {}

    # Handle MultiGraph / MultiDiGraph as well.
    if graph.is_multigraph():
        edge_iter = graph.edges(keys=True, data=True)
        for u, v, k, attrs in edge_iter:  # type: ignore[misc]
            attrs = dict(attrs or {})
            raw_type = attrs.get("type") or attrs.get("label") or "RELATED_TO"
            rel_type = _sanitize_schema_name(str(raw_type), prefix="REL_")

            props = _clean_props(attrs)
            edge_id = f"{u}|{v}|{k}"
            row = {"source": str(u), "target": str(v), "id": edge_id, "props": props}
            groups.setdefault(rel_type, []).append(row)
    else:
        edge_iter2 = graph.edges(data=True)
        for idx, (u, v, attrs) in enumerate(edge_iter2):
            attrs = dict(attrs or {})
            raw_type = attrs.get("type") or attrs.get("label") or "RELATED_TO"
            rel_type = _sanitize_schema_name(str(raw_type), prefix="REL_")

            props = _clean_props(attrs)
            edge_id = f"{u}|{v}|{idx}"
            row = {"source": str(u), "target": str(v), "id": edge_id, "props": props}
            groups.setdefault(rel_type, []).append(row)

    return groups


def _clear_database(driver, database: str) -> None:
    print(f"Clearing existing database '{database}'...")
    with driver.session(database=database) as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Database cleared.")


def _ensure_constraints(driver, database: str) -> None:
    cypher = """
    CREATE CONSTRAINT kg_node_id_unique IF NOT EXISTS
    FOR (n:KGNode)
    REQUIRE n.id IS UNIQUE
    """
    with driver.session(database=database) as session:
        session.run(cypher)


def _import_nodes(driver, database: str, node_groups: Dict[Tuple[str, ...], List[Dict[str, Any]]], batch_size: int) -> int:
    total = 0
    with driver.session(database=database) as session:
        for label_combo, rows in node_groups.items():
            extra_labels = "".join(f":{_sanitize_schema_name(lbl, prefix='N_')}" for lbl in label_combo if lbl)
            query = f"""
            UNWIND $rows AS row
            MERGE (n {{id: row.id}})
            SET n:KGNode{extra_labels}
            SET n += row.props
            """

            for batch in _iter_batches(rows, batch_size):
                session.run(query, rows=batch)
                total += len(batch)

    return total


def _import_edges(driver, database: str, edge_groups: Dict[str, List[Dict[str, Any]]], batch_size: int) -> int:
    total = 0
    with driver.session(database=database) as session:
        for rel_type, rows in edge_groups.items():
            rel_type_s = _sanitize_schema_name(rel_type, prefix="REL_")
            query = f"""
            UNWIND $rows AS row
            MATCH (a:KGNode {{id: row.source}})
            MATCH (b:KGNode {{id: row.target}})
            MERGE (a)-[r:{rel_type_s} {{id: row.id}}]->(b)
            SET r += row.props
            """

            for batch in _iter_batches(rows, batch_size):
                session.run(query, rows=batch)
                total += len(batch)

    return total


def main() -> int:
    # Load environment variables from a local .env file if present.
    # This allows using NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD as configured.
    # NOTE: override=True so a globally-set NEO4J_URI doesn't accidentally win.
    load_dotenv(find_dotenv(usecwd=True), override=True)

    parser = argparse.ArgumentParser(description="Export GraphML to Neo4j")
    parser.add_argument("--graphml", required=True, help="Path to .graphml file")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"), help="Neo4j Bolt URI")
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"), help="Neo4j username")
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "Anshuman@2005"), help="Neo4j password")
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"), help="Neo4j database")
    parser.add_argument("--batch-size", type=int, default=1000, help="UNWIND batch size")
    parser.add_argument("--clear", action="store_true", help="Clear the database before importing")

    args = parser.parse_args()

    if not os.path.exists(args.graphml):
        raise FileNotFoundError(f"GraphML not found: {args.graphml}")
    if not args.password:
        raise SystemExit("Missing Neo4j password. Provide --password or set NEO4J_PASSWORD.")

    graph = _read_graphml(args.graphml)

    node_groups = _prepare_nodes(graph)
    edge_groups = _prepare_edges(graph)

    print(f"Connecting to Neo4j at '{args.uri}' (db='{args.database}', user='{args.user}')")

    try:
        driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
        try:
            if args.clear:
                _clear_database(driver, args.database)
            _ensure_constraints(driver, args.database)
            node_count = _import_nodes(driver, args.database, node_groups, args.batch_size)
            edge_count = _import_edges(driver, args.database, edge_groups, args.batch_size)
        finally:
            driver.close()
    except AuthError as exc:
        raise SystemExit(
            "Neo4j authentication failed. Verify NEO4J_USER/NEO4J_PASSWORD (or --user/--password)."
        ) from exc
    except ServiceUnavailable as exc:
        raise SystemExit(
            "Neo4j is unreachable. Ensure Neo4j is running and Bolt is enabled on 127.0.0.1:7687. "
            "If you're using Neo4j Desktop, start the DB and confirm the Bolt port. "
            "For local single-instance Neo4j, prefer a 'bolt://' URI (not 'neo4j://')."
        ) from exc

    print(f"Imported {node_count} nodes and {edge_count} relationships into database '{args.database}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
