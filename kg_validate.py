from __future__ import annotations

from typing import Iterable, Optional, Set, Tuple

from data_corpus import KnowledgeGraph


def validate_graph(
    graph: KnowledgeGraph,
    *,
    preserve_labels: Optional[Set[str]] = None,
) -> Tuple[int, int, int, int]:
    """Cheap KG integrity checks.

    Returns:
      (dangling_edges, missing_endpoint_refs, bad_member_of, bad_part_of)

    - dangling_edges: edges where source/target is missing
    - missing_endpoint_refs: total missing endpoints counted (source+target)
    - bad_member_of: MEMBER_OF edges that are not ENTITY -> COMMUNITY
    - bad_part_of: PART_OF edges that point to non-existent nodes
    """
    preserve_labels = preserve_labels or set()

    node_ids = set(graph.nodes.keys())
    dangling_edges = 0
    missing_refs = 0
    bad_member_of = 0
    bad_part_of = 0

    for e in graph.edges:
        src_ok = e.source in node_ids
        tgt_ok = e.target in node_ids
        if not src_ok or not tgt_ok:
            dangling_edges += 1
            if not src_ok:
                missing_refs += 1
            if not tgt_ok:
                missing_refs += 1
            # Don't try to type-check further.
            continue

        if e.type == "MEMBER_OF":
            src = graph.nodes.get(e.source)
            tgt = graph.nodes.get(e.target)
            # DOMAIN nodes are concept-like (high recall / promoted terms) and are allowed
            # to participate in communities.
            if not src or not tgt or src.label not in {"ENTITY", "DOMAIN"} or tgt.label != "COMMUNITY":
                bad_member_of += 1

        if e.type == "PART_OF":
            # PART_OF is used for multiple semantics in this project; basic existence is enough.
            # (ARTICLE->PART, community child->parent, etc.)
            if e.source not in node_ids or e.target not in node_ids:
                bad_part_of += 1

    return dangling_edges, missing_refs, bad_member_of, bad_part_of
