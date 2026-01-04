from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


DEFAULT_SCHEMA_PATH = "relation_schema.json"


def load_relation_schema() -> Dict[str, Any]:
    """Load relation schema/ontology.

    Precedence (highest wins):
    1) `KG_RELATION_SCHEMA_JSON` (JSON object string)
    2) File path in `KG_RELATION_SCHEMA_PATH` (default: relation_schema.json)

    The schema is used to:
    - canonicalize relation labels via `synonyms`
    - optionally constrain allowed relation labels via `allowed_types`
    """

    schema: Dict[str, Any] = {}

    path = os.getenv("KG_RELATION_SCHEMA_PATH", "").strip() or DEFAULT_SCHEMA_PATH
    try:
        p = Path(path)
        if p.exists() and p.is_file():
            schema = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        schema = {}

    raw = os.getenv("KG_RELATION_SCHEMA_JSON", "").strip()
    if raw:
        try:
            override = json.loads(raw)
            if isinstance(override, dict):
                schema = {**schema, **override}
        except Exception:
            pass

    # Normalize expected fields
    synonyms = schema.get("synonyms")
    if not isinstance(synonyms, dict):
        schema["synonyms"] = {}

    allowed_types = schema.get("allowed_types")
    if isinstance(allowed_types, list):
        schema["allowed_types"] = [str(x).strip().upper() for x in allowed_types if str(x).strip()]
    elif allowed_types is None:
        schema["allowed_types"] = []
    else:
        schema["allowed_types"] = []

    # Ensure RELATED_TO exists if constraints are enabled
    if schema["allowed_types"] and "RELATED_TO" not in set(schema["allowed_types"]):
        schema["allowed_types"].append("RELATED_TO")

    # Defaults
    schema.setdefault("max_label_len", 48)
    schema.setdefault("min_confidence", 0.35)
    schema.setdefault("allowed_rel_regex", r"^[A-Z][A-Z0-9_]{1,47}$")
    schema.setdefault("proposal_path", "edge_type_proposals.jsonl")

    return schema


def get_allowed_relation_types(schema: Dict[str, Any]) -> Set[str]:
    allowed = schema.get("allowed_types")
    if not isinstance(allowed, list):
        return set()
    return {str(x).strip().upper() for x in allowed if str(x).strip()}


def community_edge_types_from_schema(schema: Dict[str, Any]) -> List[str]:
    """Edge types to use when building the entity-only community graph."""
    rel_types = sorted(get_allowed_relation_types(schema))
    # CO_OCCURS_WITH is not a typed semantic relation but is a core connectivity signal.
    out = ["CO_OCCURS_WITH"]
    out.extend([t for t in rel_types if t != "CO_OCCURS_WITH"])
    return out


def relation_edge_types_for_retrieval(schema: Dict[str, Any]) -> Set[str]:
    """Relation edge types to index/expand over during retrieval."""
    return get_allowed_relation_types(schema)
