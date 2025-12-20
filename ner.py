# ai_setup.py

import os
from dataclasses import dataclass, field
import logging
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Set, Tuple
import spacy
from sklearn.cluster import AgglomerativeClustering
import json
import re
import numpy as np
from data_corpus import Entity, KGEdge, KnowledgeGraph, KGNode ,SentenceInfo
import google.generativeai as genai
from cache_utils import DiskJSONCache
from itertools import combinations

nlp = spacy.load("en_core_web_sm")


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Small, fast embedding model
_EMB_MODEL: SentenceTransformer | None = None
_EMB_CACHE = DiskJSONCache("cache_embeddings.json")


def get_emb_model() -> SentenceTransformer:
    global _EMB_MODEL
    if _EMB_MODEL is None:
        _EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMB_MODEL


def _normalize_surface(surface: str) -> str:
    doc = nlp(surface)
    lemmas = [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]
    base = " ".join(lemmas) or surface.lower()
    base = re.sub(r"[^a-z0-9\s]", " ", base)
    base = re.sub(r"\s+", " ", base)
    return base.strip()


def _find_acronym_pairs(text: str) -> Dict[str, str]:
    """Return mapping of acronym -> longform for patterns like 'Long Form (LF)'"""
    pairs: Dict[str, str] = {}
    pattern = re.compile(r"([A-Za-z][A-Za-z\s]{3,})\s*\(([A-Z]{2,})\)")
    for match in pattern.finditer(text):
        long_form = match.group(1).strip()
        acronym = match.group(2).strip()
        pairs[acronym] = long_form
    return pairs


# -------------------------
# Utilities
# -------------------------
def _strip_code_fences(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return s


def _parse_json_safely(raw: str, default):
    """Best-effort JSON parsing that tolerates fenced blocks and extra text.

    - Strips triple backtick fences and leading 'json'
    - Tries full parse
    - Then tries to extract first top-level JSON object or array
    - Returns `default` on failure
    """
    try:
        cleaned = _strip_code_fences(raw)
        if not cleaned:
            return default
        # First attempt: direct parse
        return json.loads(cleaned)
    except Exception:
        pass

    try:
        cleaned = _strip_code_fences(raw)
        # Try to find a JSON object
        obj_start = cleaned.find("{")
        obj_end = cleaned.rfind("}")
        arr_start = cleaned.find("[")
        arr_end = cleaned.rfind("]")

        candidates = []
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            candidates.append(cleaned[obj_start:obj_end+1])
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            candidates.append(cleaned[arr_start:arr_end+1])

        for cand in candidates:
            try:
                return json.loads(cand)
            except Exception:
                continue
    except Exception:
        pass

    return default



@dataclass
class EntityCatalogEntry:
    canonical: str
    label: str
    aliases: Set[str] = field(default_factory=set)
    descriptions: Set[str] = field(default_factory=set)
    sources: Set[str] = field(default_factory=set)

def get_embeddings_with_cache(texts: List[str]) -> np.ndarray:
    """
    texts -> matrix [len(texts), dim], caching each text's embedding as a list of floats.
    """
    model = get_emb_model()
    cached_vectors: List[Tuple[int, List[float]]] = []
    missing_indices: List[int] = []
    missing_texts: List[str] = []

    # 1) Check cache
    for idx, t in enumerate(texts):
        cached = _EMB_CACHE.get(t)
        if cached is not None:
            cached_vectors.append((idx, cached))
        else:
            missing_indices.append(idx)
            missing_texts.append(t)

    # 2) Compute embeddings for missing texts (if any)
    if missing_texts:
        new_embs = model.encode(missing_texts, convert_to_numpy=True)
        for i, vec in enumerate(new_embs):
            idx = missing_indices[i]
            vec_list = vec.astype(float).tolist()
            _EMB_CACHE.set(texts[idx], vec_list)
            cached_vectors.append((idx, vec_list))

    # 3) Assemble into final numpy array in correct order
    # We know the dimension from any cached vector or new emb
    dim = len(cached_vectors[0][1]) if cached_vectors else 0
    mat = np.zeros((len(texts), dim), dtype=float)

    for idx, vec_list in cached_vectors:
        mat[idx, :] = np.array(vec_list, dtype=float)

    return mat

def extract_candidate_phrases(text: str) -> List[str]:
    doc = nlp(text)
    candidates: Dict[str, str] = {}

    acronym_map = _find_acronym_pairs(text)

    def _try_add(raw: str):
        cleaned = raw.strip()
        if not cleaned or len(cleaned) > 80 or len(cleaned) < 2:
            return
        norm = _normalize_surface(cleaned)
        if not norm:
            return
        candidates.setdefault(norm, cleaned)

    # Noun chunks
    for chunk in doc.noun_chunks:
        _try_add(chunk.text)

    # Named entities
    for ent in doc.ents:
        _try_add(ent.text)

    # Acronym expansions
    for acro, long_form in acronym_map.items():
        _try_add(acro)
        _try_add(long_form)

    return sorted(set(candidates.values()))


CLUSTER_LABEL_PROMPT = """
You are helping build a knowledge graph for technical text.

You will receive a cluster of surface forms (strings) that likely refer to related concepts or entities.

Task:
- Decide if this cluster corresponds to a meaningful entity/concept.
- If yes, output:
  - "is_entity": true
  - "label": semantic type (e.g., PERSON, PROJECT, FRAMEWORK, MODEL, VECTOR_DB,
    CONFERENCE, ORG, LIBRARY, DATASET, TASK, METRIC, etc.)
  - "canonical": a normalized identifier (snake_case or lowercase words)
  - "description": 1–3 sentences summarizing the concept.
- If not an entity, set "is_entity": false and leave other fields empty or null.

Return ONLY a single JSON object like:
{
  "is_entity": true,
  "label": "FRAMEWORK",
  "canonical": "autographrag",
  "description": "..."
}
"""


def label_cluster_with_llm(cluster_items: List[str]) -> Dict:
    """Label a surface-form cluster using Gemini.

    Design choice: be *permissive* so we don't under-generate entities.
    If the LLM call fails or returns malformed JSON, we treat the cluster
    as an entity by default (is_entity=True).
    """

    prompt = f"{CLUSTER_LABEL_PROMPT}\n\nCluster items:\n{json.dumps(cluster_items, ensure_ascii=False)}"

    raw = ""
    try:
        # Use gemini-2.0-flash which is the latest available model
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.1},
        )
        raw = getattr(response, "text", "") or ""
    except Exception as exc:  # pragma: no cover - network / API issues
        logging.warning("label_cluster_with_llm: LLM call failed: %s", exc)
        raw = ""

    # If parsing fails, default to treating the cluster as an entity
    data = _parse_json_safely(raw, default={"is_entity": True})
    if not isinstance(data, dict):
        data = {"is_entity": True}

    # Be generous: unless explicitly false, accept as entity
    data.setdefault("is_entity", True)
    if data["is_entity"] is False:
        return {"is_entity": False}

    # Ensure we always have a coarse label and canonical
    label = data.get("label") or "ENTITY"
    canon = data.get("canonical") or (cluster_items[0] if cluster_items else "")
    canon = _normalize_surface(canon)

    data["label"] = label
    data["canonical"] = canon
    return data


def cluster_candidates(candidates: List[str], n_clusters: int = 5) -> Dict[int, List[str]]:
    if not candidates:
        return {}

    embeddings = get_embeddings_with_cache(candidates)
    # Use more clusters for larger candidate sets to avoid over-merging
    desired_clusters = min(max(len(candidates) // 2, n_clusters), len(candidates))
    clustering = AgglomerativeClustering(
        n_clusters=desired_clusters
    )
    labels = clustering.fit_predict(embeddings)

    clusters: Dict[int, List[str]] = {}
    for cand, lbl in zip(candidates, labels):
        clusters.setdefault(int(lbl), []).append(cand)
    return clusters


def find_span(text: str, surface: str) -> Tuple[int, int]:
    """Naive span search – first occurrence (could be improved)."""
    idx = text.lower().find(surface.lower())
    if idx == -1:
        return -1, -1
    return idx, idx + len(surface)


def extract_semantic_entities_for_doc(doc_id: str, text: str) -> List[Entity]:
    """
    Fully automated:
    - extracts candidates
    - clusters them
    - labels clusters via LLM
    - creates Entity objects aligned to text
    """
    candidates = extract_candidate_phrases(text)
    clusters = cluster_candidates(candidates, n_clusters=6)
    acronym_map = _find_acronym_pairs(text)

    all_entities: List[Entity] = []

    # Helper to filter out obviously junk clusters (stopwords/numerics/one-char)
    def _cluster_is_junk(items: List[str]) -> bool:
        for s in items:
            cleaned = _normalize_surface(s)
            if not cleaned:
                continue
            # Skip purely numeric or single-character tokens
            if cleaned.isdigit() or len(cleaned) <= 1:
                continue
            # Has some usable content
            return False
        return True

    for cluster_id, items in clusters.items():
        if _cluster_is_junk(items):
            continue

        meta = label_cluster_with_llm(items)

        is_entity = bool(meta.get("is_entity", True))
        base_canonical = meta.get("canonical") or (items[0] if items else "")
        base_canonical = _normalize_surface(acronym_map.get(base_canonical, base_canonical))
        base_label = meta.get("label") or "ENTITY"
        desc = meta.get("description")

        # If LLM says not an entity, still keep it as a DOMAIN node
        if not is_entity:
            base_label = "DOMAIN"

        for surface in items:
            start, end = find_span(text, surface)
            if start == -1:
                continue
            all_entities.append(
                Entity(
                    text=text[start:end],
                    label=base_label,
                    start=start,
                    end=end,
                    source="embed_cluster_llm",
                    canonical=base_canonical,
                    description=desc,
                    context=text[max(0, start-80):min(len(text), end+80)],
                    doc_id=doc_id,
                )
            )

    # --- Ensure minimum graph density per document ---
    MIN_ENTITIES_PER_DOC = 10
    if len(all_entities) < MIN_ENTITIES_PER_DOC:
        doc = nlp(text)
        freq: Dict[str, int] = {}
        for chunk in doc.noun_chunks:
            norm = _normalize_surface(chunk.text)
            if not norm or norm.isdigit() or len(norm) <= 1:
                continue
            freq[norm] = freq.get(norm, 0) + 1

        existing_canons = { _normalize_surface(e.canonical or e.text) for e in all_entities }

        # Sort candidates by frequency descending
        for norm, _count in sorted(freq.items(), key=lambda kv: kv[1], reverse=True):
            if len(all_entities) >= MIN_ENTITIES_PER_DOC:
                break
            if norm in existing_canons:
                continue

            surface = norm
            start, end = find_span(text, surface)
            if start == -1:
                continue
            all_entities.append(
                Entity(
                    text=text[start:end],
                    label="DOMAIN",
                    start=start,
                    end=end,
                    source="auto_domain_promotion",
                    canonical=norm,
                    description=None,
                    context=text[max(0, start-80):min(len(text), end+80)],
                    doc_id=doc_id,
                )
            )

    return all_entities





def build_entity_catalog(
    all_entities_per_doc: Dict[str, List[Entity]]
) -> Dict[str, EntityCatalogEntry]:
    catalog: Dict[str, EntityCatalogEntry] = {}

    for doc_id, ents in all_entities_per_doc.items():
        for e in ents:
            canon = e.canonical or _normalize_surface(e.text)
            if canon not in catalog:
                catalog[canon] = EntityCatalogEntry(
                    canonical=canon,
                    label=e.label,
                )
            entry = catalog[canon]
            entry.aliases.add(e.text)
            if e.description:
                entry.descriptions.add(e.description)
            entry.sources.add(doc_id)

    return catalog


def merge_similar_catalog_entries(catalog: Dict[str, EntityCatalogEntry], similarity_threshold: float = 0.9) -> Dict[str, EntityCatalogEntry]:
    """Merge catalog entries that are semantically close to reduce fragmentation."""
    if not catalog:
        return catalog

    names = list(catalog.keys())
    embeddings = get_embeddings_with_cache(names)
    merged: Dict[str, EntityCatalogEntry] = {}
    used: Set[int] = set()

    for i, name in enumerate(names):
        if i in used:
            continue
        base_entry = catalog[name]
        base_vec = embeddings[i]
        merged_aliases = set(base_entry.aliases)
        merged_desc = set(base_entry.descriptions)
        merged_sources = set(base_entry.sources)
        merged_label = base_entry.label

        for j in range(i + 1, len(names)):
            if j in used:
                continue
            sim = float(np.dot(base_vec, embeddings[j]) / ((np.linalg.norm(base_vec) + 1e-9) * (np.linalg.norm(embeddings[j]) + 1e-9)))
            if sim >= similarity_threshold:
                other = catalog[names[j]]
                merged_aliases |= other.aliases
                merged_desc |= other.descriptions
                merged_sources |= other.sources
                used.add(j)

        merged[name] = EntityCatalogEntry(
            canonical=name,
            label=merged_label,
            aliases=merged_aliases,
            descriptions=merged_desc,
            sources=merged_sources,
        )

    return merged


def add_entity_nodes(graph: KnowledgeGraph, catalog: Dict[str, EntityCatalogEntry]):
    for canon, entry in catalog.items():
        node_id = f"ent:{canon}"
        node_label = "DOMAIN" if entry.label == "DOMAIN" else "ENTITY"
        properties = {
            "canonical": entry.canonical,
            "type": entry.label,
            "aliases": sorted(entry.aliases),
            "descriptions": list(entry.descriptions),
            "sources": list(entry.sources),
        }
        # Mark confidence lower for DOMAIN nodes that often come from promotion or non-entity clusters
        if node_label == "DOMAIN":
            properties["confidence"] = "low"
        else:
            properties["confidence"] = "high"

        graph.add_node(
            KGNode(
                id=node_id,
                label=node_label,
                properties=properties,
            )
        )


def find_sentences_for_entity(
    entity: Entity,
    sent_index: Dict[str, SentenceInfo],
) -> List[str]:
    sids = []
    for sid, s in sent_index.items():
        if (
            (entity.start >= s.start_char and entity.start < s.end_char)
            or (entity.end > s.start_char and entity.end <= s.end_char)
        ):
            sids.append(sid)
    return sids


def add_mention_and_cooccurrence_edges(
    graph: KnowledgeGraph,
    all_entities_per_doc: Dict[str, List[Entity]],
    sent_index: Dict[str, SentenceInfo],
):
    edge_counter = 0
    entities_per_sentence: Dict[str, List[Tuple[Entity, str]]] = {}

    # MENTION_IN edges
    for doc_id, ents in all_entities_per_doc.items():
        for e in ents:
            canon = (e.canonical or e.text.lower().strip())
            ent_node_id = f"ent:{canon}"

            sent_ids = find_sentences_for_entity(e, sent_index)
            for sid in sent_ids:
                edge_id = f"e:{edge_counter}"; edge_counter += 1
                graph.add_edge(
                    KGEdge(
                        id=edge_id,
                        source=ent_node_id,
                        target=sid,
                        type="MENTION_IN",
                        properties={
                            "surface": e.text,
                            "doc_id": doc_id,
                            "char_start": e.start,
                            "char_end": e.end,
                        },
                    )
                )
                entities_per_sentence.setdefault(sid, []).append((e, ent_node_id))

    # CO_OCCURS_WITH edges
    for sid, ent_list in entities_per_sentence.items():
        n = len(ent_list)
        for i in range(n):
            for j in range(i + 1, n):
                _, node1 = ent_list[i]
                _, node2 = ent_list[j]
                if node1 == node2:
                    continue
                edge_id = f"e:{edge_counter}"; edge_counter += 1
                graph.add_edge(
                    KGEdge(
                        id=edge_id,
                        source=node1,
                        target=node2,
                        type="CO_OCCURS_WITH",
                        properties={"sentence_id": sid},
                    )
                )


REL_EXTRACT_SYSTEM_PROMPT = """
You are an expert in relation extraction for knowledge graphs.

You will receive:
- A sentence of technical text.
- A list of entities in that sentence, with their canonical names and types.

Task:
- Extract semantic relations between these entities, as (head, relation, tail).
- Use short, uppercase relation labels like: USED_WITH, DEPLOYS_ON,
  PROPOSED, INTRODUCED, LOCATED_IN, PRESENTED_AT, PUBLISHED_AT, EVALUATED_ON.
- Only output relations where you're reasonably confident.
- 'head' and 'tail' must be canonical names exactly as given.

Return ONLY a JSON list like:
[
  {"head": "autographrag", "relation": "USED_WITH", "tail": "neo4j"},
  ...
]
"""


def extract_relations_for_sentence(
    sentence: SentenceInfo,
    entities_in_sentence: List[Tuple[Entity, str]]  # (entity, ent_node_id)
) -> List[Dict]:
    ent_payload = []
    for e, ent_node_id in entities_in_sentence:
        canon = (e.canonical or e.text.lower().strip())
        ent_payload.append({
            "canonical": canon,
            "type": e.label,
            "surface": e.text,
        })

    user_payload = {
        "sentence": sentence.text,
        "entities": ent_payload,
    }

    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = f"{REL_EXTRACT_SYSTEM_PROMPT}\n\nExtract relations from:\n{json.dumps(user_payload, ensure_ascii=False)}"

    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.1}
        )
        raw = getattr(response, "text", "") or ""
    except Exception as exc:
        logging.warning("extract_relations_for_sentence: LLM call failed: %s", exc)
        raw = ""

    data = _parse_json_safely(raw, default=[])
    if not isinstance(data, list):
        return []
    # Optionally filter to ensure expected keys
    cleaned: List[Dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if all(k in item for k in ("head", "relation", "tail")):
            cleaned.append(item)
    return cleaned

def add_semantic_relation_edges(
    graph: KnowledgeGraph,
    all_entities_per_doc: Dict[str, List[Entity]],
    sent_index: Dict[str, SentenceInfo],
):
    # Build per-sentence entity lists
    entities_per_sentence: Dict[str, List[Tuple[Entity, str]]] = {}
    for doc_id, ents in all_entities_per_doc.items():
        for e in ents:
            canon = (e.canonical or e.text.lower().strip())
            ent_node_id = f"ent:{canon}"
            sids = find_sentences_for_entity(e, sent_index)
            for sid in sids:
                entities_per_sentence.setdefault(sid, []).append((e, ent_node_id))

    edge_counter = len(graph.edges)

    # For each sentence with ≥ 2 entities, extract relations
    for sid, ent_list in entities_per_sentence.items():
        if len(ent_list) < 2:
            continue

        sentence = sent_index[sid]
        rels = extract_relations_for_sentence(sentence, ent_list)

        for rel in rels:
            head_canon = rel["head"]
            tail_canon = rel["tail"]
            rel_type = rel["relation"]

            head_node_id = f"ent:{head_canon}"
            tail_node_id = f"ent:{tail_canon}"

            if head_node_id not in graph.nodes or tail_node_id not in graph.nodes:
                continue

            edge_id = f"e:{edge_counter}"; edge_counter += 1
            graph.add_edge(
                KGEdge(
                    id=edge_id,
                    source=head_node_id,
                    target=tail_node_id,
                    type=rel_type,
                    properties={"sentence_id": sid},
                )
            )