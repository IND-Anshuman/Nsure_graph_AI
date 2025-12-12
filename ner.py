# ai_setup.py

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Set, Tuple
import spacy
from sklearn.cluster import AgglomerativeClustering
import json
import numpy as np
from data_corpus import Entity, KGEdge, KnowledgeGraph, KGNode ,SentenceInfo
import google.generativeai as genai
from cache_utils import DiskJSONCache

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
    candidates: Set[str] = set()

    # Noun chunks
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        if 2 <= len(phrase) <= 60:
            candidates.add(phrase)

    # Named entities
    for ent in doc.ents:
        candidates.add(ent.text.strip())

    return sorted(candidates)


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
    # Use gemini-2.0-flash which is the latest available model
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = f"{CLUSTER_LABEL_PROMPT}\n\nCluster items:\n{json.dumps(cluster_items, ensure_ascii=False)}"
    
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.1}
    )
    
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    data = json.loads(raw)
    return data


def cluster_candidates(candidates: List[str], n_clusters: int = 5) -> Dict[int, List[str]]:
    if not candidates:
        return {}

    embeddings = get_embeddings_with_cache(candidates)
    clustering = AgglomerativeClustering(
        n_clusters=min(n_clusters, len(candidates))
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

    all_entities: List[Entity] = []

    for cluster_id, items in clusters.items():
        meta = label_cluster_with_llm(items)
        if not meta.get("is_entity", False):
            continue

        canonical = meta.get("canonical") or items[0].lower().replace(" ", "_")
        label = meta.get("label") or "ENTITY"
        desc = meta.get("description")

        for surface in items:
            start, end = find_span(text, surface)
            if start == -1:
                continue
            all_entities.append(
                Entity(
                    text=text[start:end],
                    label=label,
                    start=start,
                    end=end,
                    source="embed_cluster_llm",
                    canonical=canonical,
                    description=desc,
                    context=text[max(0, start-50):min(len(text), end+50)],
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
            canon = (e.canonical or e.text.lower().strip())
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


def add_entity_nodes(graph: KnowledgeGraph, catalog: Dict[str, EntityCatalogEntry]):
    for canon, entry in catalog.items():
        node_id = f"ent:{canon}"
        graph.add_node(
            KGNode(
                id=node_id,
                label="ENTITY",
                properties={
                    "canonical": entry.canonical,
                    "type": entry.label,
                    "aliases": sorted(entry.aliases),
                    "descriptions": list(entry.descriptions),
                    "sources": list(entry.sources),
                },
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
                        properties={"surface": e.text, "doc_id": doc_id},
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

    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.1}
    )

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    data = json.loads(raw)
    return data

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