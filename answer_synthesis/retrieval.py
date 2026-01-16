# phase8_retrieval_enhanced.py
"""
Enhanced retrieval index building for GraphRAG.
Builds a unified index containing:
- Sentences
- Entities (canonical + first alias + first description)
- Entity context windows (40-60 tokens around mentions)
- Provision context windows (multi-sentence windows for Article/Section-like entities)
- Community summaries (all levels)
- Document chunks (if present)
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import re
import numpy as np
from graph_maker.data_corpus import KnowledgeGraph, KGNode
from graph_maker.embedding_cache import get_embeddings_with_cache
from graph_maker.relation_schema import load_relation_schema, relation_edge_types_for_retrieval


@dataclass
class IndexItem:
    """
    Represents a single item in the retrieval index.
    """
    id: str                    # unique identifier (e.g., "sent:doc1:5", "ent:neo4j", "comm_l0:3")
    text: str                  # text to embed and search
    item_type: str             # "SENTENCE", "ENTITY", "ENTITY_CONTEXT", "PROVISION_CONTEXT", "COMMUNITY", "CHUNK"
    metadata: Dict[str, Any]   # additional metadata (doc_id, level, canonical, etc.)


def build_retrieval_index_enhanced(
    graph: KnowledgeGraph,
    context_window_tokens: int = 50,
    include_entity_contexts: bool = True,
    include_provision_contexts: bool = True,
    provision_window_sentences: int = 3,
    chunk_size_words: int = 160,
    chunk_overlap_words: int = 40,
    min_sentence_chars: int = 25,
    min_text_chars: int = 40,
    verbose: bool = True,
) -> Tuple[List[IndexItem], np.ndarray]:
    """
    Build a unified retrieval index from the knowledge graph.
    
    Parameters
    ----------
    graph : KnowledgeGraph
        The knowledge graph containing nodes and edges
    context_window_tokens : int
        Number of tokens (approx) to include around entity mentions
    include_entity_contexts : bool
        Whether to include entity context windows in the index
    include_provision_contexts : bool
        Whether to include provision (Article/Section) multi-sentence context windows in the index
    verbose : bool
        Whether to print progress
    
    Returns
    -------
    index_items : List[IndexItem]
        List of all indexed items with metadata
    embeddings : np.ndarray
        Embedding matrix of shape [len(index_items), embedding_dim]
    """
    index_items: List[IndexItem] = []
    seen_texts: set[str] = set()

    _PROVISION_CANON_RE = re.compile(r"\b(article|articles|section|sections)\s+\d+[a-z]?\b", re.IGNORECASE)

    def _normalize_text(t: str) -> str:
        return " ".join(t.split()).strip().lower()

    def _min_chars_for_item(item: IndexItem) -> int:
        # Legal corpora need very short ENTITY items (e.g., 'article 3', 'article 31a').
        if item.item_type == "ENTITY":
            return 6
        if item.item_type == "RELATION":
            return 8
        if item.item_type == "COMMUNITY_MICRO":
            return max(20, min_text_chars)
        return min_text_chars

    def _maybe_add(item: IndexItem) -> None:
        norm = _normalize_text(item.text)
        if len(norm) < _min_chars_for_item(item):
            return
        # Dedup by text for long items; keep anchor items even if text overlaps.
        if item.item_type not in {"ENTITY", "PROVISION_CONTEXT"}:
            if norm in seen_texts:
                return
            seen_texts.add(norm)
        index_items.append(item)

    def _chunk_text(doc_id: str, text: str) -> List[Tuple[str, str]]:
        words = text.split()
        chunks: List[Tuple[str, str]] = []
        if not words:
            return chunks
        step = max(1, chunk_size_words - chunk_overlap_words)
        for start in range(0, len(words), step):
            end = min(len(words), start + chunk_size_words)
            chunk_words = words[start:end]
            if not chunk_words:
                continue
            chunk_text = " ".join(chunk_words)
            chunk_id = f"chunk:{doc_id}:{start}-{end}"
            chunks.append((chunk_id, chunk_text))
            if end == len(words):
                break
        return chunks
    
    # 0) Index DOCUMENT chunks (overlapping windows)
    chunk_count = 0
    for node_id, node in graph.nodes.items():
        if node.label == "DOCUMENT":
            doc_text = node.properties.get("text", "")
            doc_id = node.properties.get("doc_id") or node_id.replace("doc:", "")
            for chunk_id, chunk_text in _chunk_text(doc_id, doc_text):
                _maybe_add(IndexItem(
                    id=chunk_id,
                    text=chunk_text,
                    item_type="CHUNK",
                    metadata={"doc_id": doc_id}
                ))
                chunk_count += 1

    if verbose:
        print(f"[Index] Added {chunk_count} CHUNK items")

    # 1) Index SENTENCE nodes (filtering short/duplicate sentences)
    sentence_count = 0
    for node_id, node in graph.nodes.items():
        if node.label == "SENTENCE":
            text = node.properties.get("text", "")
            if not text or len(text.strip()) < min_sentence_chars:
                continue
            _maybe_add(IndexItem(
                id=node_id,
                text=text,
                item_type="SENTENCE",
                metadata={
                    "doc_id": node.properties.get("doc_id"),
                    "index": node.properties.get("index")
                }
            ))
            sentence_count += 1

    if verbose:
        print(f"[Index] Added {sentence_count} SENTENCE nodes")
    
    # 2) Index ENTITY nodes (canonical + first alias + first description)
    entity_count = 0
    for node_id, node in graph.nodes.items():
        if node.label == "ENTITY":
            canonical = node.properties.get("canonical", "")
            aliases = node.properties.get("aliases", [])
            descriptions = node.properties.get("descriptions", [])
            
            # Build entity text: "canonical. first_alias. first_description"
            parts = [canonical]
            if aliases and len(aliases) > 0:
                parts.append(aliases[0])
            if descriptions and len(descriptions) > 0:
                parts.append(descriptions[0])
            
            entity_text = ". ".join(str(p) for p in parts if p)
            
            if entity_text:
                _maybe_add(IndexItem(
                    id=node_id,
                    text=entity_text,
                    item_type="ENTITY",
                    metadata={
                        "canonical": canonical,
                        "type": node.properties.get("type"),
                        "sources": node.properties.get("sources", [])
                    }
                ))
                entity_count += 1
    
    if verbose:
        print(f"[Index] Added {entity_count} ENTITY nodes")

    # 2b) Index typed relation edges as text facts (structured evidence)
    relation_count = 0
    NON_INDEXABLE_RELATIONS = {
        "HAS_SENTENCE", "MENTION_IN", "CO_OCCURS_WITH", 
        "COMMUNITY_CONTAINS", "MEMBER_OF", "HAS_CHUNK", "SOURCE_OF"
    }

    for edge in graph.edges:
        if edge.type in NON_INDEXABLE_RELATIONS:
            continue
            
        src = graph.nodes.get(edge.source)
        tgt = graph.nodes.get(edge.target)
        if not src or not tgt:
            continue
            
        # Allow ENTITY or DOMAIN nodes as endpoints for semantic relations.
        valid_labels = {"ENTITY", "DOMAIN"}
        if src.label not in valid_labels or tgt.label not in valid_labels:
            continue
            
        sc = str(src.properties.get("canonical") or edge.source)
        tc = str(tgt.properties.get("canonical") or edge.target)
        sent_id = (edge.properties or {}).get("sentence_id")
        evidence = (edge.properties or {}).get("evidence_snippet", "")
        
        # Fact text: "Source relation Target. Evidence: ..."
        rel_text = f"{sc} {edge.type} {tc}."
        if evidence:
            rel_text += f" Evidence: {evidence}"
            
        _maybe_add(IndexItem(
            id=f"rel:{edge.id}",
            text=rel_text,
            item_type="RELATION",
            metadata={
                "edge_type": edge.type,
                "source": edge.source,
                "target": edge.target,
                "source_canonical": sc,
                "target_canonical": tc,
                "sentence_id": sent_id,
            }
        ))
        relation_count += 1

    if verbose:
        print(f"[Index] Added {relation_count} RELATION items")
    
    # 3) Index ENTITY context windows (40-60 tokens around mentions)
    context_count = 0
    if include_entity_contexts:
        # Find MENTION_IN edges to get entity-sentence associations
        entity_mentions: Dict[str, List[str]] = {}  # entity_id -> [sent_ids]
        for edge in graph.edges:
            if edge.type == "MENTION_IN":
                entity_id = edge.source
                sent_id = edge.target
                entity_mentions.setdefault(entity_id, []).append(sent_id)
        
        # For each entity, create context window from first mentioning sentence
        for entity_id, sent_ids in entity_mentions.items():
            if not sent_ids:
                continue
            
            # Use first mentioning sentence
            sent_id = sent_ids[0]
            sent_node = graph.nodes.get(sent_id)
            if not sent_node:
                continue
            
            sent_text = sent_node.properties.get("text", "")
            if not sent_text:
                continue
            
            # Create context window (approximate tokens by splitting on whitespace)
            words = sent_text.split()
            if len(words) > context_window_tokens:
                # Take middle portion
                start = max(0, (len(words) - context_window_tokens) // 2)
                end = start + context_window_tokens
                context_text = " ".join(words[start:end])
            else:
                context_text = sent_text
            
            entity_node = graph.nodes.get(entity_id)
            canonical = entity_node.properties.get("canonical", "") if entity_node else ""
            
            _maybe_add(IndexItem(
                id=f"{entity_id}_ctx",
                text=context_text,
                item_type="ENTITY_CONTEXT",
                metadata={
                    "entity_id": entity_id,
                    "canonical": canonical,
                    "sent_id": sent_id
                }
            ))
            context_count += 1
    
    if verbose:
        print(f"[Index] Added {context_count} ENTITY_CONTEXT items")

    # 3b) Index PROVISION context windows (multi-sentence windows around legal provision mentions)
    provision_context_count = 0
    if include_provision_contexts:
        # Build quick lookup for sentence nodes by doc_id and index
        doc_sent_by_index: Dict[str, Dict[int, Tuple[str, str]]] = {}
        for node_id, node in graph.nodes.items():
            if node.label != "SENTENCE":
                continue
            doc_id = node.properties.get("doc_id")
            idx = node.properties.get("index")
            text = node.properties.get("text", "")
            # GraphML exports can stringify ints; accept digit strings.
            if isinstance(idx, str) and idx.isdigit():
                try:
                    idx = int(idx)
                except Exception:
                    pass
            if doc_id is None or idx is None or not isinstance(idx, int) or not text:
                continue
            doc_sent_by_index.setdefault(str(doc_id), {})[idx] = (node_id, text)

        # Reuse entity_mentions computed above if available; otherwise compute it here.
        try:
            entity_mentions  # type: ignore[name-defined]
        except NameError:
            entity_mentions = {}
            for edge in graph.edges:
                if edge.type == "MENTION_IN":
                    entity_mentions.setdefault(edge.source, []).append(edge.target)

        for entity_id, sent_ids in entity_mentions.items():
            entity_node = graph.nodes.get(entity_id)
            if not entity_node or entity_node.label != "ENTITY":
                continue

            canonical = str(entity_node.properties.get("canonical", "") or "")
            etype = entity_node.properties.get("type")
            is_provision = (etype == "PROVISION") or bool(_PROVISION_CANON_RE.search(canonical))
            if not is_provision:
                continue

            # Group mentions by document, then build a +/-N sentence window union.
            doc_to_mention_idxs: Dict[str, List[int]] = {}
            for sent_id in sent_ids:
                sent_node = graph.nodes.get(sent_id)
                if not sent_node:
                    continue
                doc_id = sent_node.properties.get("doc_id")
                idx = sent_node.properties.get("index")
                if doc_id is None or idx is None or not isinstance(idx, int):
                    continue
                doc_to_mention_idxs.setdefault(str(doc_id), []).append(idx)

            for doc_id, mention_idxs in doc_to_mention_idxs.items():
                sent_map = doc_sent_by_index.get(doc_id)
                if not sent_map:
                    continue
                window_idxs: set[int] = set()
                for mi in mention_idxs:
                    for j in range(mi - provision_window_sentences, mi + provision_window_sentences + 1):
                        if j in sent_map:
                            window_idxs.add(j)
                if not window_idxs:
                    continue

                ordered = sorted(window_idxs)
                texts = [sent_map[i][1] for i in ordered]
                context_text = " ".join(texts)

                _maybe_add(IndexItem(
                    id=f"{entity_id}::provctx:{doc_id}:{ordered[0]}-{ordered[-1]}",
                    text=context_text,
                    item_type="PROVISION_CONTEXT",
                    metadata={
                        "entity_id": entity_id,
                        "canonical": canonical,
                        "type": etype,
                        "doc_id": doc_id,
                        "mention_sentence_indexes": sorted(set(int(x) for x in mention_idxs)),
                        "window_sentence_indexes": ordered,
                    },
                ))
                provision_context_count += 1

    if verbose:
        print(f"[Index] Added {provision_context_count} PROVISION_CONTEXT items")
    
    # 4) Index COMMUNITY summaries (all levels)
    community_count = 0
    for node_id, node in graph.nodes.items():
        if node.label == "COMMUNITY":
            # Use summary as primary text, fall back to title
            summary = node.properties.get("summary") or node.properties.get("micro_summary")
            micro = node.properties.get("micro_summary")
            bullets = node.properties.get("extractive_bullets", [])
            title = node.properties.get("title")
            text = summary or title or ""

            level = node.properties.get("level")
            coherence = node.properties.get("coherence")
            comm_id = node.properties.get("comm_id")
            members_count = node.properties.get("members_count")

            base_meta = {
                "level": level,
                "comm_id": comm_id,
                "members_count": members_count,
                "title": title,
                "sample_entities": node.properties.get("sample_entities", []),
                "coherence": coherence,
            }

            if text:
                _maybe_add(IndexItem(
                    id=node_id,
                    text=text,
                    item_type="COMMUNITY",
                    metadata=base_meta,
                ))
                community_count += 1

            if micro:
                meta_micro = dict(base_meta)
                meta_micro.update({"base_id": node_id})
                _maybe_add(IndexItem(
                    id=f"{node_id}::micro",
                    text=micro,
                    item_type="COMMUNITY_MICRO",
                    metadata=meta_micro,
                ))

            if bullets:
                joined = " | ".join(str(b) for b in bullets)
                meta_bullets = dict(base_meta)
                meta_bullets.update({"base_id": node_id, "bullet_count": len(bullets)})
                _maybe_add(IndexItem(
                    id=f"{node_id}::bullets",
                    text=joined,
                    item_type="COMMUNITY_BULLETS",
                    metadata=meta_bullets,
                ))
    
    if verbose:
        print(f"[Index] Added {community_count} COMMUNITY nodes")
    
    # 5) Compute embeddings for all items
    if verbose:
        print(f"[Index] Computing embeddings for {len(index_items)} items...")
    
    texts = [item.text for item in index_items]
    embeddings = get_embeddings_with_cache(texts)
    
    if verbose:
        print(f"[Index] Index built with {len(index_items)} items, embedding shape: {embeddings.shape}")
    
    return index_items, embeddings
