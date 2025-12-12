# phase6_communities_hierarchy.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import networkx as nx
import community as community_louvain  # python-louvain
import json
import numpy as np
import os
from ner import get_embeddings_with_cache
import google.generativeai as genai

# Optional: use OpenAI for summaries if you have client object
# from openai import OpenAI  # assumed earlier in pipeline if needed

# Reuse your core types
from data_corpus import KnowledgeGraph, KGNode, KGEdge

# Optional disk cache helper (if you added cache_utils previously)
try:
    from cache_utils import DiskJSONCache
    _COMM_SUMMARY_CACHE = DiskJSONCache("cache_community_summaries.json")
except Exception:
    _COMM_SUMMARY_CACHE = None


@dataclass
class CommunityLevelResult:
    level: int
    partition: Dict[str, int]              # node_id -> community_id
    supergraph: nx.Graph                   # collapsed graph for this level (community nodes)
    community_nodes: Dict[int, List[str]]  # community_id -> list of member node_ids


# -------------------------
# Build entity graph (helper)
# -------------------------
def _build_entity_graph(graph: KnowledgeGraph,
                        edge_types: Optional[List[str]] = None) -> nx.Graph:
    """
    Build an undirected weighted NetworkX graph of ENTITY nodes from the KnowledgeGraph.
    edge_types: list of KGEdge.type to include as links (default uses CO_OCCURS_WITH and semantic edges)
    """
    if edge_types is None:
        edge_types = ["CO_OCCURS_WITH", "USED_WITH", "PROPOSED", "EVALUATED_ON", "DEPLOYS_ON"]

    G = nx.Graph()
    # add entity nodes
    for node_id, node in graph.nodes.items():
        if node.label == "ENTITY":
            G.add_node(node_id)

    # add edges with weight aggregation
    for e in graph.edges:
        if e.type in edge_types and e.source in G.nodes and e.target in G.nodes:
            if G.has_edge(e.source, e.target):
                G[e.source][e.target]["weight"] += float(e.properties.get("weight", 1.0))
            else:
                G.add_edge(e.source, e.target, weight=float(e.properties.get("weight", 1.0)))

    return G


# -------------------------
# Collapse graph helper
# -------------------------
def _collapse_graph_by_partition(G: nx.Graph, partition: Dict[str, int]) -> nx.Graph:
    """
    Collapse G nodes into supernodes based on partition {node: community}.
    Resulting graph nodes are community ids as strings (e.g., 'c:42'), edges aggregated with weights.
    """
    H = nx.Graph()
    # Build mapping community -> members
    comm_members: Dict[int, List[str]] = {}
    for node, comm in partition.items():
        comm_members.setdefault(comm, []).append(node)

    # Add supernodes
    for comm in comm_members:
        super_id = f"comm:{comm}"
        H.add_node(super_id, members=comm_members[comm])

    # Iterate original edges and add/aggregate to supergraph
    for u, v, data in G.edges(data=True):
        cu = partition.get(u)
        cv = partition.get(v)
        if cu is None or cv is None:
            continue
        su = f"comm:{cu}"
        sv = f"comm:{cv}"
        w = float(data.get("weight", 1.0))
        if su == sv:
            # internal edges - optionally track internal weight as node attr
            H.nodes[su].setdefault("internal_weight", 0.0)
            H.nodes[su]["internal_weight"] += w
            continue
        if H.has_edge(su, sv):
            H[su][sv]["weight"] += w
        else:
            H.add_edge(su, sv, weight=w)
    return H


# -------------------------
# Multi-level community detection
# -------------------------
def compute_multilevel_communities(graph: KnowledgeGraph,
                                  max_levels: int = 2,
                                  min_comm_size: int = 1,
                                  edge_types: Optional[List[str]] = None,
                                  verbose: bool = True) -> List[CommunityLevelResult]:
    """
    Compute hierarchical communities.

    Returns a list of CommunityLevelResult objects, where index 0 is level-0 (base entities),
    index 1 is level-1 (communities of communities), etc.
    """
    results: List[CommunityLevelResult] = []

    # Level 0: build entity graph from KG
    G0 = _build_entity_graph(graph, edge_types=edge_types)
    if G0.number_of_nodes() == 0:
        if verbose:
            print("[communities] No ENTITY nodes found in KG.")
        return results

    if verbose:
        print(f"[communities] Entity graph nodes: {G0.number_of_nodes()}, edges: {G0.number_of_edges()}")

    # Level loop
    current_graph = G0
    for level in range(0, max_levels + 1):
        if current_graph.number_of_nodes() == 0:
            if verbose:
                print(f"[communities] Level {level}: graph empty, stopping.")
            break

        # run Louvain (python-louvain)
        partition = community_louvain.best_partition(current_graph, weight="weight")
        # Filter tiny communities if required (optional)
        # Build community -> members mapping
        comm_members: Dict[int, List[str]] = {}
        for node, comm in partition.items():
            comm_members.setdefault(comm, []).append(node)

        # Remove communities smaller than min_comm_size by reassigning them to -1 (or keep)
        if min_comm_size > 1:
            removed = []
            for comm_id, members in list(comm_members.items()):
                if len(members) < min_comm_size:
                    removed.append(comm_id)
            # Mark small communities with unique negative ids to avoid collapsing them alone
            if removed:
                for small in removed:
                    for n in comm_members[small]:
                        partition[n] = -1_000_000 - small  # unique negative cluster id
                # recompute mapping
                comm_members = {}
                for node, comm in partition.items():
                    comm_members.setdefault(comm, []).append(node)

        # collect result for this level
        supergraph = _collapse_graph_by_partition(current_graph, partition)

        # convert supergraph node labels from 'comm:X' to int ids for convenience
        # community_nodes mapping (comm_id_int -> members)
        community_nodes: Dict[int, List[str]] = {}
        for node, comm in partition.items():
            # if partition came from supergraph, comm may be like 'comm:42' -> keep as int if possible
            community_nodes.setdefault(int(comm), []).append(node)

        result = CommunityLevelResult(
            level=level,
            partition=dict(partition),
            supergraph=supergraph,
            community_nodes=community_nodes,
        )
        results.append(result)

        if verbose:
            ncom = len(set(partition.values()))
            print(f"[communities] Level {level}: found {ncom} communities")

        # Prepare next level: current_graph = supergraph with nodes renamed to community ids (ints)
        # Create a relabeled graph where node names are ints (community ids)
        next_graph = nx.Graph()
        for super_n, data in supergraph.nodes(data=True):
            # super_n is like 'comm:42' -> extract int
            try:
                cid = int(str(super_n).split(":")[1])
            except Exception:
                cid = hash(super_n)
            next_graph.add_node(cid, **data)

        for u, v, d in supergraph.edges(data=True):
            try:
                uid = int(str(u).split(":")[1])
                vid = int(str(v).split(":")[1])
            except Exception:
                uid = hash(u)
                vid = hash(v)
            w = float(d.get("weight", 1.0))
            if next_graph.has_edge(uid, vid):
                next_graph[uid][vid]["weight"] += w
            else:
                next_graph.add_edge(uid, vid, weight=w)

        current_graph = next_graph

    return results


# -------------------------
# Community nodes creation & linking into KG
# -------------------------
_SUMMARY_PROMPT = """
You are given a list of canonical entity names and their short descriptions.
Produce:
- a short title (3-6 words)
- a concise 2-4 sentence summary describing what unifies these entities
Return ONLY a JSON object: {"title":"...", "summary":"..."}
"""

def _summarize_community_with_llm(genai_module,
                                  entity_infos: List[Dict[str, Any]],
                                  model_name: str = "gemini-1.5-pro",
                                  temperature: float = 0.3) -> Dict[str, str]:
    """
    Use Google Generative AI (Gemini) to summarize a community.
    - genai_module: the imported google.generativeai module (configured).
    - entity_infos: list of {"canonical":..., "type":..., "descriptions":[...]}
    Returns dict with "title" and "summary".
    """
    if genai_module is None:
        return {"title": None, "summary": None}

    # build canonical key for caching
    keys = [e.get("canonical", "") or "" for e in entity_infos]
    key_raw = "||".join(sorted(keys))
    if _COMM_SUMMARY_CACHE is not None:
        cached = _COMM_SUMMARY_CACHE.get(key_raw)
        if cached:
            return cached

    # Prepare Gemini content parts (system + user)
    contents = [
        {"role": "system", "parts": [{"text": _SUMMARY_PROMPT}]},
        {"role": "user", "parts": [{"text": json.dumps(entity_infos, ensure_ascii=False)}]},
    ]

    # Build model handle and call generate_content
    model = genai_module.GenerativeModel(model_name)
    try:
        response = model.generate_content(
            contents=contents,
            generation_config={"temperature": temperature},
        )
    except Exception as e:
        # On any LLM-call failure, return a safe fallback
        if _COMM_SUMMARY_CACHE is not None:
            _COMM_SUMMARY_CACHE.set(key_raw, {"title": None, "summary": None})
        return {"title": None, "summary": None}

    # Extract text from Gemini response (different SDK versions sometimes return different fields)
    raw = ""
    if hasattr(response, "text") and response.text:
        raw = response.text
    else:
        # fallback to candidates list if present
        try:
            raw = response.candidates[0].content
        except Exception:
            raw = ""

    raw = raw.strip()

    # sanitize fences and leading "json" markers
    if raw.startswith("```"):
        # remove triple-fence block
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        obj = json.loads(raw)
        # ensure keys exist
        title = obj.get("title")
        summary = obj.get("summary")
        out = {"title": title, "summary": summary}
    except Exception:
        # fallback: minimal attempt to salvage text
        out = {"title": None, "summary": raw or None}

    if _COMM_SUMMARY_CACHE is not None:
        _COMM_SUMMARY_CACHE.set(key_raw, out)
    return out


def build_and_add_community_nodes(graph: KnowledgeGraph,
                                  community_results: List[CommunityLevelResult],
                                  
                                  create_member_edges: bool = True,
                                  create_partof_edges: bool = True,
                                  include_entity_sample: int = 6,
                                  verbose: bool = True):
    """
    Take the computed community results and add COMMUNITY nodes & edges into the KnowledgeGraph.

    - community_results: output of compute_multilevel_communities (list indexed by level)
    - client: optional OpenAI client for summaries (if None, summaries will be absent)
    """
    # Map: (level, comm_id) -> community_node_id string
    comm_node_map: Dict[Tuple[int, int], str] = {}

    # First pass: create community nodes for each level
    for res in community_results:
        level = res.level
        # community nodes in res.community_nodes: {comm_id: [member_node_ids]}
        for comm_id, members in res.community_nodes.items():
            comm_node_id = f"comm_l{level}:{comm_id}"
            # prepare props
            # sample entities for quick summary
            sample_entities = members[:include_entity_sample]
            # fetch descriptions/types from KG nodes if present
            ent_infos = []
            for n in sample_entities:
                node = graph.nodes.get(n)
                ent_infos.append({
                    "canonical": node.properties.get("canonical") if node else n,
                    "type": node.properties.get("type") if node else None,
                    "descriptions": node.properties.get("descriptions", []) if node else []
                })
            # get summary via LLM if client provided
            summary_obj = _summarize_community_with_llm(genai, ent_infos) if genai is not None else {"title": None, "summary": None}

            graph.add_node(
                KGNode(
                    id=comm_node_id,
                    label="COMMUNITY",
                    properties={
                        "level": level,
                        "comm_id": comm_id,
                        "members_count": len(members),
                        "title": summary_obj.get("title"),
                        "summary": summary_obj.get("summary"),
                        "sample_entities": sample_entities,
                    },
                )
            )
            comm_node_map[(level, comm_id)] = comm_node_id
            if verbose:
                print(f"[communities] Added COMMUNITY node {comm_node_id} (members={len(members)})")

    # Second pass: add MEMBER_OF edges from entity -> comm_l0 and PART_OF between community levels
    edge_counter = len(graph.edges)
    # MEMBER_OF (ENTITY -> COMMUNITY L0)
    if create_member_edges:
        # level 0 expected at community_results[0]
        if len(community_results) >= 1:
            level0 = community_results[0]
            for comm_id, members in level0.community_nodes.items():
                comm_node_id = comm_node_map.get((0, comm_id))
                for member_node in members:
                    # add MEMBER_OF edge
                    edge_id = f"e:comm_member:{edge_counter}"; edge_counter += 1
                    graph.add_edge(
                        KGEdge(
                            id=edge_id,
                            source=member_node,
                            target=comm_node_id,
                            type="MEMBER_OF",
                            properties={"level": 0, "comm_id": comm_id}
                        )
                    )

    # PART_OF edges between community levels: (COMM_Lk) -[:PART_OF]-> (COMM_Lk+1)
    if create_partof_edges:
        for lvl in range(0, len(community_results) - 1):
            lower = community_results[lvl]
            upper = community_results[lvl + 1]
            # mapping from lower community id -> upper community id:
            # For each member of lower.community_nodes, look up its partition in upper.partition
            for lower_cid, lower_members in lower.community_nodes.items():
                # pick any member node and look up which upper community it belongs to (via upper.partition)
                mapped_upper_cids = set()
                for member in lower_members:
                    upcid = upper.partition.get(member)
                    if upcid is not None:
                        mapped_upper_cids.add(upcid)
                # create PART_OF edges for each mapping
                for upcid in mapped_upper_cids:
                    lower_node_id = comm_node_map.get((lvl, lower_cid))
                    upper_node_id = comm_node_map.get((lvl + 1, upcid))
                    if lower_node_id and upper_node_id:
                        edge_id = f"e:comm_partof:{edge_counter}"; edge_counter += 1
                        graph.add_edge(
                            KGEdge(
                                id=edge_id,
                                source=lower_node_id,
                                target=upper_node_id,
                                type="PART_OF",
                                properties={"from_level": lvl, "to_level": lvl + 1}
                            )
                        )

    # Finally, add MEMBER_OF edges correctly using KGEdge
    # Recompute edge_counter and add MEMBER_OF edges using KGEdge
    # (We previously added nothing for MEMBER_OF; now add using KGEdge)
    
    edge_counter = len(graph.edges)
    if create_member_edges and len(community_results) >= 1:
        level0 = community_results[0]
        for comm_id, members in level0.community_nodes.items():
            comm_node_id = comm_node_map.get((0, comm_id))
            for member_node in members:
                edge_id = f"e:comm_member:{edge_counter}"; edge_counter += 1
                graph.add_edge(
                    KGEdge(
                        id=edge_id,
                        source=member_node,
                        target=comm_node_id,
                        type="MEMBER_OF",
                        properties={"level": 0, "comm_id": comm_id},
                    )
                )

    if verbose:
        print(f"[communities] Finished adding community nodes and edges. Total nodes: {len(graph.nodes)}, edges: {len(graph.edges)}")

    return comm_node_map


def build_retrieval_index(graph: KnowledgeGraph):
    """Precompute embeddings for SENTENCE and COMMUNITY nodes."""
    items = []  # (node_id, text)
    for node in graph.nodes.values():
        if node.label == "SENTENCE":
            items.append((node.id, node.properties["text"]))
        elif node.label == "COMMUNITY":
            # use summary as text
            txt = node.properties.get("summary") or node.properties.get("title")
            if txt:
                items.append((node.id, txt))

    texts = [t for _, t in items]
    embeddings = get_embeddings_with_cache(texts)
    ids = [i for i, _ in items]
    return ids, embeddings


def search_relevant_nodes(
    query: str,
    ids: List[str],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    q_emb = get_embeddings_with_cache([query])[0]

    sims = embeddings @ q_emb / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-9)
    idxs = np.argsort(-sims)[:top_k]
    return [(ids[i], float(sims[i])) for i in idxs]
