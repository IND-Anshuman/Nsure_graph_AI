# phase6_communities_hierarchy.py
from __future__ import annotations
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
import networkx as nx
import community as community_louvain  # python-louvain
import json
import numpy as np
from embedding_cache import get_embeddings_with_cache

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    genai = None  # type: ignore

# Optional: use OpenAI for summaries if you have client object
# from openai import OpenAI  # assumed earlier in pipeline if needed

# Reuse your core types
from data_corpus import KnowledgeGraph, KGNode, KGEdge, SentenceInfo

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
        edge_types = [
            "CO_OCCURS_WITH",
            "USED_WITH",
            "PROPOSED",
            "EVALUATED_ON",
            "DEPLOYS_ON",
            # typed semantic relations from extract_typed_semantic_relations
            "APPLIED_IN",
            "ENABLES",
            "IMPROVES",
            "CAUSES",
            "SUBDOMAIN_OF",
        ]

    G = nx.Graph()
    # add entity and domain nodes
    for node_id, node in graph.nodes.items():
        if node.label in {"ENTITY", "DOMAIN"}:
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

    # Level loop with stability guards
    current_graph = G0
    previous_ncom = None
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

        # Always compute ncom; it's used by stability guards below.
        ncom = len(set(partition.values()))
        if verbose:
            print(f"[communities] Level {level}: found {ncom} communities")

        # Stability guards: stop if trivial or no structural gain
        # - fewer than 4 nodes
        if current_graph.number_of_nodes() < 4:
            if verbose:
                print(f"[communities] Stopping: graph has fewer than 4 nodes ({current_graph.number_of_nodes()})")
            break
        # - only one community
        if ncom <= 1:
            if verbose:
                print(f"[communities] Stopping: only one community detected at level {level}")
            break
        # - number of communities equals number of nodes (each node its own community)
        if ncom >= current_graph.number_of_nodes():
            if verbose:
                print(f"[communities] Stopping: communities ({ncom}) == nodes ({current_graph.number_of_nodes()})")
            break
        # - no meaningful structural gain compared to previous level
        if previous_ncom is not None and previous_ncom == ncom:
            if verbose:
                print(f"[communities] Stopping: no meaningful gain (ncom unchanged from previous level)")
            break
        previous_ncom = ncom

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
You will receive a list of entities (canonical, type, descriptions).

Return strictly JSON with:
{
    "title": "3-6 word title",
    "micro_summary": "<=60 tokens, noun-phrase start, no filler",
    "extractive_bullets": ["3-5 bullet-like fact sentences, terse"]
}

Rules:
- Be concise and factual; no meta-commentary.
- Prefer extractive wording grounded in the entities/descriptions.
- Do not repeat the title text inside the bullets.
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

    # build canonical key for caching (coerce everything to string)
    keys = [str(e.get("canonical", "") or "") for e in entity_infos]
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
    except Exception:
        if _COMM_SUMMARY_CACHE is not None:
            _COMM_SUMMARY_CACHE.set(key_raw, {"title": None, "micro_summary": None, "extractive_bullets": []})
        return {"title": None, "micro_summary": None, "extractive_bullets": []}

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
        out = {
            "title": obj.get("title"),
            "micro_summary": obj.get("micro_summary") or obj.get("summary"),
            "extractive_bullets": obj.get("extractive_bullets") or [],
        }
    except Exception:
        # fallback: minimal attempt to salvage text
        out = {"title": None, "micro_summary": raw or None, "extractive_bullets": []}

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

    # Precompute centrality on the entity graph so we can rank members by importance
    entity_G = _build_entity_graph(graph)
    centrality_scores = {}
    try:
        centrality_scores = nx.betweenness_centrality(entity_G, normalized=True)
    except Exception:
        centrality_scores = {n: 0.0 for n in entity_G.nodes}

    # First pass: create community nodes for each level
    for res in community_results:
        level = res.level
        # community nodes in res.community_nodes: {comm_id: [member_node_ids]}
        for comm_id, members in res.community_nodes.items():
            comm_node_id = f"comm_l{level}:{comm_id}"
            # prepare props
            # pick most central members for summary (centrality-aware sampling)
            ranked_members = sorted(members, key=lambda n: centrality_scores.get(n, 0.0), reverse=True)
            sample_entities = ranked_members[:include_entity_sample]
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
            summary_obj = _summarize_community_with_llm(genai, ent_infos) if genai is not None else {"title": None, "micro_summary": None, "extractive_bullets": []}

            # Lightweight fallback summaries if LLM returns empty
            if not summary_obj.get("micro_summary"):
                sample_labels = [f"{info.get('canonical')} ({info.get('type') or 'entity'})" for info in ent_infos]
                summary_obj["micro_summary"] = ", ".join(sample_labels)[:220] or None
            if not summary_obj.get("extractive_bullets"):
                summary_obj["extractive_bullets"] = [info.get("canonical") for info in ent_infos if info.get("canonical")] 

            # Compute coherence: internal vs external edge weight
            internal_weight = 0.0
            external_weight = 0.0
            # res.supergraph nodes may have 'internal_weight' stored
            try:
                super_n = f"comm:{comm_id}"
                internal_weight = float(res.supergraph.nodes.get(super_n, {}).get("internal_weight", 0.0))
                # external weight is sum of incident edge weights
                ext_w = 0.0
                for _, _, d in res.supergraph.edges(super_n, data=True):
                    ext_w += float(d.get("weight", 0.0))
                external_weight = ext_w
            except Exception:
                internal_weight = 0.0
                external_weight = 0.0

            coherence = 0.0
            denom = internal_weight + external_weight
            if denom > 0.0:
                coherence = internal_weight / denom

            graph.add_node(
                KGNode(
                    id=comm_node_id,
                    label="COMMUNITY",
                    properties={
                        "level": level,
                        "comm_id": comm_id,
                        "members_count": len(members),
                        "title": summary_obj.get("title"),
                        "summary": summary_obj.get("micro_summary"),
                        "micro_summary": summary_obj.get("micro_summary"),
                        "extractive_bullets": summary_obj.get("extractive_bullets", []),
                        "coherence": float(coherence),
                        "internal_weight": float(internal_weight),
                        "external_weight": float(external_weight),
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
                # majority-vote mapping: count upper community assignments among members
                upcount: Dict[int, int] = {}
                for member in lower_members:
                    upcid = upper.partition.get(member)
                    if upcid is not None:
                        upcount[upcid] = upcount.get(upcid, 0) + 1
                if not upcount:
                    continue
                # pick upper community with maximum votes
                best_upcid, best_count = max(upcount.items(), key=lambda kv: kv[1])
                # require majority (>50%) or at least one member if community small
                majority_threshold = max(1, (len(lower_members) // 2) + 1)
                if best_count >= majority_threshold:
                    lower_node_id = comm_node_map.get((lvl, lower_cid))
                    upper_node_id = comm_node_map.get((lvl + 1, best_upcid))
                    if lower_node_id and upper_node_id:
                        edge_id = f"e:comm_partof:{edge_counter}"; edge_counter += 1
                        graph.add_edge(
                            KGEdge(
                                id=edge_id,
                                source=lower_node_id,
                                target=upper_node_id,
                                type="PART_OF",
                                properties={"from_level": lvl, "to_level": lvl + 1, "vote_count": best_count, "members_total": len(lower_members)}
                            )
                        )

    # Mark bridge members with betweenness centrality to highlight connectors
    try:
        _mark_bridge_members(
            graph,
            edge_types=[
                "CO_OCCURS_WITH",
                "USED_WITH",
                "DEPLOYS_ON",
                "PROPOSED",
                "EVALUATED_ON",
                "APPLIED_IN",
                "ENABLES",
                "IMPROVES",
                "CAUSES",
                "SUBDOMAIN_OF",
            ],
        )
    except Exception:
        # best-effort; do not fail pipeline
        pass

    return comm_node_map


def _mark_bridge_members(graph: KnowledgeGraph, edge_types: Optional[List[str]] = None, top_k: int = 12) -> None:
    """Compute betweenness centrality on the entity graph and flag high connectors."""
    G = _build_entity_graph(graph, edge_types=edge_types)
    if G.number_of_nodes() == 0:
        return
    centrality = nx.betweenness_centrality(G, normalized=True)
    if not centrality:
        return
    # pick top-k by centrality
    ranked = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    bridge_nodes = {nid for nid, _ in ranked}
    score_map = {nid: score for nid, score in ranked}

    # Annotate MEMBER_OF edges connected to bridge nodes
    for edge in graph.edges:
        if edge.type == "MEMBER_OF" and edge.source in bridge_nodes:
            props = dict(edge.properties or {})
            props["bridge_score"] = float(score_map.get(edge.source, 0.0))
            props["is_bridge_member"] = True
            edge.properties = props
    # nothing else to add here; we only annotate existing MEMBER_OF edges
    return None


def extract_typed_semantic_relations(
    graph: KnowledgeGraph,
    all_entities_per_doc: Dict[str, List[Any]],
    sent_index: Dict[str, SentenceInfo],
    use_llm: bool = True,
) -> List[KGEdge]:
    """
    Extract typed semantic relations between entities using deterministic rules with optional LLM fallback.
    Stores KGEdge objects in the graph with types among
    APPLIED_IN, ENABLES, IMPROVES, CAUSES, SUBDOMAIN_OF and a confidence score.
    Returns list of created edges.
    """
    created: List[KGEdge] = []
    # Deterministic keyword map -> (relation, confidence)
    rule_map = [
        (r"appl(ies|ied) to|applied in|applies in|applied to", "APPLIED_IN", 0.92),
        (r"enable|enables|enabled", "ENABLES", 0.9),
        (r"improv|improves|better", "IMPROVES", 0.88),
        (r"cause|causes|leads to|results in", "CAUSES", 0.9),
        (r"subdomain|subset|specializ", "SUBDOMAIN_OF", 0.85),
    ]

    import re
    from datetime import datetime
    from uuid import uuid4

    # Iterate sentences via sent_index mapping: sent_id -> SentenceInfo
    for sent_id, sent_info in sent_index.items():
        node = graph.nodes.get(sent_id)
        if not node or node.label != "SENTENCE":
            continue

        text = (node.properties or {}).get("text") or getattr(sent_info, "text", "") or ""
        text_l = text.lower()

        # find entity mentions in this sentence via MENTION_IN edges
        mentions: List[Tuple[str, int]] = []  # (entity_node_id, char_start)
        for e in graph.edges:
            if e.type == "MENTION_IN" and e.target == sent_id:
                src = e.source
                src_node = graph.nodes.get(src)
                if not src_node or src_node.label not in {"ENTITY", "DOMAIN"}:
                    continue
                try:
                    cs = int((e.properties or {}).get("char_start", -1))
                except Exception:
                    cs = -1
                mentions.append((src, cs))

        if len(mentions) < 2:
            continue

        # order mentions by position to give a directional bias (head -> tail)
        mentions.sort(key=lambda m: m[1])

        # for each ordered pair of mentions, try to detect relation
        for i in range(len(mentions)):
            for j in range(i + 1, len(mentions)):
                a, _ = mentions[i]
                b, _ = mentions[j]
                rel: Optional[str] = None
                conf: float = 0.0

                # deterministic rules
                for pat, rtype, score in rule_map:
                    if re.search(pat, text_l):
                        rel = rtype
                        conf = score
                        break

                # fallback to LLM if enabled and no deterministic match
                if rel is None and use_llm and genai is not None:
                    prompt = {
                        "role": "system",
                        "parts": [
                            {
                                "text": (
                                    "Identify the most likely semantic relation between two entities mentioned in a sentence. "
                                    "Return strict JSON: {\"relation\": <ONE OF APPLIED_IN|ENABLES|IMPROVES|CAUSES|SUBDOMAIN_OF|NONE>, \"confidence\": float(0-1)}\n"
                                )
                            }
                        ],
                    }
                    user = {
                        "role": "user",
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "sentence": text,
                                        "entity_a": a,
                                        "entity_b": b,
                                    }
                                )
                            }
                        ],
                    }
                    try:
                        model = genai.GenerativeModel("gemini-1.5-pro")
                        resp = model.generate_content(
                            contents=[prompt, user], generation_config={}
                        )
                        raw = ""
                        if hasattr(resp, "text") and resp.text:
                            raw = resp.text
                        else:
                            raw = getattr(resp, "candidates", [{}])[0].get(
                                "content", ""
                            )
                        raw = raw.strip().strip("`")
                        parsed = json.loads(raw)
                        rel = parsed.get("relation")
                        conf = float(parsed.get("confidence", 0.6))
                    except Exception:
                        rel = None
                        conf = 0.0

                if not rel or rel == "NONE":
                    continue

                # create typed edge from a -> b
                edge_id = f"e:typed_rel:{uuid4().hex}"
                props = {
                    "doc_id": getattr(sent_info, "doc_id", None),
                    "sentence_id": sent_id,
                    "confidence": float(conf),
                    "source_sent": text,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                edge = KGEdge(
                    id=edge_id,
                    source=a,
                    target=b,
                    type=rel,
                    properties=props,
                )
                graph.add_edge(edge)
                created.append(edge)

    return created


def multi_hop_traversal(
    graph: KnowledgeGraph,
    start_ids: List[str],
    hops: int = 2,
    allowed_labels: Optional[Set[str]] = None,
):
    """Collect traversal paths up to `hops` hops over ENTITY and COMMUNITY nodes.

    Returns a list of unique paths (each path is a list of node ids).
    """
    if allowed_labels is None:
        allowed_labels = {"ENTITY", "COMMUNITY"}

    from collections import deque

    paths: List[List[str]] = []

    for start in start_ids:
        if start not in graph.nodes:
            continue

        queue: "deque[List[str]]" = deque([[start]])
        while queue:
            path = queue.popleft()
            current = path[-1]

            # record non-trivial paths
            if len(path) > 1:
                paths.append(path)

            # stop expanding once hop limit is reached
            if len(path) - 1 >= hops:
                continue

            # treat edges as undirected for traversal
            for e in graph.edges:
                neighbor: Optional[str] = None
                if e.source == current:
                    neighbor = e.target
                elif e.target == current:
                    neighbor = e.source
                if neighbor is None:
                    continue

                n_node = graph.nodes.get(neighbor)
                if not n_node or n_node.label not in allowed_labels:
                    continue
                if neighbor in path:
                    continue

                queue.append(path + [neighbor])

    # Deduplicate paths
    uniq: List[List[str]] = []
    seen_s: Set[str] = set()
    for p in paths:
        key = "->".join(p)
        if key in seen_s:
            continue
        seen_s.add(key)
        uniq.append(p)
    return uniq


def classify_query(query: str) -> str:
    """
    Lightweight rule-based query classifier.
    Returns one of: definition, overview, comparison, howwhy, missing
    """
    q = (query or "").lower()
    if q.strip().startswith("what is") or q.strip().startswith("define") or q.strip().startswith("who is"):
        return "definition"
    if any(tok in q for tok in ["overview", "major", "main", "summary", "applications"]):
        return "overview"
    if " vs " in q or " compare" in q or "comparison" in q:
        return "comparison"
    if any(tok in q for tok in ["why", "how", "impact", "cause", "explain"]):
        return "howwhy"
    if any(tok in q for tok in ["missing", "lack", "absent", "no evidence", "not found"]):
        return "missing"
    # default
    return "overview"


def handle_abstention(graph: KnowledgeGraph, query: str):
    """
    Create an abstention node in the graph describing missing evidence for `query` and return node id.
    """
    from uuid import uuid4
    from datetime import datetime
    node_id = f"abstain:{uuid4().hex}"
    graph.add_node(KGNode(id=node_id, label="ABSTENTION", properties={"query": query, "timestamp": datetime.utcnow().isoformat(), "evidence_count": 0}))
    return node_id


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
