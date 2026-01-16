# phase6_communities_hierarchy.py
from __future__ import annotations
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
import networkx as nx
try:
    import community as community_louvain  # type: ignore  # python-louvain
except Exception:  # pragma: no cover
    community_louvain = None  # type: ignore
import json
import numpy as np
import os
from graph_maker.embedding_cache import get_embeddings_with_cache

# Optional: Leiden community detection (recommended for more stable, well-connected clusters)
try:
    import igraph as ig  # type: ignore
    import leidenalg  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ig = None  # type: ignore
    leidenalg = None  # type: ignore

from utils.genai_compat import generate_text as genai_generate_text, is_available as genai_is_available

# Optional: use OpenAI for summaries if you have client object
# from openai import OpenAI  # assumed earlier in pipeline if needed

# Reuse your core types
from graph_maker.data_corpus import KnowledgeGraph, KGNode, KGEdge, SentenceInfo

# Optional disk cache helper (if you added cache_utils previously)
try:
    from utils.cache_utils import DiskJSONCache
    _COMM_SUMMARY_CACHE = DiskJSONCache("cache_community_summaries.json")
except Exception:
    _COMM_SUMMARY_CACHE = None


@dataclass
class CommunityLevelResult:
    level: int
    partition: Dict[str, int]                 # node_id (current level) -> community_id
    supergraph: nx.Graph                      # collapsed graph; nodes are community node ids (comm_l{level}:{id})
    community_nodes: Dict[int, List[str]]     # community_id -> list of member node_ids (current-level ids)
    origin_members: Dict[int, List[str]]      # community_id -> list of original entity ids
    modularity: Optional[float] = None
    n_nodes: int = 0
    n_communities: int = 0


def _community_node_id(level: int, comm_id: int) -> str:
    return f"comm_l{level}:{comm_id}"


def _compute_entity_importance(entity_G: nx.Graph, seed: int = 0) -> Dict[str, float]:
    """Compute a stable importance score per entity id.

    Uses exact betweenness for small graphs, approximate betweenness for medium graphs,
    and PageRank fallback for larger graphs.
    """
    n = entity_G.number_of_nodes()
    if n == 0:
        return {}

    try:
        if n <= 250:
            return nx.betweenness_centrality(entity_G, normalized=True, weight="weight")
        if n <= 2000:
            k = min(250, n)
            return nx.betweenness_centrality(entity_G, normalized=True, weight="weight", k=k, seed=seed)
        return nx.pagerank(entity_G, weight="weight")
    except Exception:
        try:
            return nx.pagerank(entity_G, weight="weight")
        except Exception:
            return {str(nid): 0.0 for nid in entity_G.nodes}


def _reassign_small_communities(
    G: nx.Graph,
    partition: Dict[str, int],
    min_comm_size: int,
) -> Dict[str, int]:
    """Merge communities smaller than min_comm_size into the best neighboring community.

    This avoids creating singleton negative IDs which breaks hierarchy collapsing.
    """
    if min_comm_size <= 1:
        return partition

    # Build comm -> members
    comm_members: Dict[int, List[str]] = {}
    for node, comm in partition.items():
        comm_members.setdefault(comm, []).append(node)

    small_comms = {cid for cid, members in comm_members.items() if len(members) < min_comm_size}
    if not small_comms:
        return partition

    new_partition = dict(partition)

    for small_cid in small_comms:
        members = comm_members.get(small_cid, [])
        for node in members:
            # Choose neighbor community with max total edge weight
            best_comm: Optional[int] = None
            best_w = 0.0
            for neigh in G.neighbors(node):
                neigh_comm = new_partition.get(neigh)
                if neigh_comm is None or neigh_comm == small_cid:
                    continue
                w = float(G[node][neigh].get("weight", 1.0))
                if w > best_w:
                    best_w = w
                    best_comm = neigh_comm
            if best_comm is not None:
                new_partition[node] = best_comm

    return new_partition


def _collapse_with_provenance(
    G: nx.Graph,
    partition: Dict[str, int],
    level: int,
    node_origin_members: Dict[str, Set[str]],
) -> Tuple[nx.Graph, Dict[int, List[str]], Dict[int, List[str]]]:
    """Collapse G into a supergraph whose nodes are community node ids.

    Returns:
      - supergraph H (nodes are comm_l{level}:{comm_id})
      - community_nodes: comm_id -> list of member node ids (from G)
      - origin_members: comm_id -> list of original entity ids
    """
    H = nx.Graph()

    community_nodes: Dict[int, List[str]] = {}
    for node, comm in partition.items():
        community_nodes.setdefault(comm, []).append(node)

    origin_members: Dict[int, List[str]] = {}
    for comm_id, members in community_nodes.items():
        origin: Set[str] = set()
        for m in members:
            origin |= node_origin_members.get(m, set())
        origin_members[comm_id] = sorted(origin)
        cid = _community_node_id(level, int(comm_id))
        H.add_node(cid, members=list(members), origin_members=origin, internal_weight=0.0)

    for u, v, data in G.edges(data=True):
        cu = partition.get(u)
        cv = partition.get(v)
        if cu is None or cv is None:
            continue
        su = _community_node_id(level, int(cu))
        sv = _community_node_id(level, int(cv))
        w = float(data.get("weight", 1.0))
        if su == sv:
            H.nodes[su]["internal_weight"] = float(H.nodes[su].get("internal_weight", 0.0)) + w
            continue
        if H.has_edge(su, sv):
            H[su][sv]["weight"] += w
        else:
            H.add_edge(su, sv, weight=w)

    return H, community_nodes, origin_members


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
        # Prefer a single source of truth if provided.
        try:
            from graph_maker.relation_schema import load_relation_schema, community_edge_types_from_schema

            edge_types = community_edge_types_from_schema(load_relation_schema())
        except Exception:
            edge_types = [
                "CO_OCCURS_WITH",
                "USED_WITH",
                "PROPOSED",
                "EVALUATED_ON",
                "DEPLOYS_ON",
                # typed semantic relations
                "APPLIED_IN",
                "ENABLES",
                "IMPROVES",
                "CAUSES",
                "SUBDOMAIN_OF",
            ]

    include_domain = os.getenv("KG_COMMUNITY_INCLUDE_DOMAIN", "0") == "1"
    min_edge_weight = float(os.getenv("KG_COMMUNITY_MIN_EDGE_WEIGHT", "0.0") or 0.0)

    G = nx.Graph()
    # Prefer ENTITY nodes; DOMAIN nodes tend to fragment and create singleton communities.
    allowed_labels = {"ENTITY"} | ({"DOMAIN"} if include_domain else set())
    for node_id, node in graph.nodes.items():
        if node.label in allowed_labels:
            G.add_node(node_id)

    # add edges with weight aggregation
    # Heuristic:
    # - CO_OCCURS_WITH: use computed weight/count if present
    # - typed relations: use confidence if present
    for e in graph.edges:
        if e.type not in edge_types:
            continue
        if e.source not in G.nodes or e.target not in G.nodes:
            continue

        props = e.properties or {}
        if e.type == "CO_OCCURS_WITH":
            w = float(props.get("weight", 0.0) or 0.0)
            if w <= 0.0:
                # fall back to count if weight wasn't computed
                w = float(props.get("count", 1) or 1.0)
        else:
            # typed relation edge
            w = float(props.get("confidence", 1.0) or 1.0)

        # Filter extremely weak edges to reduce singleton communities.
        if w < min_edge_weight:
            continue

        if G.has_edge(e.source, e.target):
            G[e.source][e.target]["weight"] += w
        else:
            G.add_edge(e.source, e.target, weight=w)

    # Hub downweighting: legal/finance corpora have generic hubs (e.g., insurer/insured/policy)
    # that can collapse communities into one blob. This rebalances clustering without dropping nodes.
    if (os.getenv("KG_COMMUNITY_HUB_DOWNWEIGHT", "1") or "1").strip() != "0" and G.number_of_nodes() > 0:
        try:
            deg_map = dict(G.degree(weight="weight"))
            deg_vals = [float(v) for v in deg_map.values() if float(v) > 0.0]
            if deg_vals:
                pctl = float(os.getenv("KG_COMMUNITY_HUB_DEGREE_PCTL", "0.98") or 0.98)
                pctl = min(0.999, max(0.5, pctl))
                thr = float(np.quantile(np.array(deg_vals, dtype=np.float64), pctl))
                penalty = float(os.getenv("KG_COMMUNITY_HUB_PENALTY", "0.2") or 0.2)
                penalty = min(1.0, max(0.01, penalty))
                hubs = {n for n, d in deg_map.items() if float(d) >= thr and float(d) > 0.0}
                if hubs:
                    for u, v, data in G.edges(data=True):
                        if u in hubs or v in hubs:
                            data["weight"] = float(data.get("weight", 1.0) or 1.0) * penalty
        except Exception:
            pass

    return G


def _partition_louvain(G: nx.Graph, *, seed: int = 0) -> Dict[str, int]:
    if community_louvain is None:
        raise RuntimeError("python-louvain is not installed (module 'community'). Install requirements.txt.")
    try:
        return community_louvain.best_partition(G, weight="weight", random_state=seed)
    except TypeError:
        return community_louvain.best_partition(G, weight="weight")


def _partition_leiden(G: nx.Graph, *, seed: int = 0, resolution: float = 1.0) -> Dict[str, int]:
    """Run Leiden on a NetworkX graph.

    Requires: `python-igraph` + `leidenalg`.
    Uses RBConfigurationVertexPartition which supports a resolution parameter.
    """
    if ig is None or leidenalg is None:
        raise RuntimeError("Leiden not available (install igraph and leidenalg)")
    if G.number_of_nodes() == 0:
        return {}

    # Stable ordering
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges: List[Tuple[int, int]] = []
    weights: List[float] = []
    for u, v, data in G.edges(data=True):
        edges.append((idx[u], idx[v]))
        weights.append(float(data.get("weight", 1.0) or 1.0))

    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.es["weight"] = weights

    # RBConfiguration is a robust default; it behaves like modularity with resolution.
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        seed=int(seed),
    )

    membership = list(part.membership)
    return {str(nodes[i]): int(membership[i]) for i in range(len(nodes))}


def _score_partition(G: nx.Graph, partition: Dict[str, int]) -> Tuple[float, int]:
    """Score a partition with a bias against community explosion.

    Returns (score, n_communities).
    We use modularity when available and penalize too many communities.
    """
    n_nodes = max(1, G.number_of_nodes())
    ncom = len(set(partition.values())) if partition else 0

    modularity = 0.0
    try:
        modularity = float(community_louvain.modularity(partition, G, weight="weight"))
    except Exception:
        modularity = 0.0

    # Penalize fragmentation. lambda defaults to 0.08; tune via env.
    lam = float(os.getenv("KG_COMMUNITY_FRAGMENT_PENALTY", "0.08") or 0.08)
    score = modularity - lam * (ncom / float(n_nodes))
    return float(score), int(ncom)


def _choose_partition(G: nx.Graph, *, seed: int, min_comm_size: int) -> Tuple[str, Dict[str, int], Optional[float]]:
    """Choose a partition method based on env.

    KG_COMMUNITY_METHOD:
      - louvain
      - leiden
      - auto (default): prefer Leiden if installed, else Louvain
      - ensemble: compute both (if possible), pick best score (modularity - fragmentation penalty)
    """
    method = (os.getenv("KG_COMMUNITY_METHOD", "auto") or "auto").strip().lower()
    resolution = float(os.getenv("KG_COMMUNITY_RESOLUTION", "1.0") or 1.0)

    candidates: List[Tuple[str, Dict[str, int]]] = []
    errors: List[str] = []

    def _add(name: str, part: Dict[str, int]) -> None:
        if part:
            # Merge tiny communities for stability/quality
            part = _reassign_small_communities(G, part, min_comm_size=min_comm_size)
        candidates.append((name, part))

    if method in {"louvain"}:
        _add("louvain", _partition_louvain(G, seed=seed))
    elif method in {"leiden"}:
        _add("leiden", _partition_leiden(G, seed=seed, resolution=resolution))
    elif method in {"ensemble"}:
        _add("louvain", _partition_louvain(G, seed=seed))
        try:
            _add("leiden", _partition_leiden(G, seed=seed, resolution=resolution))
        except Exception as e:
            errors.append(str(e))
    else:
        # auto
        if ig is not None and leidenalg is not None:
            try:
                _add("leiden", _partition_leiden(G, seed=seed, resolution=resolution))
            except Exception as e:
                errors.append(str(e))
        _add("louvain", _partition_louvain(G, seed=seed))

    # pick the best scored partition
    best_name = "louvain"
    best_part: Dict[str, int] = candidates[-1][1] if candidates else {}
    best_mod: Optional[float] = None
    best_score = float("-inf")
    for name, part in candidates:
        if not part:
            continue
        score, _ncom = _score_partition(G, part)
        if score > best_score:
            best_score = score
            best_name = name
            best_part = part
            try:
                best_mod = float(community_louvain.modularity(part, G, weight="weight"))
            except Exception:
                best_mod = None

    return best_name, best_part, best_mod


# -------------------------
# Multi-level community detection
# -------------------------
def compute_multilevel_communities(graph: KnowledgeGraph,
                                  max_levels: Optional[int] = None,
                                  min_comm_size: int = 1,
                                  edge_types: Optional[List[str]] = None,
                                  seed: int = 0,
                                  stop_min_nodes: int = 4,
                                  stop_compression_ratio: float = 0.95,
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

    # Track provenance: each current_graph node maps to a set of original entity ids
    node_origin_members: Dict[str, Set[str]] = {str(n): {str(n)} for n in G0.nodes}

    # Level loop with stability guards (depth is data-driven unless max_levels is set)
    current_graph = G0
    previous_ncom: Optional[int] = None
    max_depth_guard = 64
    level = 0
    while True:
        if current_graph.number_of_nodes() == 0:
            if verbose:
                print(f"[communities] Level {level}: graph empty, stopping.")
            break

        # choose community method (Louvain/Leiden/ensemble)
        chosen, partition, modularity = _choose_partition(current_graph, seed=seed, min_comm_size=min_comm_size)

        # collect result for this level with provenance
        supergraph, community_nodes, origin_members = _collapse_with_provenance(
            current_graph, partition, level=level, node_origin_members=node_origin_members
        )

        result = CommunityLevelResult(
            level=level,
            partition=dict(partition),
            supergraph=supergraph,
            community_nodes=community_nodes,
            origin_members=origin_members,
            modularity=modularity,
            n_nodes=int(current_graph.number_of_nodes()),
            n_communities=int(len(community_nodes)),
        )
        results.append(result)

        # Always compute ncom; it's used by stability guards below.
        ncom = len(set(partition.values()))
        if verbose:
            mod_s = f", modularity={modularity:.4f}" if modularity is not None else ""
            print(f"[communities] Level {level}: found {ncom} communities{mod_s} (method={chosen})")

        # explicit cap if requested
        if max_levels is not None and level >= max_levels:
            if verbose:
                print(f"[communities] Stopping: reached max_levels={max_levels}")
            break

        # Stability guards: stop if trivial or no structural gain
        # - too few nodes
        if current_graph.number_of_nodes() < stop_min_nodes:
            if verbose:
                print(f"[communities] Stopping: graph has fewer than {stop_min_nodes} nodes ({current_graph.number_of_nodes()})")
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
        # - barely any compression
        if current_graph.number_of_nodes() > 0:
            ratio = ncom / float(current_graph.number_of_nodes())
            if ratio >= stop_compression_ratio:
                if verbose:
                    print(f"[communities] Stopping: compression ratio too high ({ratio:.3f} >= {stop_compression_ratio})")
                break
        # - no meaningful structural gain compared to previous level
        if previous_ncom is not None and previous_ncom == ncom:
            if verbose:
                print(f"[communities] Stopping: no meaningful gain (ncom unchanged from previous level)")
            break
        previous_ncom = ncom

        # Guard against runaway depth
        if level >= max_depth_guard:
            if verbose:
                print(f"[communities] Stopping: reached depth guard {max_depth_guard}")
            break

        # Prepare next level: supergraph nodes are community node ids
        next_origin: Dict[str, Set[str]] = {}
        for cid, data in supergraph.nodes(data=True):
            origin = data.get("origin_members")
            if isinstance(origin, set):
                next_origin[str(cid)] = set(origin)
            elif isinstance(origin, (list, tuple)):
                next_origin[str(cid)] = {str(x) for x in origin}
            else:
                next_origin[str(cid)] = set()

        current_graph = supergraph
        node_origin_members = next_origin
        level += 1

    return results


# -------------------------
# Community nodes creation & linking into KG
# -------------------------
_SUMMARY_PROMPT = """
You are an expert at synthesizing Insurance and Legal information into a coherent knowledge graph hierarchy.

You will receive:
1) A list of entities (canonical, type, descriptions) that belong to a "community" (a thematic cluster).
2) A set of evidence sentences citing these entities.

Task:
Synthesize this information into a high-quality summary that explains the core theme of this community.

Return strictly JSON with:
{
    "title": "A descriptive 3-7 word title reflecting the core legal/financial theme",
    "micro_summary": "A 1-2 sentence overview (max 70 tokens) that defines this community's purpose.",
    "extractive_bullets": [
        "4-6 detailed fact sentences that synthesize the relationship between members.",
        "Ensure bullets explain how entities like 'Policy' and 'Coverage' interact.",
        "Include domain-specific details from the evidence."
    ],
    "citations": {
        "micro_summary": ["sent:..."],
        "bullets": [["sent:..."], ["sent:..."]]
    }
}

Rules:
- Be formal, factual, and informative.
- Ground ALL claims in the provided evidence.
- Ensure the summary highlights the connection between different entities, especially DOMAIN and ENTITY nodes.
"""

def _summarize_community_with_llm(
        genai_module,
        entity_infos: List[Dict[str, Any]],
        evidence_sentences: Optional[List[Dict[str, Any]]] = None,
        model_name: str = "gemini-1.5-pro",
        temperature: float = 0.3,
) -> Dict[str, Any]:
    """
    Use Google Generative AI (Gemini) to summarize a community.
    - genai_module: the imported google.generativeai module (configured).
    - entity_infos: list of {"canonical":..., "type":..., "descriptions":[...]}
    Returns dict with "title" and "summary".
    """
    # NOTE: We intentionally do not depend on a specific Google SDK here.
    # Calls are routed through genai_compat so both `google-generativeai` and
    # `google-genai` work.

    # build canonical key for caching (coerce everything to string)
    keys = [str(e.get("canonical", "") or "") for e in (entity_infos or [])]
    sent_ids = [str(s.get("sentence_id", "") or "") for s in (evidence_sentences or [])]
    key_raw = "||".join(sorted(keys)) + "##" + "||".join(sorted(sent_ids))
    if _COMM_SUMMARY_CACHE is not None:
        cached = _COMM_SUMMARY_CACHE.get(key_raw)
        if cached:
            return cached

    payload = {"entities": entity_infos, "evidence_sentences": (evidence_sentences or [])}
    prompt = _SUMMARY_PROMPT.strip() + "\n\nInput:\n" + json.dumps(payload, ensure_ascii=False)
    raw = (genai_generate_text(model_name, prompt, temperature=temperature, purpose="ENTITY") or "").strip()

    if not raw:
        if _COMM_SUMMARY_CACHE is not None:
            _COMM_SUMMARY_CACHE.set(key_raw, {"title": None, "micro_summary": None, "extractive_bullets": []})
        return {"title": None, "micro_summary": None, "extractive_bullets": []}

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
            "citations": obj.get("citations") or {},
        }
    except Exception:
        # fallback: minimal attempt to salvage text
        out = {"title": None, "micro_summary": raw or None, "extractive_bullets": []}

    if _COMM_SUMMARY_CACHE is not None:
        _COMM_SUMMARY_CACHE.set(key_raw, out)
    return out


def _collect_evidence_sentences_for_members(
    graph: KnowledgeGraph,
    members: List[str],
    *,
    max_sentences: int = 12,
    mention_map: Optional[Dict[str, Dict[str, int]]] = None,
) -> List[Dict[str, Any]]:
    """Collect top evidence sentences for a community, grounded via MENTION_IN edges."""
    if not members or max_sentences <= 0:
        return []

    counts: Dict[str, int] = {}
    if mention_map is not None:
        # Fast path: aggregate precomputed entity->sentence mention counts.
        for m in members:
            smap = mention_map.get(str(m))
            if not smap:
                continue
            for sid, c in smap.items():
                counts[sid] = counts.get(sid, 0) + int(c)
    else:
        # Fallback: scan edges (legacy behavior).
        member_set = {str(m) for m in members}
        for e in graph.edges:
            if e.type != "MENTION_IN":
                continue
            if e.source not in member_set:
                continue
            sid = str(e.target)
            if not sid.startswith("sent:"):
                continue
            counts[sid] = counts.get(sid, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:max_sentences]
    out: List[Dict[str, Any]] = []
    for sid, c in ranked:
        node = graph.nodes.get(sid)
        if not node or not isinstance(node.properties, dict):
            continue
        txt = str(node.properties.get("text") or "").strip()
        if not txt:
            continue
        out.append({"sentence_id": sid, "text": txt, "mention_count": int(c)})
    return out


def _precompute_mention_in_sentence_counts(graph: KnowledgeGraph) -> Dict[str, Dict[str, int]]:
    """Build map: entity_id -> {sent_id -> mention_count} from MENTION_IN edges."""
    out: Dict[str, Dict[str, int]] = {}
    for e in graph.edges:
        if e.type != "MENTION_IN":
            continue
        src = str(e.source)
        sid = str(e.target)
        if not sid.startswith("sent:"):
            continue
        smap = out.get(src)
        if smap is None:
            smap = {}
            out[src] = smap
        smap[sid] = smap.get(sid, 0) + 1
    return out


def _add_semantic_part_communities(graph: KnowledgeGraph, *, verbose: bool = True) -> Dict[str, str]:
    """Create semantic constitution communities: one per PART.

    Requires ARTICLE->PART edges of type PART_OF (created deterministically in ner.py).
    The resulting communities help retrieval expansion without relying on statistical clustering.
    """
    enabled = (os.getenv("KG_SEMANTIC_PART_COMMUNITIES", "1") or "1").strip() != "0"
    if not enabled:
        return {}

    # Discover PART nodes
    part_nodes: List[KGNode] = []
    for node in graph.nodes.values():
        if node.label != "ENTITY":
            continue
        ntype = (node.properties or {}).get("type")
        if ntype == "PART" or str(node.id).startswith("part:"):
            part_nodes.append(node)

    if not part_nodes:
        return {}

    # Precompute article members by PART_OF edges
    part_to_articles: Dict[str, Set[str]] = {p.id: set() for p in part_nodes}
    for e in graph.edges:
        if e.type != "PART_OF":
            continue
        # Only treat ENTITY->ENTITY edges as constitutional structure
        src = graph.nodes.get(e.source)
        tgt = graph.nodes.get(e.target)
        if not src or not tgt:
            continue
        if src.label != "ENTITY" or tgt.label != "ENTITY":
            continue
        if not str(src.id).startswith("art:"):
            continue
        if tgt.id in part_to_articles:
            part_to_articles[tgt.id].add(src.id)

    # Create community nodes + MEMBER_OF edges
    comm_map: Dict[str, str] = {}
    edge_counter = len(graph.edges)
    for p in sorted(part_nodes, key=lambda n: str(n.id)):
        members = sorted(part_to_articles.get(p.id, set()))
        if not members:
            continue

        part_key = str(p.id).split(":", 1)[1] if ":" in str(p.id) else str(p.id)
        comm_id = f"comm_part:{part_key}"

        # Stable, non-LLM summary: use PART description/title if present
        title = (p.properties or {}).get("description") or (p.properties or {}).get("canonical") or str(p.id)
        title = str(title).strip()
        micro = f"Constitutional Part grouping; contains {len(members)} Articles."

        if comm_id not in graph.nodes:
            graph.add_node(
                KGNode(
                    id=comm_id,
                    label="COMMUNITY",
                    properties={
                        "level": 0,
                        "comm_id": part_key,
                        "members_count": len(members),
                        "title": title,
                        "summary": micro,
                        "micro_summary": micro,
                        "extractive_bullets": [title],
                        "coherence": 1.0,
                        "method": "semantic_part",
                        "part_node_id": p.id,
                    },
                )
            )
            if verbose:
                print(f"[semantic communities] Added {comm_id} (members={len(members)})")

        comm_map[p.id] = comm_id

        for mid in members:
            edge_id = f"e:sem_part:{edge_counter}"; edge_counter += 1
            graph.add_edge(
                KGEdge(
                    id=edge_id,
                    source=mid,
                    target=comm_id,
                    type="MEMBER_OF",
                    properties={"level": 0, "method": "semantic_part", "part": part_key},
                )
            )

    return comm_map


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
    # Always add semantic Part-based communities first (legal backbone).
    try:
        _add_semantic_part_communities(graph, verbose=verbose)
    except Exception:
        pass
    # Avoid flooding the KG with low-signal community nodes.
    # This is separate from min_comm_size (used during partitioning) because you may want
    # to keep assignments but only materialize larger communities as explicit nodes.
    min_comm_node_size = int(os.getenv("KG_MIN_COMMUNITY_NODE_SIZE", "3") or 3)

    # If higher-level hierarchy collapses (common on small/noisy graphs), skip materializing it.
    skip_degenerate_levels = (os.getenv("KG_SKIP_DEGENERATE_COMMUNITY_LEVELS", "1") or "1").strip() != "0"
    modularity_floor = float(os.getenv("KG_COMMUNITY_MODULARITY_FLOOR", "0.000001") or 0.000001)

    # Community summaries can be expensive (LLM). Default to sequential; allow opt-in parallelism.
    disable_summaries = (os.getenv("KG_DISABLE_COMMUNITY_SUMMARIES", "0") or "0").strip() == "1"
    summary_workers = int(os.getenv("KG_COMMUNITY_SUMMARY_WORKERS", "1") or 1)
    summary_workers = max(1, min(summary_workers, 16))

    # Map: (level, comm_id) -> community_node_id string
    comm_node_map: Dict[Tuple[int, int], str] = {}

    # Precompute centrality on the entity graph so we can rank members by importance
    entity_G = _build_entity_graph(graph)
    centrality_scores = _compute_entity_importance(entity_G, seed=0)

    # Precompute evidence grounding map once (dominant cost otherwise).
    mention_map = _precompute_mention_in_sentence_counts(graph)

    # Filter out degenerate levels before materializing nodes.
    filtered_results: List[CommunityLevelResult] = []
    for res in community_results:
        if not skip_degenerate_levels:
            filtered_results.append(res)
            continue
        ncom = int(res.n_communities or 0)
        mod = res.modularity
        mod_v = float(mod) if mod is not None else None
        if res.level >= 1 and (ncom <= 1 or (mod_v is not None and mod_v <= modularity_floor)):
            if verbose:
                print(f"[communities] Skipping degenerate level {res.level} (ncom={ncom}, modularity={mod_v})")
            continue
        filtered_results.append(res)

    community_results = filtered_results

    def _build_summary(ent_infos: List[Dict[str, Any]], evidence_sents: List[Dict[str, Any]]) -> Dict[str, Any]:
        if disable_summaries or genai is None:
            return {"title": None, "micro_summary": None, "extractive_bullets": []}
        return _summarize_community_with_llm(genai, ent_infos, evidence_sents)

    # First pass: create community nodes for each level
    # Optionally parallelize summary generation; node insertion stays on main thread.
    pending: List[Tuple[int, int, str, List[str], List[Dict[str, Any]], List[Dict[str, Any]], float, float]] = []
    for res in community_results:
        level = res.level
        # community nodes in res.community_nodes: {comm_id: [member_node_ids]}
        for comm_id, members in res.community_nodes.items():
            if len(members) < min_comm_node_size:
                continue
            comm_node_id = _community_node_id(level, comm_id)

            # Compute cohesion signals from the collapsed supergraph:
            # - internal_weight: sum of edge weights within the community in the prior graph
            # - external_weight: total weight of edges from this community to other communities
            sg_node_id = _community_node_id(level, int(comm_id))
            internal_weight = 0.0
            external_weight = 0.0
            try:
                if sg_node_id in res.supergraph:
                    internal_weight = float(res.supergraph.nodes[sg_node_id].get("internal_weight", 0.0) or 0.0)
                    for _u, _v, data in res.supergraph.edges(sg_node_id, data=True):
                        external_weight += float((data or {}).get("weight", 1.0) or 0.0)
            except Exception:
                internal_weight = 0.0
                external_weight = 0.0

            # prepare props
            # pick most central ORIGINAL entities for summary (provenance-aware)
            origin = res.origin_members.get(comm_id, [])
            ranked_origin = sorted(origin, key=lambda n: centrality_scores.get(n, 0.0), reverse=True)
            sample_entities = ranked_origin[:include_entity_sample]
            # fetch descriptions/types from KG nodes if present
            ent_infos = []
            for n in sample_entities:
                node = graph.nodes.get(n)
                ent_infos.append({
                    "canonical": node.properties.get("canonical") if node else n,
                    "type": node.properties.get("type") if node else None,
                    "descriptions": node.properties.get("descriptions", []) if node else []
                })
            evidence_sents = _collect_evidence_sentences_for_members(
                graph,
                members,
                max_sentences=int(os.getenv("KG_COMMUNITY_SUMMARY_MAX_SENTENCES", "12") or 12),
                mention_map=mention_map,
            )
            pending.append((level, comm_id, comm_node_id, members, ent_infos, evidence_sents, float(internal_weight), float(external_weight)))

    summaries: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if pending:
        if summary_workers <= 1 or len(pending) <= 4:
            for (level, comm_id, _comm_node_id, _members, ent_infos, evidence_sents, _iw, _ew) in pending:
                summaries[(level, comm_id)] = _build_summary(ent_infos, evidence_sents)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=summary_workers) as ex:
                futs = {
                    ex.submit(_build_summary, ent_infos, evidence_sents): (level, comm_id)
                    for (level, comm_id, _cid, _m, ent_infos, evidence_sents, _iw, _ew) in pending
                }
                for fut in as_completed(futs):
                    key = futs[fut]
                    try:
                        summaries[key] = fut.result() or {"title": None, "micro_summary": None, "extractive_bullets": []}
                    except Exception:
                        summaries[key] = {"title": None, "micro_summary": None, "extractive_bullets": []}

    for (level, comm_id, comm_node_id, members, ent_infos, evidence_sents, internal_weight, external_weight) in pending:
        summary_obj = summaries.get((level, comm_id), {"title": None, "micro_summary": None, "extractive_bullets": []})

        # Lightweight fallback summaries if LLM returns empty
        if not summary_obj.get("micro_summary"):
            sample_labels = [f"{info.get('canonical')} ({info.get('type') or 'entity'})" for info in ent_infos]
            summary_obj["micro_summary"] = ", ".join(sample_labels)[:220] or None
        if not summary_obj.get("extractive_bullets"):
            summary_obj["extractive_bullets"] = [info.get("canonical") for info in ent_infos if info.get("canonical")] 

        # Compute coherence: internal vs external edge weight
        coherence = 0.0
        denom = float(internal_weight) + float(external_weight)
        if denom > 0.0:
            coherence = float(internal_weight) / denom

        # Find the CommunityLevelResult for this level to attach modularity.
        res_mod = None
        for r in community_results:
            if r.level == level:
                res_mod = r
                break

        props = {
            "level": level,
            "comm_id": comm_id,
            "members_count": len(members),
            "origin_members_count": len(res_mod.origin_members.get(comm_id, [])) if res_mod is not None else None,
            "title": summary_obj.get("title"),
            "summary": summary_obj.get("micro_summary"),
            "micro_summary": summary_obj.get("micro_summary"),
            "extractive_bullets": summary_obj.get("extractive_bullets", []),
            "citations": summary_obj.get("citations") if isinstance(summary_obj, dict) else None,
            "evidence_sentence_ids": [str(s.get("sentence_id")) for s in (evidence_sents or []) if s.get("sentence_id")],
            "coherence": float(coherence),
            "internal_weight": float(internal_weight),
            "external_weight": float(external_weight),
            "modularity": (res_mod.modularity if res_mod is not None else None),
        }

        graph.add_node(
            KGNode(
                id=comm_node_id,
                label="COMMUNITY",
                properties=props,
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
                if not comm_node_id:
                    continue
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
            upper = community_results[lvl + 1]
            # upper.partition maps child node id (comm_l{lvl}:{child}) -> upper comm id
            for child_node_id, parent_comm_id in upper.partition.items():
                parent_node_id = _community_node_id(lvl + 1, int(parent_comm_id))
                if child_node_id in graph.nodes and parent_node_id in graph.nodes:
                    edge_id = f"e:comm_partof:{edge_counter}"; edge_counter += 1
                    graph.add_edge(
                        KGEdge(
                            id=edge_id,
                            source=str(child_node_id),
                            target=parent_node_id,
                            type="PART_OF",
                            properties={"from_level": lvl, "to_level": lvl + 1},
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
    centrality = _compute_entity_importance(G, seed=0)
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
        allowed_labels = {"ENTITY", "COMMUNITY", "DOMAIN"}

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
