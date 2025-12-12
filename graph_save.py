import networkx as nx
from data_corpus import KnowledgeGraph
from neo4j import GraphDatabase
import json


def _serialize_value(value):
    """Convert Python objects to GraphML-compatible types."""
    if isinstance(value, (list, dict, set)):
        return json.dumps(value, ensure_ascii=False)
    elif value is None:
        return ""
    else:
        return str(value)


def kg_to_networkx(graph: KnowledgeGraph) -> nx.MultiDiGraph:
    """
    Convert KnowledgeGraph -> NetworkX MultiDiGraph (directed).
    Node attrs:
      - label
      - all properties from node.properties
    Edge attrs:
      - type
      - all properties from edge.properties
    """
    G = nx.MultiDiGraph()

    # Nodes
    for node in graph.nodes.values():
        attrs = {"label": node.label}
        # Serialize complex types to strings
        for key, value in node.properties.items():
            attrs[key] = _serialize_value(value)
        G.add_node(node.id, **attrs)

    # Edges
    for edge in graph.edges:
        attrs = {"type": edge.type}
        # Serialize complex types to strings
        for key, value in edge.properties.items():
            attrs[key] = _serialize_value(value)
        G.add_edge(edge.source, edge.target, key=edge.id, **attrs)

    return G


def save_kg_to_graphml(graph: KnowledgeGraph, path: str) -> None:
    """
    Save the KnowledgeGraph as a GraphML file.
    """
    G = kg_to_networkx(graph)
    nx.write_graphml(G, path)
    print(f"[GraphML] Saved KG to {path}")






def export_to_neo4j(
    graph: KnowledgeGraph,
    uri: str = "neo4j://127.0.0.1:7687",
    user: str = "neo4j",
    password: str = "password",
    clear_existing: bool = False,
) -> None:
    """
    Push the KnowledgeGraph into Neo4j.

    - Creates nodes with label = KGNode.label (ENTITY, DOCUMENT, SENTENCE, COMMUNITY, ...)
    - Creates relationships with type = KGEdge.type
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        if clear_existing:
            print("[Neo4j] Clearing existing graph...")
            session.run("MATCH (n) DETACH DELETE n")

        # Constraints (Neo4j 5.x+ syntax: FOR and REQUIRE)
        # Adjust labels if you add new ones.
        constraint_statements = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:ENTITY) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:DOCUMENT) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT sentence_id IF NOT EXISTS FOR (s:SENTENCE) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (c:COMMUNITY) REQUIRE c.id IS UNIQUE",
        ]
        for stmt in constraint_statements:
            try:
                session.run(stmt)
            except Exception as e:
                print(f"[WARN] Constraint creation failed (may already exist): {e}")

        # --- Create nodes ---
        print("[Neo4j] Inserting nodes...")
        for node in graph.nodes.values():
            label = node.label  # single label
            props = {}
            props["id"] = node.id
            # Serialize complex types to JSON strings for Neo4j
            for key, value in node.properties.items():
                props[key] = _serialize_value(value)

            cypher = f"""
            MERGE (n:{label} {{id: $id}})
            SET n += $props
            """
            session.run(cypher, id=node.id, props=props)

        # --- Create edges ---
        print("[Neo4j] Inserting edges...")
        for edge in graph.edges:
            props = {}
            props["id"] = edge.id
            # Serialize complex types to JSON strings for Neo4j
            for key, value in edge.properties.items():
                props[key] = _serialize_value(value)

            cypher = f"""
            MATCH (s {{id: $source}}), (t {{id: $target}})
            MERGE (s)-[r:{edge.type} {{id: $id}}]->(t)
            SET r += $props
            """
            session.run(
                cypher,
                source=edge.source,
                target=edge.target,
                id=edge.id,
                props=props,
            )

    driver.close()
    print("[Neo4j] Export complete.")