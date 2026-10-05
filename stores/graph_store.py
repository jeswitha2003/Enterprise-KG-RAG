"""
stores/graph_store.py
------------------------
The knowledge-graph half of the system. Two implementations behind one
interface (add_triple / neighbors / search_entities / find_path):

  - Neo4jGraphStore: real Cypher queries against a real Neo4j instance,
    used when NEO4J_URI (+ username/password) is set in the environment
    and the driver can actually connect.
  - NetworkXGraphStore: an in-memory directed multigraph with the exact
    same method signatures, used otherwise (and always in tests). This is
    the same "graceful degradation" pattern used for the LLM and the
    embedding model — the whole pipeline runs, and every test passes, with
    no external services and no API keys.

get_graph_store() tries Neo4j first (a real connectivity check, not just
"is the env var set" — a var could point at a dead instance) and falls back
silently on any failure.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from ingestion import Triple


@dataclass
class GraphResult:
    subject: str
    relation: str
    object: str


class NetworkXGraphStore:
    def __init__(self):
        self._graph = nx.MultiDiGraph()

    def add_triple(self, subject: str, relation: str, obj: str, source_chunk_id: Optional[str] = None) -> None:
        self._graph.add_edge(subject, obj, relation=relation, source_chunk_id=source_chunk_id)

    def add_triples(self, triples: List[Triple]) -> int:
        count = 0
        for t in triples:
            self.add_triple(t.subject, t.relation, t.object)
            count += 1
        return count

    def search_entities(self, keyword: str) -> List[str]:
        keyword_lower = keyword.lower()
        return [n for n in self._graph.nodes if keyword_lower in n.lower()]

    def neighbors(self, entity: str, depth: int = 1) -> List[GraphResult]:
        if entity not in self._graph:
            return []
        results: List[GraphResult] = []
        # Outgoing edges
        for _, target, data in self._graph.out_edges(entity, data=True):
            results.append(GraphResult(subject=entity, relation=data["relation"], object=target))
        # Incoming edges too — relationships are meaningful in both
        # directions for a "what's connected to X" query.
        for source, _, data in self._graph.in_edges(entity, data=True):
            results.append(GraphResult(subject=source, relation=data["relation"], object=entity))
        return results

    def find_path(self, source: str, target: str) -> Optional[List[str]]:
        try:
            undirected = self._graph.to_undirected()
            return nx.shortest_path(undirected, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def stats(self) -> Dict[str, int]:
        return {"nodes": self._graph.number_of_nodes(), "edges": self._graph.number_of_edges()}


class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()  # raises if unreachable — caller catches this

    def add_triple(self, subject: str, relation: str, obj: str, source_chunk_id: Optional[str] = None) -> None:
        # `relation` becomes the Cypher relationship TYPE, which must be a
        # valid identifier — the entity extractor already emits
        # UPPER_SNAKE_CASE labels for exactly this reason.
        query = (
            "MERGE (a:Entity {name: $subject}) "
            "MERGE (b:Entity {name: $object}) "
            f"MERGE (a)-[r:{relation}]->(b) "
            "SET r.source_chunk_id = $source_chunk_id"
        )
        with self._driver.session() as session:
            session.run(query, subject=subject, object=obj, source_chunk_id=source_chunk_id)

    def add_triples(self, triples: List[Triple]) -> int:
        count = 0
        for t in triples:
            self.add_triple(t.subject, t.relation, t.object)
            count += 1
        return count

    def search_entities(self, keyword: str) -> List[str]:
        query = "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($kw) RETURN e.name AS name"
        with self._driver.session() as session:
            return [r["name"] for r in session.run(query, kw=keyword)]

    def neighbors(self, entity: str, depth: int = 1) -> List[GraphResult]:
        query = (
            "MATCH (a:Entity {name: $entity})-[r]-(b:Entity) "
            "RETURN a.name AS a, type(r) AS rel, b.name AS b, startNode(r).name AS start"
        )
        results = []
        with self._driver.session() as session:
            for record in session.run(query, entity=entity):
                if record["start"] == entity:
                    results.append(GraphResult(subject=record["a"], relation=record["rel"], object=record["b"]))
                else:
                    results.append(GraphResult(subject=record["b"], relation=record["rel"], object=record["a"]))
        return results

    def find_path(self, source: str, target: str) -> Optional[List[str]]:
        query = (
            "MATCH p = shortestPath((a:Entity {name: $source})-[*..6]-(b:Entity {name: $target})) "
            "RETURN [n IN nodes(p) | n.name] AS path"
        )
        with self._driver.session() as session:
            record = session.run(query, source=source, target=target).single()
            return record["path"] if record else None

    def stats(self) -> Dict[str, int]:
        query = "MATCH (n) OPTIONAL MATCH ()-[r]->() RETURN count(DISTINCT n) AS nodes, count(r) AS edges"
        with self._driver.session() as session:
            record = session.run(query).single()
            return {"nodes": record["nodes"], "edges": record["edges"]}

    def close(self):
        self._driver.close()


def get_graph_store(cfg: Dict[str, Any]):
    """Tries a real Neo4j connection; falls back to NetworkX on any failure
    (missing env vars, unreachable host, wrong credentials)."""
    uri = os.getenv(cfg.get("graph_store", {}).get("neo4j_uri_env", "NEO4J_URI"))
    user = os.getenv(cfg.get("graph_store", {}).get("neo4j_user_env", "NEO4J_USERNAME"))
    password = os.getenv(cfg.get("graph_store", {}).get("neo4j_password_env", "NEO4J_PASSWORD"))

    if uri and user and password:
        try:
            return Neo4jGraphStore(uri, user, password), "neo4j"
        except Exception:
            pass
    return NetworkXGraphStore(), "networkx"
