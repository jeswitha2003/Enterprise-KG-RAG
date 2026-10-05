"""tests/test_stores.py — vector store (offline hash embedding) and graph
store (NetworkX fallback) determinism and correctness."""

from config import load_config
from ingestion import Chunk
from stores import VectorStore, get_graph_store, NetworkXGraphStore
from stores.embeddings import HashEmbedding


def test_hash_embedding_is_deterministic():
    emb = HashEmbedding()
    v1 = emb._get_text_embedding("Acme Corp revenue grew significantly")
    v2 = emb._get_text_embedding("Acme Corp revenue grew significantly")
    assert v1 == v2


def test_hash_embedding_different_text_gives_different_vector():
    emb = HashEmbedding()
    v1 = emb._get_text_embedding("Acme Corp revenue grew significantly")
    v2 = emb._get_text_embedding("The Munich facility had supply delays")
    assert v1 != v2


def test_vector_store_query_ranks_relevant_chunk_higher():
    cfg = load_config()
    cfg["llm"]["provider"] = "none"  # force offline embedding regardless of env
    store = VectorStore(cfg)
    chunks = [
        Chunk(id="a", text="Acme Corp revenue grew significantly this quarter.", doc_id="d", section="Summary"),
        Chunk(id="b", text="The Munich facility experienced supply chain delays.", doc_id="d", section="Risk"),
    ]
    store.add_chunks(chunks)
    results = store.query("revenue growth at Acme Corp", top_k=2)
    assert results[0].section == "Summary"


def test_networkx_graph_store_add_and_neighbors():
    store = NetworkXGraphStore()
    store.add_triple("John Smith", "REPORTS_TO", "Jane Doe")
    store.add_triple("Maria Fischer", "REPORTS_TO", "John Smith")

    neighbors = store.neighbors("John Smith")
    relations = {(n.subject, n.relation, n.object) for n in neighbors}
    assert ("John Smith", "REPORTS_TO", "Jane Doe") in relations
    assert ("Maria Fischer", "REPORTS_TO", "John Smith") in relations


def test_networkx_graph_store_find_path():
    store = NetworkXGraphStore()
    store.add_triple("Maria Fischer", "REPORTS_TO", "John Smith")
    store.add_triple("John Smith", "REPORTS_TO", "Jane Doe")
    path = store.find_path("Maria Fischer", "Jane Doe")
    assert path == ["Maria Fischer", "John Smith", "Jane Doe"]


def test_networkx_graph_store_no_path_returns_none():
    store = NetworkXGraphStore()
    store.add_triple("A", "RELATES_TO", "B")
    assert store.find_path("A", "Nonexistent") is None


def test_get_graph_store_falls_back_to_networkx_without_neo4j_env(monkeypatch):
    monkeypatch.delenv("NEO4J_URI", raising=False)
    cfg = load_config()
    store, backend = get_graph_store(cfg)
    assert backend == "networkx"
    assert isinstance(store, NetworkXGraphStore)


def test_search_entities_is_case_insensitive_substring_match():
    store = NetworkXGraphStore()
    store.add_triple("John Smith", "REPORTS_TO", "Jane Doe")
    assert "John Smith" in store.search_entities("john")
    assert "John Smith" in store.search_entities("SMITH")
