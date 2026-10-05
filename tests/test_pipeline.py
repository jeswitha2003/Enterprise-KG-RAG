"""tests/test_pipeline.py — end-to-end: ingest the sample report, then
answer a graph-routed and a vector-routed question. Fully offline."""

from pathlib import Path

from config import load_config
from ingestion import load_raw_text
from stores import VectorStore, get_graph_store
from pipeline import ingest_document
from rag import answer_query


def _offline_cfg():
    cfg = load_config()
    cfg["llm"]["provider"] = "none"
    return cfg


def test_full_pipeline_ingest_and_query():
    cfg = _offline_cfg()
    vector_store = VectorStore(cfg)
    graph_store, backend = get_graph_store(cfg)
    assert backend == "networkx"

    raw_text = load_raw_text(str(Path(__file__).resolve().parent.parent / "sample_data" / "q3_report.txt"),
                              is_file_path=True)
    result = ingest_document(raw_text, cfg, vector_store, graph_store, doc_id="q3")
    assert result.chunks_added == 5
    assert result.triples_added > 0
    assert result.extraction_method == "heuristic"

    graph_answer = answer_query("Who does John Smith report to?", cfg, vector_store, graph_store)
    assert graph_answer.route == "graph"
    subjects_objects = [(t.subject, t.object) for t in graph_answer.graph_triples]
    assert any("John Smith" in s or "John Smith" in o for s, o in subjects_objects)

    vector_answer = answer_query("What was the revenue growth in Q3?", cfg, vector_store, graph_store)
    assert vector_answer.route == "vector"
    assert len(vector_answer.vector_sources) > 0
