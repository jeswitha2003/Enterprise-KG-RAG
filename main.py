"""
main.py
--------
CLI entry point: ingests the sample corporate report, then runs a handful
of example questions through the router so you can see all three retrieval
paths (vector / graph / hybrid) fire in one run.

Usage:
    python main.py                          # runs the built-in demo
    python main.py --file path/to/doc.pdf    # ingest a real PDF, then demo queries
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from config import load_config
from ingestion import load_raw_text
from stores import VectorStore, get_graph_store
from pipeline import ingest_document
from rag import answer_query

DEMO_QUESTIONS = [
    "What was the revenue growth in Q3?",           # vector: pure content
    "Who does John Smith report to?",                # graph: pure relationship
    "How did the Munich supply chain issues affect the company's risk outlook?",  # hybrid
]


def print_result(result) -> None:
    print(f"\nQ: {result.question}")
    print(f"   route: {result.route} ({result.route_method} — {result.route_reason})")
    if result.matched_entities:
        print(f"   matched entities: {result.matched_entities}")
    print(f"   answer ({result.answer_method}):")
    for line in result.answer.splitlines():
        print(f"     {line}")


def main():
    parser = argparse.ArgumentParser(description="Enterprise Knowledge-Graph RAG demo")
    parser.add_argument("--file", type=str, default=None, help="Path to a .pdf or .txt file to ingest instead of the sample report")
    parser.add_argument("--question", type=str, default=None, help="Ask a single custom question instead of running the demo set")
    args = parser.parse_args()

    cfg = load_config()

    print("Setting up vector store and graph store...")
    vector_store = VectorStore(cfg)
    graph_store, graph_backend = get_graph_store(cfg)
    print(f"   graph backend: {graph_backend}")

    if args.file:
        raw_text = load_raw_text(args.file, is_file_path=True)
        doc_id = Path(args.file).stem
    else:
        raw_text = load_raw_text(str(Path(__file__).parent / "sample_data" / "q3_report.txt"), is_file_path=True)
        doc_id = "q3_report"

    print(f"Ingesting document '{doc_id}'...")
    result = ingest_document(raw_text, cfg, vector_store, graph_store, doc_id=doc_id)
    print(f"   chunks added: {result.chunks_added}")
    print(f"   triples added: {result.triples_added} (extraction method: {result.extraction_method})")
    print(f"   graph stats: {graph_store.stats()}")

    questions = [args.question] if args.question else DEMO_QUESTIONS
    for q in questions:
        r = answer_query(q, cfg, vector_store, graph_store)
        print_result(r)


if __name__ == "__main__":
    main()
