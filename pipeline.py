"""
pipeline.py
-------------
The ingestion pipeline that both main.py (CLI) and api/server.py call:
raw text -> chunks (Unstructured) -> triples (heuristic or LLM) -> stored
in both the vector store and the graph store.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ingestion import chunk_document, extract_triples
from stores import VectorStore


@dataclass
class IngestResult:
    doc_id: str
    chunks_added: int
    triples_added: int
    extraction_method: str


def ingest_document(raw_text: str, cfg: Dict[str, Any], vector_store: VectorStore,
                     graph_store, doc_id: Optional[str] = None) -> IngestResult:
    chunks = chunk_document(raw_text, doc_id=doc_id)
    chunks_added = vector_store.add_chunks(chunks)

    triples_added = 0
    extraction_method = "none"
    for chunk in chunks:
        triples, method = extract_triples(chunk.text, cfg)
        extraction_method = method  # last chunk's method; heuristic/llm don't mix per-run
        triples_added += graph_store.add_triples(triples)

    return IngestResult(
        doc_id=chunks[0].doc_id if chunks else (doc_id or "unknown"),
        chunks_added=chunks_added,
        triples_added=triples_added,
        extraction_method=extraction_method,
    )
