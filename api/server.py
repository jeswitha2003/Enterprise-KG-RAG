"""
api/server.py
----------------
FastAPI wrapper around the ingestion pipeline and the RAG query engine.

Run locally:
    uvicorn api.server:app --reload --port 8001

Endpoints:
    GET  /api/health              -> {"status": "ok"}
    POST /api/ingest              -> runs a raw text document through the
                                      full pipeline (chunk, embed, extract,
                                      store in both the vector and graph stores)
    POST /api/query               -> answers a question, returning which
                                      route was chosen and why, the raw
                                      retrieved context, and the answer
    GET  /api/graph/stats         -> node/edge counts in the graph store
    GET  /api/sample-document     -> returns the built-in sample report text
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import load_config
from stores import VectorStore, get_graph_store
from pipeline import ingest_document
from rag import answer_query

app = FastAPI(title="Enterprise Knowledge-Graph RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cfg = load_config()
_vector_store = VectorStore(_cfg)
_graph_store, _graph_backend = get_graph_store(_cfg)


class IngestRequest(BaseModel):
    text: str
    doc_id: Optional[str] = None


class IngestResponse(BaseModel):
    doc_id: str
    chunks_added: int
    triples_added: int
    extraction_method: str
    graph_backend: str


class QueryRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    text: str
    section: Optional[str] = None
    score: float


class GraphTriple(BaseModel):
    subject: str
    relation: str
    object: str


class QueryResponse(BaseModel):
    question: str
    route: str
    route_reason: str
    route_method: str
    answer: str
    answer_method: str
    matched_entities: List[str]
    vector_sources: List[SourceChunk]
    graph_triples: List[GraphTriple]


@app.get("/api/health")
def health():
    return {"status": "ok", "graph_backend": _graph_backend}


@app.get("/api/sample-document")
def sample_document():
    path = Path(__file__).resolve().parent.parent / "sample_data" / "q3_report.txt"
    return {"text": path.read_text(encoding="utf-8")}


@app.get("/api/graph/stats")
def graph_stats():
    return _graph_store.stats()


@app.post("/api/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    result = ingest_document(request.text, _cfg, _vector_store, _graph_store, doc_id=request.doc_id)
    return IngestResponse(
        doc_id=result.doc_id,
        chunks_added=result.chunks_added,
        triples_added=result.triples_added,
        extraction_method=result.extraction_method,
        graph_backend=_graph_backend,
    )


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest):
    result = answer_query(request.question, _cfg, _vector_store, _graph_store)
    return QueryResponse(
        question=result.question,
        route=result.route,
        route_reason=result.route_reason,
        route_method=result.route_method,
        answer=result.answer,
        answer_method=result.answer_method,
        matched_entities=result.matched_entities,
        vector_sources=[
            SourceChunk(text=s.text, section=s.section, score=s.score) for s in result.vector_sources
        ],
        graph_triples=[
            GraphTriple(subject=t.subject, relation=t.relation, object=t.object) for t in result.graph_triples
        ],
    )
