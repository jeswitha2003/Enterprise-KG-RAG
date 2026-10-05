"""
rag/query_engine.py
----------------------
Ties everything together: given a question, decide the route, retrieve
context from whichever store(s) the route calls for, then synthesize a
final answer.

Like extraction and routing, synthesis has an LLM path and an offline
fallback: with a configured LLM, the retrieved context is handed to Gemini
to produce a natural-language answer; without one, the context itself
(clearly labeled) is returned as the "answer" so the retrieval pipeline is
still fully demoable and testable with no API key.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ingestion import candidate_entities
from router import route_query, RouteDecision
from stores import VectorStore, RetrievedChunk, GraphResult


@dataclass
class QueryResult:
    question: str
    route: str
    route_reason: str
    route_method: str
    answer: str
    answer_method: str          # "llm" | "context_only"
    vector_sources: List[RetrievedChunk] = field(default_factory=list)
    graph_triples: List[GraphResult] = field(default_factory=list)
    matched_entities: List[str] = field(default_factory=list)


def _graph_context(question: str, graph_store) -> tuple[List[GraphResult], List[str]]:
    candidates = candidate_entities(question)
    matched: List[str] = []
    for c in candidates:
        hits = graph_store.search_entities(c)
        for h in hits:
            if h not in matched:
                matched.append(h)

    triples: List[GraphResult] = []
    seen = set()
    for entity in matched:
        for triple in graph_store.neighbors(entity):
            key = (triple.subject, triple.relation, triple.object)
            if key not in seen:
                seen.add(key)
                triples.append(triple)

    # "relationship between X and Y" style questions: also try the direct path.
    if len(matched) >= 2:
        path = graph_store.find_path(matched[0], matched[1])
        if path:
            for a, b in zip(path, path[1:]):
                for triple in graph_store.neighbors(a):
                    if triple.object == b or triple.subject == b:
                        key = (triple.subject, triple.relation, triple.object)
                        if key not in seen:
                            seen.add(key)
                            triples.append(triple)

    return triples, matched


def _format_context(vector_sources: List[RetrievedChunk], graph_triples: List[GraphResult]) -> str:
    parts = []
    if vector_sources:
        parts.append("Relevant document excerpts:")
        for s in vector_sources:
            label = f" [{s.section}]" if s.section else ""
            parts.append(f"-{label} {s.text}")
    if graph_triples:
        parts.append("\nRelevant relationships from the knowledge graph:")
        for t in graph_triples:
            parts.append(f"- {t.subject} --{t.relation}--> {t.object}")
    return "\n".join(parts) if parts else "(no relevant context found)"


_SYNTHESIS_PROMPT = """Answer the user's question using ONLY the context below. If the
context doesn't contain enough information, say so plainly rather than guessing.

Context:
{context}

Question: {question}

Answer:"""


def _llm_synthesize(question: str, context: str, cfg: Dict[str, Any]) -> Optional[str]:
    if cfg.get("llm", {}).get("provider") != "gemini":
        return None
    if not os.getenv("GOOGLE_API_KEY"):
        return None
    try:
        from llm_client import gemini_complete
        prompt = _SYNTHESIS_PROMPT.format(context=context, question=question)
        return gemini_complete(prompt, f"models/{cfg['llm']['model_name']}", cfg["llm"]["temperature"]).strip()
    except Exception:
        return None


def answer_query(question: str, cfg: Dict[str, Any], vector_store: VectorStore, graph_store) -> QueryResult:
    decision: RouteDecision = route_query(question, cfg)

    vector_sources: List[RetrievedChunk] = []
    graph_triples: List[GraphResult] = []
    matched_entities: List[str] = []

    if decision.route in ("vector", "hybrid"):
        vector_sources = vector_store.query(question)
    if decision.route in ("graph", "hybrid"):
        graph_triples, matched_entities = _graph_context(question, graph_store)

    context = _format_context(vector_sources, graph_triples)
    llm_answer = _llm_synthesize(question, context, cfg)

    return QueryResult(
        question=question,
        route=decision.route,
        route_reason=decision.reason,
        route_method=decision.method,
        answer=llm_answer if llm_answer is not None else context,
        answer_method="llm" if llm_answer is not None else "context_only",
        vector_sources=vector_sources,
        graph_triples=graph_triples,
        matched_entities=matched_entities,
    )
