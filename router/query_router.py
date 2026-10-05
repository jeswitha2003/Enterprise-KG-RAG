"""
router/query_router.py
------------------------
Decides, for an incoming question, which retrieval path(s) actually make
sense to run:

  - "vector": the question is asking about *content* — a fact, a figure, a
    summary of what a section says. Best answered by semantic similarity
    search over chunks.
  - "graph": the question is asking about *relationships* — who reports to
    whom, what's connected to what, how two entities relate. Best answered
    by graph traversal, which a vector search can't do (embeddings don't
    encode multi-hop relational structure).
  - "hybrid": both signals are present — run both and merge the context.

Same LLM-with-heuristic-fallback pattern as the rest of the project: an LLM
classification call is attempted first if configured, and any failure
(including "provider: none") falls back to a deterministic keyword-overlap
heuristic, so routing is always testable offline.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RouteDecision:
    route: str          # "vector" | "graph" | "hybrid"
    method: str         # "llm" | "heuristic"
    reason: str          # human-readable explanation, shown in API responses


def heuristic_route(question: str, cfg: Dict[str, Any]) -> RouteDecision:
    q_lower = question.lower()
    graph_keywords: List[str] = cfg.get("router", {}).get("graph_keywords", [])
    vector_keywords: List[str] = cfg.get("router", {}).get("vector_keywords", [])

    graph_hits = [kw for kw in graph_keywords if kw in q_lower]
    vector_hits = [kw for kw in vector_keywords if kw in q_lower]

    if graph_hits and vector_hits:
        return RouteDecision(
            route="hybrid", method="heuristic",
            reason=f"matched both relational cues {graph_hits} and content cues {vector_hits}",
        )
    if graph_hits:
        return RouteDecision(
            route="graph", method="heuristic",
            reason=f"matched relational cue(s) {graph_hits}",
        )
    if vector_hits:
        return RouteDecision(
            route="vector", method="heuristic",
            reason=f"matched content cue(s) {vector_hits}",
        )
    # Default: most natural-language questions with no relational language
    # are asking about content, not structure.
    return RouteDecision(route="vector", method="heuristic", reason="no relational cues found; defaulting to content search")


_ROUTER_SYSTEM_PROMPT = """Classify the user's question into exactly one of these three
categories, based on what kind of retrieval would best answer it:
- "vector": asking about facts, figures, or a summary of document content
- "graph": asking about relationships, connections, or paths between entities
- "hybrid": needs both relational context and document content
Respond with ONLY the single word: vector, graph, or hybrid."""


def _llm_route(question: str, cfg: Dict[str, Any]) -> Optional[RouteDecision]:
    if cfg.get("llm", {}).get("provider") != "gemini":
        return None
    if not os.getenv("GOOGLE_API_KEY"):
        return None
    try:
        from llm_client import gemini_complete
        prompt = f"{_ROUTER_SYSTEM_PROMPT}\n\nQuestion: {question}"
        answer = gemini_complete(prompt, f"models/{cfg['llm']['model_name']}", 0.0).strip().lower()
        if answer not in ("vector", "graph", "hybrid"):
            return None
        return RouteDecision(route=answer, method="llm", reason="classified by LLM")
    except Exception:
        return None


def route_query(question: str, cfg: Dict[str, Any]) -> RouteDecision:
    llm_result = _llm_route(question, cfg)
    if llm_result is not None:
        return llm_result
    return heuristic_route(question, cfg)
