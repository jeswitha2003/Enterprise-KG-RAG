"""tests/test_router.py — semantic routing heuristic, fully offline."""

from config import load_config
from router import heuristic_route


def _cfg():
    cfg = load_config()
    cfg["llm"]["provider"] = "none"  # force heuristic path regardless of env
    return cfg


def test_routes_relationship_question_to_graph():
    decision = heuristic_route("Who does John Smith report to?", _cfg())
    assert decision.route == "graph"


def test_routes_factual_question_to_vector():
    decision = heuristic_route("What was the revenue growth in Q3?", _cfg())
    assert decision.route == "vector"


def test_routes_mixed_question_to_hybrid():
    decision = heuristic_route("What is the relationship between the revenue growth and John Smith?", _cfg())
    assert decision.route == "hybrid"


def test_defaults_to_vector_with_no_keyword_matches():
    decision = heuristic_route("Tell me something.", _cfg())
    assert decision.route == "vector"
    assert "defaulting" in decision.reason
