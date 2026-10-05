"""
llm_client.py
---------------
Direct calls to the Gemini SDK (google-generativeai), used instead of
llama-index's own Gemini wrapper packages (llama-index-llms-gemini,
llama-index-embeddings-gemini).

Why: llama-index-llms-gemini==0.6.2 (the latest release) has a hard
dependency on `pillow<11`, and no version of Pillow below 11 ships a
prebuilt Windows wheel for Python 3.14 — building it from source requires
a C compiler and zlib headers most machines don't have installed. Rather
than fight that constraint, this project calls the underlying Gemini SDK
directly, which has no such pin. Same pattern used in this project's
sibling, EcoRoute-Agent, which calls Gemini via langchain_google_genai
instead of a heavier wrapper for the same reason: fewer transitive
dependencies, fewer places for an unrelated package to break the install.
"""

from __future__ import annotations
from typing import List


def gemini_complete(prompt: str, model_name: str, temperature: float = 0.0) -> str:
    """Single-turn text completion. Raises on any failure — callers wrap
    this in their own try/except and fall back to the offline heuristic,
    exactly like every other LLM call site in this project."""
    import google.generativeai as genai
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt, generation_config={"temperature": temperature})
    return response.text


def gemini_embed(text: str, model_name: str = "models/text-embedding-004") -> List[float]:
    """Returns the embedding vector for a single piece of text."""
    import google.generativeai as genai
    result = genai.embed_content(model=model_name, content=text)
    return result["embedding"]
