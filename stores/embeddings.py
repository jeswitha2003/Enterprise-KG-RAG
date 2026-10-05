"""
stores/embeddings.py
----------------------
Provides the embedding model used by the vector index, with the same
try-real-thing-then-fall-back pattern used everywhere else in this project.

HashEmbedding is a deterministic, dependency-free "embedding" built on the
hashing trick (each word hashes into one of `dim` buckets; the resulting
bag-of-words vector is L2-normalized). It has none of the semantic depth of
a real embedding model, but it is:
  - Deterministic (same text -> same vector, every time — required for
    reproducible tests).
  - Directionally reasonable for keyword-overlap style queries, since
    shared words hash to shared dimensions — enough to prove the retrieval
    *pipeline* (indexing, querying, ranking) works correctly, independent
    of embedding quality.
  - Zero network calls, zero API keys, zero extra ML dependencies.

Swap to Gemini's real embedding model by setting GOOGLE_API_KEY — no other
code changes needed, since both implementations conform to LlamaIndex's
BaseEmbedding interface.
"""

from __future__ import annotations
import hashlib
import os
import re
from typing import Any, Dict, List

import numpy as np
from llama_index.core.embeddings import BaseEmbedding


class HashEmbedding(BaseEmbedding):
    """Deterministic offline fallback embedding (see module docstring)."""
    dim: int = 256

    def _embed(self, text: str) -> List[float]:
        v = np.zeros(self.dim)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v.tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)


def get_embed_model(cfg: Dict[str, Any]) -> BaseEmbedding:
    """Returns a Gemini-backed embedding (via direct SDK call) if configured
    and reachable, else HashEmbedding. Calls the Gemini SDK directly rather
    than through llama-index-embeddings-gemini — see llm_client.py for why."""
    if cfg.get("llm", {}).get("provider") == "gemini" and os.getenv("GOOGLE_API_KEY"):
        try:
            from llm_client import gemini_embed
            model_name = f"models/{cfg['llm']['embed_model_name']}"

            class DirectGeminiEmbedding(BaseEmbedding):
                def _embed(self, text: str) -> List[float]:
                    return gemini_embed(text, model_name)

                def _get_query_embedding(self, query: str) -> List[float]:
                    return self._embed(query)

                def _get_text_embedding(self, text: str) -> List[float]:
                    return self._embed(text)

                async def _aget_query_embedding(self, query: str) -> List[float]:
                    return self._embed(query)

                async def _aget_text_embedding(self, text: str) -> List[float]:
                    return self._embed(text)

            # Fail fast on a bad key/network issue right here, so the
            # caller's try/except falls back to HashEmbedding immediately
            # rather than failing later on the first real query.
            embed_model = DirectGeminiEmbedding()
            embed_model._embed("connectivity check")
            return embed_model
        except Exception:
            pass
    return HashEmbedding()
