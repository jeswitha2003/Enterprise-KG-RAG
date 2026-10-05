"""
stores/vector_store.py
------------------------
Wraps a LlamaIndex VectorStoreIndex backed by an embedded ChromaDB
collection. LlamaIndex handles chunk-to-node conversion, embedding calls,
and similarity ranking; Chroma handles the actual vector storage and
similarity search under the hood. Embedded (no server) by default —
`persist_directory` in config/settings.yaml switches to on-disk storage.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from ingestion import Chunk
from .embeddings import get_embed_model


@dataclass
class RetrievedChunk:
    text: str
    section: Optional[str]
    doc_id: str
    score: float


class VectorStore:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        Settings.embed_model = get_embed_model(cfg)

        persist_dir = cfg.get("vector_store", {}).get("persist_directory")
        client = chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.EphemeralClient()
        collection = client.get_or_create_collection("chunks")

        self._chroma_vector_store = ChromaVectorStore(chroma_collection=collection)
        self._storage_context = StorageContext.from_defaults(vector_store=self._chroma_vector_store)
        self._index = VectorStoreIndex.from_vector_store(
            self._chroma_vector_store, storage_context=self._storage_context
        )

    def add_chunks(self, chunks: List[Chunk]) -> int:
        documents = [
            Document(text=c.text, doc_id=c.id, metadata={"doc_id": c.doc_id, "section": c.section or ""})
            for c in chunks
        ]
        for doc in documents:
            self._index.insert(doc)
        return len(documents)

    def query(self, question: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        top_k = top_k or self.cfg.get("vector_store", {}).get("top_k", 3)
        retriever = self._index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(question)
        return [
            RetrievedChunk(
                text=n.node.text,
                section=n.node.metadata.get("section") or None,
                doc_id=n.node.metadata.get("doc_id", ""),
                score=round(float(n.score), 4) if n.score is not None else 0.0,
            )
            for n in nodes
        ]
