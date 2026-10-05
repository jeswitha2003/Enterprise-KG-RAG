"""
ingestion/document_loader.py
-----------------------------
Turns a messy corporate document (PDF bytes, or already-extracted text) into
a list of clean, section-aware chunks.

Two-stage design, deliberately:
  1. Raw text extraction (pypdf for PDFs — lightweight, no ML models).
  2. Structure-aware chunking (Unstructured's `partition_text` +
     `chunk_by_title`) — this is what actually earns the "handles messy
     real-world documents" claim: it recovers section boundaries (titles,
     narrative text, lists) from plain text without needing layout-detection
     ML models, which keeps the whole pipeline installable without a
     multi-gigabyte PyTorch/CUDA dependency chain.
"""

from __future__ import annotations
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from unstructured.partition.text import partition_text
from unstructured.chunking.title import chunk_by_title


@dataclass
class Chunk:
    """One section-aware chunk of a document, ready for embedding + storage."""
    id: str
    text: str
    doc_id: str
    section: Optional[str] = field(default=None)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF using pypdf (fast, no ML models)."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_raw_text(source: str, is_file_path: bool = False) -> str:
    """
    Returns raw text from either:
      - a file path to a .pdf or .txt file (is_file_path=True), or
      - a raw text string already in memory (is_file_path=False) — this is
        the path used by the API and by tests, so nothing here requires a
        real PDF on disk to exercise the chunking/extraction logic.
    """
    if not is_file_path:
        return source

    path = Path(source)
    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(str(path))
    return path.read_text(encoding="utf-8")


def chunk_document(raw_text: str, doc_id: Optional[str] = None,
                    max_characters: int = 350, new_after_n_chars: int = 300) -> List[Chunk]:
    """
    Splits raw text into section-aware chunks: Unstructured's `partition_text`
    classifies lines into Title / NarrativeText / Text elements, and
    `chunk_by_title` groups consecutive elements under their nearest title
    into chunks capped at `max_characters`. This is what lets the retrieval
    layer return "the Risk Factors section" rather than an arbitrary
    fixed-size slice that might cut a sentence (or an entity) in half.
    """
    doc_id = doc_id or str(uuid.uuid4())[:8]

    elements = partition_text(text=raw_text)
    if not elements:
        return []

    title_chunks = chunk_by_title(elements, max_characters=max_characters,
                                   new_after_n_chars=new_after_n_chars)

    chunks: List[Chunk] = []
    for i, tc in enumerate(title_chunks):
        # Unstructured's chunk objects carry the originating elements'
        # metadata; the first Title element under a chunk (if any) becomes
        # a human-readable "section" label for that chunk.
        section = None
        for el in getattr(tc, "metadata", None) and [tc] or []:
            pass  # chunk_by_title chunks don't retain sub-element types directly
        first_line = str(tc).split("\n", 1)[0].strip()
        # Heuristic: a short first line with no sentence-ending punctuation
        # is almost certainly the section title chunk_by_title grouped under.
        if len(first_line) < 80 and not first_line.endswith((".", "!", "?")):
            section = first_line

        chunks.append(Chunk(
            id=f"{doc_id}-{i}",
            text=str(tc),
            doc_id=doc_id,
            section=section,
        ))
    return chunks


def split_sentences(text: str) -> List[str]:
    """Simple, dependency-free sentence splitter used by entity extraction.

    Splits on line breaks FIRST, then on sentence punctuation within each
    line. This matters more than it looks: without it, a section heading
    like "Personnel Changes" (its own line, no terminal punctuation) would
    run straight into the next line's first sentence — "John Smith was
    promoted..." — and the entity regex would then merge them into one
    garbage entity, "Personnel Changes John Smith". Treating line breaks as
    boundaries keeps headings and body text from bleeding into each other,
    which is exactly the shape of a real chunked corporate document (title
    line, blank line, paragraph).

    Not linguistically perfect (doesn't handle 'Mr. Smith' etc.), but
    deterministic and good enough for corporate-report style prose."""
    sentences: List[str] = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            continue
        sentences.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
    return sentences
