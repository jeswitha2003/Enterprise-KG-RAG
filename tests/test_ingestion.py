"""tests/test_ingestion.py — chunking and heuristic entity/triple extraction.
All deterministic and offline: no API keys, no network access needed."""

from ingestion.document_loader import chunk_document, split_sentences
from ingestion.entity_extractor import heuristic_extract_triples, candidate_entities

SAMPLE = """Q3 2026 FINANCIAL REVIEW

Executive Summary

Revenue grew 12 percent year over year.

Personnel Changes

John Smith was promoted to VP of Operations in September, reporting to CEO Jane Doe.
"""

RELATION_PATTERNS = [
    ("reports? to", "REPORTS_TO"),
    ("reporting to", "REPORTS_TO"),
    ("promoted to", "PROMOTED_TO"),
]


def test_chunk_document_splits_by_section():
    chunks = chunk_document(SAMPLE, doc_id="test", max_characters=60, new_after_n_chars=50)
    assert len(chunks) >= 2
    sections = [c.section for c in chunks]
    assert "Executive Summary" in sections
    assert "Personnel Changes" in sections


def test_split_sentences_does_not_merge_heading_into_next_sentence():
    """Regression test for a real bug: a section heading on its own line
    ('Personnel Changes') must not run into the following sentence's first
    entity ('John Smith'), or entity extraction produces a garbage combined
    entity like 'Personnel Changes John Smith'."""
    sentences = split_sentences("Personnel Changes\n\nJohn Smith was promoted.")
    assert "Personnel Changes" in sentences
    assert not any("Personnel Changes John Smith" in s for s in sentences)


def test_candidate_entities_filters_stopwords_and_months():
    entities = candidate_entities("In September, CEO Jane Doe met The Board.")
    assert "Jane Doe" in entities
    assert "September" not in entities
    assert "CEO" not in entities
    assert "The" not in entities


def test_heuristic_extract_triples_picks_sentence_subject_not_nearest_word():
    """Regression test for a real bug: for a compound sentence, the subject
    of a relation must be the sentence's actual (first) entity, not
    whichever capitalized word happens to sit closest to the relation
    phrase — otherwise 'John Smith was promoted to VP of Operations in
    September, reporting to CEO Jane Doe' incorrectly extracts 'September'
    or 'Operations' as the one who reports to Jane Doe."""
    sentence = "John Smith was promoted to VP of Operations in September, reporting to CEO Jane Doe."
    triples = heuristic_extract_triples(sentence, RELATION_PATTERNS)
    reports_to = [t for t in triples if t.relation == "REPORTS_TO"]
    assert len(reports_to) == 1
    assert reports_to[0].subject == "John Smith"
    assert "Jane Doe" in reports_to[0].object


def test_heuristic_extract_triples_generic_cooccurrence_fallback():
    """When no relation pattern matches, two co-occurring entities in a
    sentence still produce a weak MENTIONED_WITH edge, so the graph stays
    queryable even for sentences with no recognized relation verb."""
    triples = heuristic_extract_triples("Acme Corp works closely with Nordwind Logistics.", RELATION_PATTERNS)
    assert any(t.relation == "MENTIONED_WITH" for t in triples)
