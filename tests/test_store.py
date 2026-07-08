"""Unit tests for the in-memory Vector Store fake (Req 3.4).

These example-based tests complement the property tests by asserting a concrete
edge case: a filtered search whose category matches no stored chunk returns an
empty result set without raising (Req 3.4).
"""

from __future__ import annotations

from datetime import datetime, timezone

from acdp.models import EmbeddedChunk, SourceCategory
from acdp.knowledge_base.store import InMemoryVectorStore


def _chunk(chunk_id: str, category: SourceCategory) -> EmbeddedChunk:
    """Build a minimal :class:`EmbeddedChunk` tagged with ``category``."""
    return EmbeddedChunk(
        chunk_id=chunk_id,
        source_id=f"source-{chunk_id}",
        category=category,
        ingested_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        text=f"text for {chunk_id}",
        vector=[1.0, 0.0, 0.0],
    )


def test_search_with_non_matching_category_returns_empty_without_raising() -> None:
    """A filtered search matching no stored chunk returns ``[]`` (Req 3.4).

    Chunks are stored only under OWASP_GENAI and MITRE_ATTACK; searching with a
    category none of them carry (COMPLIANCE) must return an empty result set and
    must not raise.
    """
    store = InMemoryVectorStore()
    store.upsert(
        [
            _chunk("a", SourceCategory.OWASP_GENAI),
            _chunk("b", SourceCategory.OWASP_GENAI),
            _chunk("c", SourceCategory.MITRE_ATTACK),
        ]
    )

    results = store.search([1.0, 0.0, 0.0], top_k=5, category=SourceCategory.COMPLIANCE)

    assert results == []


def test_search_non_matching_category_string_returns_empty() -> None:
    """The empty-result contract also holds for a raw category string (Req 3.4)."""
    store = InMemoryVectorStore()
    store.upsert([_chunk("a", SourceCategory.PLAYBOOK)])

    results = store.search([1.0, 0.0, 0.0], top_k=5, category="topology")

    assert results == []
