"""Property-based tests for the ingestion pipeline (Req 2.1, 2.2, 2.3).

These tests exercise :class:`~acdp.rag.ingest.IngestionPipeline` against the
in-memory :class:`~acdp.rag.store.InMemoryVectorStore` fake and a deterministic
stub LLM Gateway, so no live Ollama/Qdrant services are required.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from hypothesis import given, settings

from acdp.exceptions import IngestionError
from acdp.models import KnowledgeSource, SourceCategory
from acdp.knowledge_base.ingest import IngestionPipeline
from acdp.knowledge_base.store import InMemoryVectorStore
from tests.strategies import knowledge_sources


class _StubGateway:
    """A deterministic stub :class:`~acdp.llm.LLMGateway` for ingestion tests.

    Embedding is a pure function of the chunk text (its length and first
    codepoint) so results are reproducible across runs without any network
    calls. Only ``embed`` is needed by the ingestion pipeline; ``generate`` is
    provided to satisfy the interface but is never called here.
    """

    def embed(self, text: str, model: str | None = None) -> list[float]:
        first = float(ord(text[0])) if text else 0.0
        return [float(len(text)), first, 1.0]

    def generate(self, request):  # pragma: no cover - unused by ingestion
        raise NotImplementedError


# A pinned clock so the ingestion timestamp is deterministic and can be asserted
# exactly on every stored chunk.
_FIXED_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


# Feature: autonomous-cyber-defense-platform, Property 6: Ingested chunks are
# stored and tagged — For any readable knowledge source, every chunk stored by
# the ingestion pipeline SHALL carry the source identifier, an ingestion
# timestamp, and the source's category.
# Validates: Requirements 2.1, 2.2, 2.3
@settings(max_examples=100)
@given(source=knowledge_sources())
def test_ingested_chunks_are_stored_and_tagged(source: KnowledgeSource) -> None:
    store = InMemoryVectorStore()
    pipeline = IngestionPipeline(
        gateway=_StubGateway(),
        store=store,
        now=lambda: _FIXED_NOW,
    )

    result = pipeline.ingest(source)

    # Req 2.1: the source is split into chunks and their embeddings are stored.
    # A readable source yields at least one chunk, and the store holds exactly
    # the chunks reported by the ingestion result.
    stored = list(store._chunks.values())
    assert result.chunk_count >= 1
    assert len(stored) == result.chunk_count
    assert {c.chunk_id for c in stored} == set(result.chunk_ids)

    for chunk in stored:
        # Req 2.2: every stored chunk records the originating source identifier
        # and an ingestion timestamp.
        assert chunk.source_id == source.source_id
        assert chunk.ingested_at == _FIXED_NOW
        # Req 2.3: every stored chunk is tagged with the source's category.
        assert chunk.category == source.category
        # Req 2.1: an embedding vector is stored for each chunk.
        assert chunk.vector


# Feature: autonomous-cyber-defense-platform, Property 7: Ingestion is
# idempotent — For any knowledge source, ingesting it twice with unchanged
# content SHALL produce the same set of chunk identifiers and leave the vector
# store's chunk set for that source unchanged.
# Validates: Requirements 2.5
@settings(max_examples=100)
@given(source=knowledge_sources())
def test_ingestion_is_idempotent(source: KnowledgeSource) -> None:
    store = InMemoryVectorStore()
    pipeline = IngestionPipeline(
        gateway=_StubGateway(),
        store=store,
        now=lambda: _FIXED_NOW,
    )

    # First ingestion establishes the chunk set for the source.
    first = pipeline.ingest(source)
    first_ids = set(first.chunk_ids)
    first_snapshot = dict(store._chunks)

    # Re-ingesting unchanged content must produce the same set of chunk
    # identifiers (Req 2.5): chunk_ids are derived from source_id + index +
    # text, so identical content yields identical ids.
    second = pipeline.ingest(source)
    assert set(second.chunk_ids) == first_ids

    # The vector store's chunk set for that source is unchanged: upsert keyed by
    # deterministic chunk_id replaces rather than duplicates, so neither the set
    # of stored chunk ids nor their count grows on re-ingestion.
    assert set(store._chunks) == set(first_snapshot)
    assert len(store._chunks) == len(first_snapshot)


class _FailingGateway:
    """A stub :class:`~acdp.llm.LLMGateway` whose ``embed`` always fails.

    Models an unreadable/malformed source whose content cannot be embedded (for
    example, a source that reaches the embedding step but the gateway rejects
    it). Every ``embed`` call raises so we can assert the pipeline surfaces an
    :class:`~acdp.exceptions.IngestionError` without writing partial results.
    """

    def embed(self, text: str, model: str | None = None) -> list[float]:
        raise RuntimeError("unreadable/malformed source content")

    def generate(self, request):  # pragma: no cover - unused by ingestion
        raise NotImplementedError


# Req 2.4: IF an ingestion source is unreadable or malformed, THEN the RAG_Core
# SHALL reject the source and return a descriptive error identifying the source.
#
# An empty/whitespace-only source has no ingestible content: the pipeline must
# reject it with an IngestionError that names the source, and store nothing.
def test_empty_source_is_rejected_and_stores_nothing() -> None:
    store = InMemoryVectorStore()
    pipeline = IngestionPipeline(
        gateway=_StubGateway(),
        store=store,
        now=lambda: _FIXED_NOW,
    )
    source = KnowledgeSource(
        source_id="malformed-empty",
        category=SourceCategory.OTHER,
        content="   \t\n  ",  # whitespace only -> nothing ingestible
    )

    with pytest.raises(IngestionError) as exc_info:
        pipeline.ingest(source)

    # The raised error identifies the offending source (Req 2.4).
    assert exc_info.value.source_id == source.source_id
    assert source.source_id in str(exc_info.value)

    # Nothing was stored: rejection leaves no partial results.
    assert store._chunks == {}


# Req 2.4: a source that cannot be embedded (unreadable content the gateway
# rejects) must also be rejected with a source-identifying IngestionError, and
# must leave the vector store empty — embedding is all-or-nothing so a failure
# part-way through never persists a partial result.
def test_unreadable_source_is_rejected_and_stores_nothing() -> None:
    store = InMemoryVectorStore()
    pipeline = IngestionPipeline(
        gateway=_FailingGateway(),
        store=store,
        now=lambda: _FIXED_NOW,
    )
    source = KnowledgeSource(
        source_id="malformed-unreadable",
        category=SourceCategory.PLAYBOOK,
        content="some content that will fail to embed",
    )

    with pytest.raises(IngestionError) as exc_info:
        pipeline.ingest(source)

    # The raised error identifies the offending source (Req 2.4).
    assert exc_info.value.source_id == source.source_id
    assert source.source_id in str(exc_info.value)

    # No partial results were written to the store.
    assert store._chunks == {}
