"""Property-based tests for the semantic Retriever (Req 3.1, 3.2, 3.5).

These tests exercise :class:`~acdp.rag.retrieve.Retriever` against the in-memory
:class:`~acdp.rag.store.InMemoryVectorStore` fake and a deterministic stub LLM
Gateway, so no live Ollama/Qdrant services are required. The stub gateway
returns a fixed query vector so retrieval is fully reproducible.
"""

from __future__ import annotations

import math
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from acdp.models import EmbeddedChunk, ScoredChunk
from acdp.knowledge_base.retrieve import Retriever
from acdp.knowledge_base.store import InMemoryVectorStore, _cosine_similarity
from tests.strategies import category_filtered_retrieval_scenarios, retrieval_scenarios


class _FixedVectorGateway:
    """A stub :class:`~acdp.llm.LLMGateway` that embeds any query to a fixed vector.

    Retrieval's only use of the gateway is to embed the query; returning a
    predetermined vector makes the similarity ranking deterministic and lets the
    test reason about the expected top-K directly. ``generate`` is provided only
    to satisfy the interface and is never called here.
    """

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed(self, text: str, model: str | None = None) -> list[float]:
        return list(self._vector)

    def generate(self, request: Any):  # pragma: no cover - unused by retrieval
        raise NotImplementedError


def _expected_top_k_scores(
    query_vector: list[float], chunks: list[EmbeddedChunk], top_k: int
) -> list[float]:
    """Return the descending similarity scores of the true top-K chunks."""
    scores = sorted(
        (_cosine_similarity(query_vector, chunk.vector) for chunk in chunks),
        reverse=True,
    )
    return scores[:top_k]


# Feature: autonomous-cyber-defense-platform, Property 8: Retrieval returns
# ranked top-K with provenance — For any populated vector store, query, and
# configurable K, the retriever SHALL return at most K chunks, all drawn from
# the K highest-similarity chunks, each annotated with a source identifier and
# similarity score, ordered by descending similarity.
# Validates: Requirements 3.1, 3.2, 3.5
@settings(max_examples=100)
@given(scenario=retrieval_scenarios(), query=st.text(max_size=40))
def test_retrieval_returns_ranked_top_k_with_provenance(
    scenario: dict[str, Any], query: str
) -> None:
    query_vector: list[float] = scenario["query_vector"]
    chunks: list[EmbeddedChunk] = scenario["chunks"]
    top_k: int = scenario["top_k"]

    store = InMemoryVectorStore()
    store.upsert(chunks)
    retriever = Retriever(gateway=_FixedVectorGateway(query_vector), store=store)

    results = retriever.retrieve(query, top_k=top_k)

    # Every result is a ScoredChunk carrying provenance (Req 3.2): a source
    # identifier and a similarity score.
    for result in results:
        assert isinstance(result, ScoredChunk)
        assert result.chunk.source_id != ""
        assert isinstance(result.score, float)

    # (a) At most K chunks are returned, and since the store is populated we get
    # exactly min(K, stored) chunks (Req 3.1).
    assert len(results) <= top_k
    assert len(results) == min(top_k, len(chunks))

    scores = [r.score for r in results]

    # (d) Results are ordered by descending similarity score (Req 3.5).
    assert scores == sorted(scores, reverse=True)

    # (b) The returned chunks are exactly the K highest-similarity chunks: the
    # multiset of returned scores equals the true top-K scores. Comparing the
    # sorted score sequences is tie-safe (chunks with equal scores are
    # interchangeable for the "top-K" contract).
    expected = _expected_top_k_scores(query_vector, chunks, top_k)
    assert len(scores) == len(expected)
    for got, want in zip(scores, expected):
        assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12)


# Feature: autonomous-cyber-defense-platform, Property 9: Category-filtered
# retrieval returns only that category — For any query with a source-category
# filter, every returned chunk SHALL be tagged with the requested category.
# Validates: Requirements 3.3
@settings(max_examples=100)
@given(scenario=category_filtered_retrieval_scenarios(), query=st.text(max_size=40))
def test_category_filtered_retrieval_returns_only_that_category(
    scenario: dict[str, Any], query: str
) -> None:
    query_vector: list[float] = scenario["query_vector"]
    chunks: list[EmbeddedChunk] = scenario["chunks"]
    category = scenario["category"]
    top_k: int = scenario["top_k"]

    store = InMemoryVectorStore()
    store.upsert(chunks)
    retriever = Retriever(gateway=_FixedVectorGateway(query_vector), store=store)

    results = retriever.retrieve(query, top_k=top_k, category=category)

    # Property 9 (Req 3.3): every returned chunk is tagged with exactly the
    # requested category — the filter never leaks a chunk of another category.
    for result in results:
        assert isinstance(result, ScoredChunk)
        assert result.chunk.category == category

    # The filter is faithful in both directions: the returned set is precisely
    # the top-K over the chunks that carry the requested category (it drops no
    # matching chunk that should have ranked in, and admits no non-matching one).
    matching = [c for c in chunks if c.category == category]
    assert len(results) == min(top_k, len(matching))

    store = InMemoryVectorStore()
    retriever = Retriever(gateway=_FixedVectorGateway([1.0, 0.0, 0.0]), store=store)

    assert retriever.retrieve("anything", top_k=5) == []
