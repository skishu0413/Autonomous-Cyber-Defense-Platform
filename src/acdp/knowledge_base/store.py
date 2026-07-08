"""Vector Store interface and adapters (Layer 2 — Enterprise RAG Core).

The :class:`VectorStore` protocol is the storage boundary for the RAG Core. It
is deliberately abstract so that the ingestion pipeline and retriever depend on
behavior, not on Qdrant specifics — enabling a fast, deterministic in-memory
fake for property/unit tests while a real Qdrant instance backs production and
integration tests (design: "The ``Vector_Store`` interface is abstract, so
Milvus remains a drop-in alternative").

Two adapters are provided:

* :class:`InMemoryVectorStore` — a dependency-free fake that ranks by
  descending cosine similarity and supports ``source_category`` payload
  filtering. Used by property/unit tests.
* :class:`QdrantVectorStore` — the production adapter backed by a single Qdrant
  collection. ``source_category`` and ``source_id`` are stored as payload
  fields to support category filtering (Req 3.3) and re-ingestion by source
  (Req 2.5).

Chunks are keyed by their deterministic ``chunk_id`` so re-ingesting unchanged
content replaces rather than duplicates a chunk (Req 2.5). Search returns up to
``top_k`` chunks ordered by descending similarity, optionally filtered by source
category (Req 3.1, 3.3, 3.5).
"""

from __future__ import annotations

import math
import uuid
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from acdp.models import EmbeddedChunk, ScoredChunk, SourceCategory

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from qdrant_client import QdrantClient

__all__ = ["VectorStore", "InMemoryVectorStore", "QdrantVectorStore"]


@runtime_checkable
class VectorStore(Protocol):
    """Storage boundary for embedded knowledge chunks (Req 3.1, 3.3, 3.5)."""

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        """Insert or replace chunks keyed by deterministic ``chunk_id``.

        Re-upserting a chunk with an existing ``chunk_id`` replaces it rather
        than creating a duplicate, which makes re-ingestion idempotent (Req 2.5).
        """
        ...

    def search(
        self,
        vector: list[float],
        top_k: int,
        category: str | None = None,
    ) -> list[ScoredChunk]:
        """Return up to ``top_k`` chunks ranked by descending similarity.

        When ``category`` is provided, only chunks tagged with that source
        category are considered (Req 3.3). Returns an empty list when nothing
        matches; it never raises for an empty result set (Req 3.4).
        """
        ...

    def delete_by_source(self, source_id: str) -> None:
        """Remove every chunk whose ``source_id`` matches (Req 2.5 re-ingestion)."""
        ...


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors; higher means closer.

    Returns ``0.0`` when either vector has zero magnitude (no meaningful
    direction to compare) so ranking stays well-defined for degenerate inputs.
    """
    if len(a) != len(b):
        raise ValueError(
            f"vector dimension mismatch: query has {len(a)}, chunk has {len(b)}"
        )

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class InMemoryVectorStore:
    """An in-memory :class:`VectorStore` fake for property/unit tests.

    Chunks are held in an insertion-ordered mapping keyed by ``chunk_id`` so
    upserts replace by key (Req 2.5) and iteration order is stable. Search ranks
    every stored chunk by descending cosine similarity and returns the top
    ``top_k`` after optional source-category filtering (Req 3.1, 3.3, 3.5).
    """

    def __init__(self) -> None:
        self._chunks: dict[str, EmbeddedChunk] = {}

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        """Insert or replace each chunk keyed by ``chunk_id`` (Req 2.5)."""
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def search(
        self,
        vector: list[float],
        top_k: int,
        category: str | None = None,
    ) -> list[ScoredChunk]:
        """Return the ``top_k`` most similar chunks, descending (Req 3.1, 3.3, 3.5)."""
        if top_k <= 0:
            return []

        wanted = self._normalize_category(category)

        scored: list[ScoredChunk] = []
        for chunk in self._chunks.values():
            if wanted is not None and chunk.category != wanted:
                continue
            score = _cosine_similarity(vector, chunk.vector)
            scored.append(ScoredChunk(chunk=chunk, score=score))

        # Descending similarity (Req 3.5). ``sorted`` is stable, so chunks with
        # equal scores keep their insertion order for deterministic results.
        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:top_k]

    def delete_by_source(self, source_id: str) -> None:
        """Drop every chunk originating from ``source_id`` (Req 2.5)."""
        self._chunks = {
            cid: chunk
            for cid, chunk in self._chunks.items()
            if chunk.source_id != source_id
        }

    @staticmethod
    def _normalize_category(category: str | None) -> SourceCategory | None:
        """Coerce a category filter (enum or raw string) to :class:`SourceCategory`."""
        if category is None:
            return None
        if isinstance(category, SourceCategory):
            return category
        return SourceCategory(category)


# Namespace for deriving stable Qdrant point UUIDs from deterministic chunk_ids.
_POINT_ID_NAMESPACE = uuid.UUID("6b3f5c8a-3f7a-4a2e-9d1b-2c4e5f6a7b8c")


class QdrantVectorStore:
    """A Qdrant-backed :class:`VectorStore` using a single collection.

    Each :class:`EmbeddedChunk` becomes one point. The point id is a stable
    UUID derived from the chunk's deterministic ``chunk_id`` (via ``uuid5``), so
    re-upserting unchanged content overwrites the same point rather than
    duplicating it (Req 2.5). ``source_category`` and ``source_id`` are stored
    as payload fields to support category filtering (Req 3.3) and deletion by
    source (Req 2.5); the full chunk is stored under ``chunk`` so results can be
    rehydrated into :class:`ScoredChunk` without a second lookup.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "acdp_knowledge",
    ) -> None:
        self._client = client
        self._collection = collection_name

    def _ensure_collection(self, vector_size: int) -> None:
        """Create the backing collection on first use if it does not exist."""
        from qdrant_client import models as qmodels

        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        """Insert or replace ``chunks`` as points keyed by chunk_id (Req 2.5)."""
        if not chunks:
            return

        from qdrant_client import models as qmodels

        self._ensure_collection(vector_size=len(chunks[0].vector))

        points = [
            qmodels.PointStruct(
                id=self._point_id(chunk.chunk_id),
                vector=chunk.vector,
                payload={
                    "source_category": chunk.category.value,
                    "source_id": chunk.source_id,
                    "chunk": chunk.model_dump(mode="json"),
                },
            )
            for chunk in chunks
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(
        self,
        vector: list[float],
        top_k: int,
        category: str | None = None,
    ) -> list[ScoredChunk]:
        """Return up to ``top_k`` chunks by descending similarity (Req 3.1, 3.3, 3.5)."""
        if top_k <= 0 or not self._client.collection_exists(self._collection):
            return []

        from qdrant_client import models as qmodels

        query_filter: qmodels.Filter | None = None
        if category is not None:
            wanted = category.value if isinstance(category, SourceCategory) else category
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source_category",
                        match=qmodels.MatchValue(value=wanted),
                    )
                ]
            )

        hits = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points

        results: list[ScoredChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            chunk = EmbeddedChunk.model_validate(payload["chunk"])
            results.append(ScoredChunk(chunk=chunk, score=hit.score))
        return results

    def delete_by_source(self, source_id: str) -> None:
        """Delete every point tagged with ``source_id`` (Req 2.5)."""
        if not self._client.collection_exists(self._collection):
            return

        from qdrant_client import models as qmodels

        self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="source_id",
                            match=qmodels.MatchValue(value=source_id),
                        )
                    ]
                )
            ),
        )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Derive a stable UUID point id from a deterministic ``chunk_id``."""
        return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))
