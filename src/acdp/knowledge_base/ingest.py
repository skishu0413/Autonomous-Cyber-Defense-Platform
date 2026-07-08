"""Ingestion pipeline (Layer 2 — Enterprise RAG Core).

The :class:`IngestionPipeline` turns a raw :class:`~acdp.models.KnowledgeSource`
into embedded chunks stored in the :class:`~acdp.rag.store.VectorStore`:

    chunk -> embed (via the LLM Gateway) -> upsert

Every stored chunk is tagged with its originating ``source_id``, an ingestion
timestamp, and the source's ``category`` (Req 2.1, 2.2, 2.3). Each ``chunk_id``
is derived deterministically from ``source_id + chunk_index + chunk_text`` using
a stable cryptographic hash, so re-ingesting unchanged content yields the same
chunk identifiers and the upsert replaces rather than duplicates them — making
ingestion idempotent (Req 2.5). A stable hash (SHA-256) is used deliberately
rather than the builtin :func:`hash`, whose salt is randomized per process and
would break cross-run idempotence.

Ingestion is all-or-nothing: chunks are embedded in full before anything is
written, so a source that is unreadable or malformed raises
:class:`~acdp.exceptions.IngestionError` naming the source without leaving
partial results in the store (Req 2.4).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime, timezone

from acdp.exceptions import IngestionError
from acdp.llm_gateway import LLMGateway
from acdp.models import EmbeddedChunk, KnowledgeSource
from acdp.knowledge_base.store import VectorStore

__all__ = ["IngestionResult", "IngestionPipeline", "DEFAULT_CHUNK_SIZE"]

# Default target chunk size in characters. Chosen as a small, embedding-friendly
# window; the pipeline splits on whitespace boundaries so chunks stay close to
# this size without cutting words.
DEFAULT_CHUNK_SIZE = 512


class IngestionResult:
    """The outcome of ingesting a single knowledge source.

    Carries the source identifier, the deterministic chunk ids that were stored,
    and the ingestion timestamp shared by those chunks. Because ``chunk_ids`` is
    derived only from the source id and chunk content (not the timestamp),
    re-ingesting unchanged content produces an equal ``chunk_ids`` list, which
    is the observable form of idempotence (Req 2.5).
    """

    def __init__(
        self,
        source_id: str,
        chunk_ids: list[str],
        ingested_at: datetime,
    ) -> None:
        self.source_id = source_id
        self.chunk_ids = chunk_ids
        self.ingested_at = ingested_at

    @property
    def chunk_count(self) -> int:
        """Number of chunks stored for the source."""
        return len(self.chunk_ids)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"IngestionResult(source_id={self.source_id!r}, "
            f"chunk_count={self.chunk_count})"
        )


class IngestionPipeline:
    """Chunk, embed, and store knowledge sources into the Vector Store.

    The pipeline depends only on the narrow :class:`~acdp.llm.LLMGateway` and
    :class:`~acdp.rag.store.VectorStore` interfaces, so it can be exercised with
    the in-memory store fake and a stub gateway without any live services.
    """

    def __init__(
        self,
        gateway: LLMGateway,
        store: VectorStore,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        embedding_model: str | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        """Create an ingestion pipeline.

        Args:
            gateway: LLM Gateway used to embed chunk text (Req 4.2).
            store: Vector Store that chunks are upserted into (Req 2.1).
            chunk_size: Target chunk size in characters. Must be positive.
            embedding_model: Optional named embedding model; when ``None`` the
                gateway's configured embedding model is used.
            now: Optional clock returning the ingestion timestamp. Injectable so
                tests can pin the timestamp; defaults to UTC ``datetime.now``.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self._gateway = gateway
        self._store = store
        self._chunk_size = chunk_size
        self._embedding_model = embedding_model
        self._now = now or (lambda: datetime.now(timezone.utc))

    def ingest(self, source: KnowledgeSource) -> IngestionResult:
        """Chunk, embed, and store ``source``; return the ingestion result.

        The source content is split into chunks, each chunk is embedded via the
        LLM Gateway, and the resulting :class:`~acdp.models.EmbeddedChunk` set is
        upserted in a single call so re-ingestion is idempotent (Req 2.1, 2.5).
        Every chunk is tagged with the source id, an ingestion timestamp, and
        the source category (Req 2.2, 2.3).

        Raises:
            IngestionError: If the source is unreadable/malformed (e.g. empty or
                whitespace-only content) or embedding fails. Raised before any
                upsert so no partial results are stored (Req 2.4).
        """
        texts = self._chunk(source.content)
        if not texts:
            # An empty or whitespace-only source has nothing to embed or store;
            # treat it as malformed rather than silently storing nothing.
            raise IngestionError(
                source.source_id,
                f"Knowledge source {source.source_id!r} has no ingestible content",
            )

        ingested_at = self._now()

        # Embed everything first. Building the full chunk set before touching the
        # store keeps ingestion all-or-nothing: if any embed call fails we raise
        # without having written a partial result (Req 2.4).
        chunks: list[EmbeddedChunk] = []
        for index, text in enumerate(texts):
            try:
                vector = self._gateway.embed(text, model=self._embedding_model)
            except Exception as exc:
                raise IngestionError(
                    source.source_id,
                    f"Failed to embed chunk {index} of source "
                    f"{source.source_id!r}: {exc}",
                ) from exc

            chunks.append(
                EmbeddedChunk(
                    chunk_id=self._chunk_id(source.source_id, index, text),
                    source_id=source.source_id,
                    category=source.category,
                    ingested_at=ingested_at,
                    text=text,
                    vector=vector,
                )
            )

        self._store.upsert(chunks)

        return IngestionResult(
            source_id=source.source_id,
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            ingested_at=ingested_at,
        )

    def _chunk(self, content: str) -> list[str]:
        """Split ``content`` into whitespace-delimited chunks near ``chunk_size``.

        Splitting is deterministic and boundary-aware: tokens (whitespace-
        separated words) are packed greedily up to ``chunk_size`` characters so
        the same content always yields the same chunk texts — a prerequisite for
        deterministic ``chunk_id`` derivation and idempotent ingestion (Req 2.5).
        A single token longer than ``chunk_size`` is emitted on its own rather
        than being dropped or split mid-token.
        """
        tokens = content.split()
        if not tokens:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for token in tokens:
            # +1 accounts for the single space that will join this token to the
            # existing ones in the chunk.
            added_len = len(token) if not current else current_len + 1 + len(token)
            if current and added_len > self._chunk_size:
                chunks.append(" ".join(current))
                current = [token]
                current_len = len(token)
            else:
                current.append(token)
                current_len = added_len
        if current:
            chunks.append(" ".join(current))
        return chunks

    @staticmethod
    def _chunk_id(source_id: str, index: int, text: str) -> str:
        """Derive a stable ``chunk_id`` from ``source_id + index + text``.

        Uses SHA-256 over a delimiter-joined key so the id is a pure function of
        its inputs and stable across processes and runs. The delimiter (a NUL
        byte) prevents ambiguity between fields (e.g. so ``("a", 1, "b")`` and
        ``("a1", "", "b")`` cannot collide). Determinism here is what makes
        re-ingestion idempotent under upsert (Req 2.5).
        """
        key = "\x00".join([source_id, str(index), text])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()
