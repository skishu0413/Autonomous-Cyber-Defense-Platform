"""Semantic retrieval over the Vector Store (Layer 2 — Enterprise RAG Core).

The :class:`Retriever` is the read side of the RAG Core. Given a natural-language
query it embeds the query through the :class:`~acdp.llm.LLMGateway` (using the
same embedding model as ingestion, so query and chunk vectors live in the same
space) and searches the :class:`~acdp.rag.store.VectorStore` for the most
similar chunks:

    query -> embed (via the LLM Gateway) -> search (Vector Store) -> ranked chunks

Results are the top-K most similar :class:`~acdp.models.ScoredChunk` objects,
each carrying its originating ``source_id`` and similarity ``score`` (Req 3.2),
ordered by descending similarity (Req 3.5). ``K`` is configurable: an explicit
``top_k`` argument wins, otherwise the retriever falls back to its configured
default (Req 3.1). An optional ``category`` filter restricts results to a single
source category (Req 3.3).

Retrieval never raises for an empty result set: when the store holds no matching
chunks (empty store, or nothing tagged with the requested category) the
retriever returns ``[]`` (Req 3.4).

The retriever depends only on the narrow :class:`~acdp.llm.LLMGateway` and
:class:`~acdp.rag.store.VectorStore` interfaces, so it can be exercised with the
in-memory store fake and a stub gateway without any live services.
"""

from __future__ import annotations

from acdp.llm_gateway import LLMGateway
from acdp.models import ScoredChunk
from acdp.knowledge_base.store import VectorStore

__all__ = ["Retriever", "DEFAULT_TOP_K"]

# Default number of chunks to return when a caller does not specify ``top_k``.
# Mirrors :attr:`~acdp.models.PlatformConfig.top_k` (Req 3.1) so the retriever's
# standalone default matches the platform-wide configured default.
DEFAULT_TOP_K = 5


class Retriever:
    """Embed queries and return ranked context from the Vector Store (Req 3.1-3.5)."""

    def __init__(
        self,
        gateway: LLMGateway,
        store: VectorStore,
        *,
        default_top_k: int = DEFAULT_TOP_K,
        embedding_model: str | None = None,
    ) -> None:
        """Create a retriever.

        Args:
            gateway: LLM Gateway used to embed the query (Req 3.1). The same
                embedding model used at ingestion should be used here so query
                and chunk vectors are comparable.
            store: Vector Store searched for similar chunks (Req 3.1).
            default_top_k: Number of chunks to return when a ``retrieve`` call
                does not pass ``top_k``. Must be positive; mirrors the
                platform's configured ``top_k`` (Req 3.1).
            embedding_model: Optional named embedding model; when ``None`` the
                gateway's configured embedding model is used.
        """
        if default_top_k <= 0:
            raise ValueError(f"default_top_k must be positive, got {default_top_k}")
        self._gateway = gateway
        self._store = store
        self._default_top_k = default_top_k
        self._embedding_model = embedding_model

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        category: str | None = None,
    ) -> list[ScoredChunk]:
        """Return the top-K chunks most similar to ``query``, ranked descending.

        Embeds ``query`` via the LLM Gateway and searches the Vector Store for
        the most similar chunks (Req 3.1). Each returned
        :class:`~acdp.models.ScoredChunk` carries its originating ``source_id``
        and similarity ``score`` (Req 3.2), and results are ordered by
        descending similarity (Req 3.5).

        Args:
            query: The natural-language query to search for.
            top_k: Maximum number of chunks to return. When ``None`` the
                configured ``default_top_k`` is used (Req 3.1).
            category: Optional source-category filter; when provided, only
                chunks tagged with that category are returned (Req 3.3).

        Returns:
            Up to ``top_k`` scored chunks ordered by descending similarity, or
            an empty list when nothing matches. Never raises for an empty result
            set (Req 3.4).
        """
        effective_top_k = self._default_top_k if top_k is None else top_k

        # Embed the query in the same space as the stored chunk vectors so
        # similarity is meaningful (Req 3.1).
        vector = self._gateway.embed(query, model=self._embedding_model)

        # The Vector Store enforces similarity ranking and category filtering and
        # returns an empty list (never raises) when nothing matches, so empty
        # results propagate naturally as ``[]`` (Req 3.3, 3.4, 3.5).
        return self._store.search(vector, effective_top_k, category)
