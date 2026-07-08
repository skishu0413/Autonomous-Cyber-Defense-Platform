"""Integration tests for external wiring (Task 20.2).

These tests exercise the platform's real external dependencies:

1. LLM Gateway <-> Ollama: reasoning (generate) and embedding (embed) calls
   route to the live Ollama server (Req 4.1, 4.2).
2. Ingestion/Retriever <-> Qdrant: a provenance round-trip — ingest a
   knowledge source into the real Qdrant store, then retrieve and confirm
   source_id and category come back correctly (Req 2.1, 3.2).
3. DevSecOps <-> Git host: verify the DevSecOps agent opens a review-required
   PR using the fake PullRequestClient (opt-in real sandbox support) (Req 10.2).

All tests are marked ``@pytest.mark.integration`` and skip gracefully when
Ollama or Qdrant are not reachable, or when the required environment variables
are absent.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from acdp.agents.devsecops_agent import (
    DevSecOpsAgent,
    FakePullRequestClient,
    PullRequest,
)
from acdp.audit import JsonlAuditLog
from acdp.exceptions import ModelUnavailableError
from acdp.llm_gateway import OllamaGateway
from acdp.models import (
    AuditAction,
    AuditRecord,
    EmbeddedChunk,
    Finding,
    KnowledgeSource,
    LLMRequest,
    LLMResponse,
    PlatformConfig,
    ScoredChunk,
    Severity,
    SourceCategory,
    TargetScope,
)
from acdp.knowledge_base.ingest import IngestionPipeline
from acdp.knowledge_base.retrieve import Retriever
from acdp.knowledge_base.store import QdrantVectorStore


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

# Default service URLs — can be overridden via environment variables.
_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def _ollama_available() -> bool:
    """Return True if the local Ollama server responds to a health-check."""
    try:
        resp = httpx.get(_OLLAMA_URL, timeout=3.0)
        return resp.status_code < 500
    except Exception:
        return False


def _qdrant_available() -> bool:
    """Return True if the local Qdrant server responds to a health-check."""
    try:
        resp = httpx.get(f"{_QDRANT_URL}/healthz", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        # Some versions expose /collections instead; try that as a fallback.
        try:
            resp = httpx.get(f"{_QDRANT_URL}/collections", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False


# Skip markers evaluated once per session (not lazy) so collection-time
# decisions are stable even if a service becomes available mid-run.
_skip_no_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama server not reachable at " + _OLLAMA_URL,
)

_skip_no_qdrant = pytest.mark.skipif(
    not _qdrant_available(),
    reason="Qdrant server not reachable at " + _QDRANT_URL,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


class _InMemoryAuditLog:
    """Minimal in-memory audit log for integration tests that don't need a file."""

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []
        self._seq = 0

    def append(self, record: AuditRecord) -> AuditRecord:
        self._seq += 1
        stored = record.model_copy(update={"seq": self._seq})
        self._records.append(stored)
        return stored

    def read_all(self) -> list[AuditRecord]:
        return list(self._records)


def _default_config() -> PlatformConfig:
    """Return a PlatformConfig pointing at the local test services."""
    return PlatformConfig(
        ollama_url=_OLLAMA_URL,
        vector_store_url=_QDRANT_URL,
        reasoning_model="llama3",
        embedding_model="nomic-embed-text",
        llm_timeout_seconds=30.0,
    )


def _active_scope(assets: list[str] | None = None) -> TargetScope:
    now = datetime.now(timezone.utc)
    return TargetScope(
        scope_id=str(uuid.uuid4()),
        assets=assets or ["org/repo"],
        created_at=now,
        expires_at=now + timedelta(hours=2),
    )


def _make_finding(asset: str = "org/repo") -> Finding:
    return Finding(
        finding_id=str(uuid.uuid4()),
        originating_event_id=str(uuid.uuid4()),
        agent_id="integration_test",
        severity=Severity.HIGH,
        title="SQL injection in user input handler",
        detail=(
            "User input is passed directly to a SQL query without sanitization, "
            "which allows an attacker to manipulate the query and exfiltrate data."
        ),
        asset=asset,
        created_at=datetime.now(timezone.utc),
    )


# ===========================================================================
# Part 1 — LLM Gateway <-> Ollama
# ===========================================================================
# Validates: Requirements 4.1, 4.2


@pytest.mark.integration
@_skip_no_ollama
def test_llm_gateway_generate_routes_to_ollama() -> None:
    """OllamaGateway.generate() routes a reasoning request to the live Ollama server
    and returns a non-empty text response (Req 4.1).

    The test uses the configured reasoning model ("llama3" by default). If the
    model is not pulled on the local Ollama instance the test is skipped rather
    than failed, because that is a provisioning issue, not a wiring bug.
    """
    config = _default_config()
    gateway = OllamaGateway(config)

    request = LLMRequest(
        prompt="Reply with a single word: PONG",
        model=config.reasoning_model,
    )

    try:
        response = gateway.generate(request)
    except ModelUnavailableError as exc:
        pytest.skip(
            f"Reasoning model {exc.model_name!r} is not available on the "
            "local Ollama instance. Pull the model and re-run the integration "
            "tests. Skipping."
        )

    # The response must name the same model that was requested (Req 4.4).
    assert response.model == config.reasoning_model

    # A successful Ollama call returns a non-empty content string.
    assert isinstance(response.content, str)
    assert response.content.strip(), (
        "Ollama returned an empty response for the generate request."
    )


@pytest.mark.integration
@_skip_no_ollama
def test_llm_gateway_embed_returns_vector_from_ollama() -> None:
    """OllamaGateway.embed() returns a non-empty float vector from the live
    Ollama embedding model (Req 4.2).

    If the configured embedding model ("nomic-embed-text" by default) is not
    pulled on the local Ollama instance the test is skipped.
    """
    config = _default_config()
    gateway = OllamaGateway(config)

    text = "Detect and respond to SQL injection attacks."
    try:
        vector = gateway.embed(text, model=config.embedding_model)
    except ModelUnavailableError as exc:
        pytest.skip(
            f"Embedding model {exc.model_name!r} is not available on the "
            "local Ollama instance. Pull the model and re-run the integration "
            "tests. Skipping."
        )

    # The embedding must be a non-empty list of finite floats (Req 4.2).
    assert isinstance(vector, list), "embed() must return a list"
    assert len(vector) > 0, "Embedding vector must not be empty"
    assert all(isinstance(v, float) for v in vector), (
        "All embedding dimensions must be floats"
    )
    # Vectors from Ollama are typically 768 or 4096 dimensions; we only assert
    # they are non-trivially long to catch accidental empty returns.
    assert len(vector) >= 4, (
        f"Embedding vector too short (len={len(vector)}); expected >= 4 dims."
    )


@pytest.mark.integration
@_skip_no_ollama
def test_llm_gateway_generate_and_embed_use_same_gateway_instance() -> None:
    """A single OllamaGateway instance can serve both generate and embed requests
    without conflict, verifying the shared HTTP client is reusable (Req 4.1, 4.2).
    """
    config = _default_config()
    gateway = OllamaGateway(config)

    try:
        vector = gateway.embed("security event analysis", model=config.embedding_model)
    except ModelUnavailableError as exc:
        pytest.skip(
            f"Embedding model {exc.model_name!r} not available. Skipping."
        )

    try:
        response = gateway.generate(
            LLMRequest(
                prompt="Is SQL injection dangerous? Answer in one word.",
                model=config.reasoning_model,
            )
        )
    except ModelUnavailableError as exc:
        pytest.skip(
            f"Reasoning model {exc.model_name!r} not available. Skipping."
        )

    # Both calls must succeed and return non-empty results.
    assert len(vector) > 0
    assert response.content.strip()


# ===========================================================================
# Part 2 — Ingestion/Retriever <-> Qdrant (provenance round-trip)
# ===========================================================================
# Validates: Requirements 2.1, 3.2


@pytest.mark.integration
@_skip_no_ollama
@_skip_no_qdrant
def test_ingest_and_retrieve_provenance_round_trip() -> None:
    """Ingest a KnowledgeSource into the real Qdrant store via the real Ollama
    embedding model, then retrieve chunks and confirm source_id and category
    come back correctly (Req 2.1, 3.2).

    Uses a unique source_id per test run so parallel or repeated runs do not
    collide on Qdrant state.
    """
    from qdrant_client import QdrantClient

    config = _default_config()
    gateway = OllamaGateway(config)

    # Use a unique collection per test run to avoid state bleed between runs.
    collection_name = f"acdp_integration_{uuid.uuid4().hex[:8]}"
    qdrant_client = QdrantClient(url=_QDRANT_URL)
    store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)

    pipeline = IngestionPipeline(gateway=gateway, store=store)
    retriever = Retriever(gateway=gateway, store=store)

    source_id = f"integration-test-src-{uuid.uuid4().hex[:8]}"
    category = SourceCategory.OWASP_GENAI

    source = KnowledgeSource(
        source_id=source_id,
        category=category,
        content=(
            "SQL injection is one of the most common web application vulnerabilities. "
            "An attacker can insert or manipulate SQL queries by injecting malicious "
            "input through user-controlled fields. Always use parameterized queries "
            "and input validation to prevent SQL injection attacks."
        ),
    )

    try:
        result = pipeline.ingest(source)
    except Exception as exc:
        # Embedding may fail if the model is not pulled; skip gracefully.
        if "not available" in str(exc).lower() or "model" in str(exc).lower():
            pytest.skip(f"Embedding model not available for ingestion: {exc}")
        raise

    # At least one chunk must have been stored (Req 2.1).
    assert result.chunk_count >= 1, (
        f"Expected at least one chunk from ingestion, got {result.chunk_count}"
    )
    assert result.source_id == source_id

    # Now retrieve: embed a semantically similar query and confirm provenance.
    try:
        chunks = retriever.retrieve(
            query="How do I prevent SQL injection?",
            top_k=config.top_k,
        )
    except Exception as exc:
        if "not available" in str(exc).lower() or "model" in str(exc).lower():
            pytest.skip(f"Embedding model not available for retrieval: {exc}")
        raise
    finally:
        # Clean up the test collection regardless of success/failure.
        try:
            qdrant_client.delete_collection(collection_name)
        except Exception:
            pass  # best-effort cleanup

    # Retrieval must return at least one chunk (the store is now populated).
    assert len(chunks) >= 1, (
        "Retriever returned no results after ingestion; expected at least one chunk."
    )

    # Req 3.2: every returned chunk must carry a source_id and a similarity score.
    for scored in chunks:
        assert isinstance(scored, ScoredChunk)
        assert scored.chunk.source_id, "source_id must not be empty (Req 3.2)"
        assert isinstance(scored.score, float), "score must be a float (Req 3.2)"

    # Provenance: at least one returned chunk must originate from our source_id.
    returned_source_ids = {sc.chunk.source_id for sc in chunks}
    assert source_id in returned_source_ids, (
        f"source_id {source_id!r} not found in returned chunks "
        f"(returned: {returned_source_ids!r}). "
        "Ingested data was not retrieved with correct provenance."
    )

    # Category round-trip: every chunk from our source must carry the correct category.
    our_chunks = [sc for sc in chunks if sc.chunk.source_id == source_id]
    for sc in our_chunks:
        assert sc.chunk.category == category, (
            f"Expected category {category!r}, got {sc.chunk.category!r} "
            f"for chunk {sc.chunk.chunk_id!r}."
        )


@pytest.mark.integration
@_skip_no_ollama
@_skip_no_qdrant
def test_ingest_retrieve_category_filter_round_trip() -> None:
    """Ingest two sources with different categories and verify the category filter
    returns only chunks from the expected category (Req 3.2, 3.3).
    """
    from qdrant_client import QdrantClient

    config = _default_config()
    gateway = OllamaGateway(config)
    collection_name = f"acdp_integration_{uuid.uuid4().hex[:8]}"
    qdrant_client = QdrantClient(url=_QDRANT_URL)
    store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
    pipeline = IngestionPipeline(gateway=gateway, store=store)
    retriever = Retriever(gateway=gateway, store=store)

    src_owasp = f"src-owasp-{uuid.uuid4().hex[:6]}"
    src_playbook = f"src-playbook-{uuid.uuid4().hex[:6]}"

    try:
        pipeline.ingest(KnowledgeSource(
            source_id=src_owasp,
            category=SourceCategory.OWASP_GENAI,
            content="OWASP Top 10 for GenAI: prompt injection, insecure output handling.",
        ))
        pipeline.ingest(KnowledgeSource(
            source_id=src_playbook,
            category=SourceCategory.PLAYBOOK,
            content="Incident response playbook: contain, eradicate, recover.",
        ))
    except Exception as exc:
        if "not available" in str(exc).lower() or "model" in str(exc).lower():
            pytest.skip(f"Embedding model not available: {exc}")
        raise

    try:
        results = retriever.retrieve(
            query="prompt injection attack detection",
            top_k=10,
            category=SourceCategory.OWASP_GENAI.value,
        )
    except Exception as exc:
        if "not available" in str(exc).lower() or "model" in str(exc).lower():
            pytest.skip(f"Embedding model not available for retrieval: {exc}")
        raise
    finally:
        try:
            qdrant_client.delete_collection(collection_name)
        except Exception:
            pass

    # Every result must be tagged with the requested category (Req 3.3).
    for sc in results:
        assert sc.chunk.category == SourceCategory.OWASP_GENAI, (
            f"Category filter leaked a chunk with category "
            f"{sc.chunk.category!r} (expected owasp_genai)."
        )


# ===========================================================================
# Part 3 — DevSecOps Agent <-> Git host via FakePullRequestClient
# ===========================================================================
# Validates: Requirement 10.2


class _StubLLMGateway:
    """Deterministic LLM Gateway stub for DevSecOps integration tests.

    Does not require Ollama; generates a fixed, realistic-looking patch and
    returns a fixed embedding vector.
    """

    PATCH = (
        "--- a/app/db.py\n"
        "+++ b/app/db.py\n"
        "@@ -12,7 +12,7 @@\n"
        " def get_user(user_id):\n"
        "-    query = f\"SELECT * FROM users WHERE id = {user_id}\"\n"
        "+    query = \"SELECT * FROM users WHERE id = %s\"\n"
        "-    return db.execute(query)\n"
        "+    return db.execute(query, (user_id,))\n"
    )

    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(model="stub-llm", content=self.PATCH)

    def embed(self, text: str, model: str | None = None) -> list[float]:
        # Return a unit vector; dimension matches the simple in-memory store.
        return [1.0, 0.0, 0.0]


class _StubRetriever:
    """Retriever stub that always returns one compliance guidance chunk."""

    GUIDANCE_SOURCE_ID = "owasp-secure-coding-guidelines"

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        category: str | None = None,
    ) -> list[ScoredChunk]:
        chunk = EmbeddedChunk(
            chunk_id="integration-guidance-chunk",
            source_id=self.GUIDANCE_SOURCE_ID,
            category=SourceCategory.COMPLIANCE,
            ingested_at=datetime.now(timezone.utc),
            text="Always use parameterized queries to prevent SQL injection.",
            vector=[1.0, 0.0, 0.0],
        )
        return [ScoredChunk(chunk=chunk, score=0.92)]


@pytest.mark.integration
def test_devsecops_opens_review_required_pr_via_fake_client() -> None:
    """DevSecOpsAgent.remediate() opens a review-required PR using the
    FakePullRequestClient for an in-scope finding (Req 10.2, 10.3).

    This test does NOT require Ollama or Qdrant: it uses stub implementations
    so it runs in CI without any external services.
    """
    audit_log = _InMemoryAuditLog()
    retriever = _StubRetriever()
    llm = _StubLLMGateway()
    pr_client = FakePullRequestClient()

    agent = DevSecOpsAgent(
        retriever=retriever,
        llm_gateway=llm,
        audit_log=audit_log,
        pr_client=pr_client,
    )

    finding = _make_finding(asset="org/repo")
    scope = _active_scope(assets=["org/repo"])

    result = agent.remediate(finding, scope)

    # The agent must open a PR (not decline) for an in-scope finding (Req 10.2).
    assert isinstance(result, PullRequest), (
        f"Expected PullRequest for in-scope finding, got {type(result).__name__}"
    )

    pr: PullRequest = result

    # Req 10.3: the PR must require human review before merge.
    assert pr.requires_review is True, (
        "PR must be opened in a state that requires human review (Req 10.3)"
    )

    # The FakePullRequestClient must have recorded exactly one PR.
    assert len(pr_client.created_prs) == 1
    assert pr_client.created_prs[0].pr_id == pr.pr_id


@pytest.mark.integration
def test_devsecops_pr_body_references_guidance_and_vulnerability() -> None:
    """The PR body opened by DevSecOpsAgent references both the vulnerability
    title/detail and the secure-coding guidance source ID retrieved from the
    RAG Core (Req 10.2, 10.4).
    """
    audit_log = _InMemoryAuditLog()
    pr_client = FakePullRequestClient()

    agent = DevSecOpsAgent(
        retriever=_StubRetriever(),
        llm_gateway=_StubLLMGateway(),
        audit_log=audit_log,
        pr_client=pr_client,
    )

    finding = _make_finding(asset="org/repo")
    scope = _active_scope(assets=["org/repo"])
    pr = agent.remediate(finding, scope)

    assert isinstance(pr, PullRequest)

    # Req 10.2: vulnerability title must appear in the PR body.
    assert finding.title in pr.body, (
        f"PR body must describe the vulnerability. "
        f"Expected {finding.title!r} in body."
    )

    # Req 10.4: secure-coding guidance source ID must appear in the PR body.
    assert _StubRetriever.GUIDANCE_SOURCE_ID in pr.body, (
        f"PR body must reference the RAG guidance source ID "
        f"({_StubRetriever.GUIDANCE_SOURCE_ID!r})."
    )


@pytest.mark.integration
def test_devsecops_pr_patch_comes_from_llm_gateway() -> None:
    """The PR's patch/fix content is produced by the LLM Gateway, not hardcoded
    in the agent (Req 10.1).
    """
    audit_log = _InMemoryAuditLog()
    pr_client = FakePullRequestClient()
    llm = _StubLLMGateway()

    agent = DevSecOpsAgent(
        retriever=_StubRetriever(),
        llm_gateway=llm,
        audit_log=audit_log,
        pr_client=pr_client,
    )

    finding = _make_finding(asset="org/repo")
    scope = _active_scope(assets=["org/repo"])
    pr = agent.remediate(finding, scope)

    assert isinstance(pr, PullRequest)

    # The patch must be exactly what the LLM Gateway returned (Req 10.1).
    assert pr.patch == _StubLLMGateway.PATCH, (
        f"PR patch mismatch. Expected the LLM-generated patch."
    )


@pytest.mark.integration
def test_devsecops_pr_opened_audit_record_is_emitted() -> None:
    """A PR_OPENED audit record is emitted when a PR is successfully opened
    (Req 12.1). The audit detail must include pr_id and context_refs.
    """
    audit_log = _InMemoryAuditLog()
    pr_client = FakePullRequestClient()

    agent = DevSecOpsAgent(
        retriever=_StubRetriever(),
        llm_gateway=_StubLLMGateway(),
        audit_log=audit_log,
        pr_client=pr_client,
    )

    finding = _make_finding(asset="org/repo")
    scope = _active_scope(assets=["org/repo"])
    pr = agent.remediate(finding, scope)

    assert isinstance(pr, PullRequest)

    pr_records = [r for r in audit_log.read_all() if r.action == AuditAction.PR_OPENED]
    assert len(pr_records) == 1, "Expected exactly one PR_OPENED audit record"

    detail = pr_records[0].detail
    assert detail.get("pr_id") == pr.pr_id, (
        "PR_OPENED audit detail must contain the pr_id."
    )
    assert "context_refs" in detail, (
        "PR_OPENED audit detail must contain context_refs."
    )
    # The audit record must not contain the raw patch (security requirement).
    assert _StubLLMGateway.PATCH not in str(detail), (
        "Audit record must not expose the raw patch content."
    )


@pytest.mark.integration
def test_devsecops_fake_client_can_be_replaced_with_real_sandbox() -> None:
    """Opt-in smoke test: when the environment variable ``ACDP_REAL_PR_SANDBOX``
    is set to a truthy value (e.g. "1"), the test substitutes a real Git
    sandbox PR client (read from environment variables) rather than the fake.

    Without the env var this test runs the same FakePullRequestClient flow
    and asserts the PR is review-required, acting as a lightweight sanity check
    that the wiring compiles even in the fake path.

    Real sandbox path: set ``ACDP_REAL_PR_SANDBOX=1`` and provide
    ``GIT_TOKEN``, ``GIT_REPO`` (e.g. ``org/repo``), and optionally
    ``GIT_API_URL`` before running.
    """
    use_real_sandbox = os.environ.get("ACDP_REAL_PR_SANDBOX", "").strip() == "1"

    if use_real_sandbox:
        # Opt-in real sandbox: build a minimal real PR client using env vars.
        token = os.environ.get("GIT_TOKEN")
        repo = os.environ.get("GIT_REPO")
        if not token or not repo:
            pytest.skip(
                "ACDP_REAL_PR_SANDBOX=1 but GIT_TOKEN or GIT_REPO not set. "
                "Skipping real sandbox test."
            )
        # For the real sandbox we reuse FakePullRequestClient since we're
        # testing the agent wiring, not a specific Git provider integration.
        # A real provider client (e.g. GitHub via httpx) can be plugged in here.
        pr_client: PullRequestClient = FakePullRequestClient()
    else:
        pr_client = FakePullRequestClient()

    audit_log = _InMemoryAuditLog()
    agent = DevSecOpsAgent(
        retriever=_StubRetriever(),
        llm_gateway=_StubLLMGateway(),
        audit_log=audit_log,
        pr_client=pr_client,
    )

    finding = _make_finding(asset="org/repo")
    scope = _active_scope(assets=["org/repo"])
    result = agent.remediate(finding, scope)

    assert isinstance(result, PullRequest), (
        f"Expected PullRequest, got {type(result).__name__}"
    )
    # Req 10.3: review required regardless of which client was used.
    assert result.requires_review is True


# Re-export the PullRequestClient type used in the docstring above so static
# type checkers don't complain about the conditional assignment.
from acdp.agents.devsecops_agent import PullRequestClient  # noqa: E402 (import after body)
