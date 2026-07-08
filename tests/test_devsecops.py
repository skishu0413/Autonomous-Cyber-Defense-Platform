"""Unit tests for the DevSecOps Agent (Task 17.1).

Covers:
- FakePullRequestClient stores created PRs and always sets requires_review=True (Req 10.3)
- DevSecOpsAgent.remediate() opens a review-required PR for in-scope repos (Req 10.1-10.4)
- The PR body references secure-coding guidance source IDs from RAG (Req 10.4)
- The PR_OPENED audit record contains pr_id and context_refs (Req 12.1)
- Out-of-scope findings are declined with a REMEDIATION_DECLINED audit record (Req 10.5)
- Expired scopes are treated as out-of-scope (Req 10.5)
- Revoked scopes are treated as out-of-scope (Req 10.5)
- Findings with no asset (None) are declined (Req 10.5)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from acdp.agents.devsecops_agent import (
    Declination,
    DevSecOpsAgent,
    FakePullRequestClient,
    PullRequest,
)
from acdp.models import (
    AuditAction,
    AuditRecord,
    EmbeddedChunk,
    Finding,
    LLMRequest,
    LLMResponse,
    ScoredChunk,
    Severity,
    SourceCategory,
    TargetScope,
)


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class _InMemoryAuditLog:
    """Simple in-memory audit log stub for tests."""

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


class _StubRetriever:
    """Retriever stub returning a fixed list of chunks."""

    def __init__(self, chunks: list[ScoredChunk] | None = None) -> None:
        self._chunks = chunks or []

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        category: str | None = None,
    ) -> list[ScoredChunk]:
        if category is None:
            return list(self._chunks)
        return [c for c in self._chunks if c.chunk.category.value == category]


class _StubLLMGateway:
    """LLM Gateway stub returning a fixed response."""

    def __init__(self, patch_content: str = "--- a/file.py\n+++ b/file.py\n-bad\n+good") -> None:
        self._patch = patch_content
        self.calls: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(model="stub", content=self._patch)

    def embed(self, text: str, model: str | None = None) -> list[float]:
        return [0.1, 0.2, 0.3]


def _make_scored_chunk(source_id: str, category: SourceCategory = SourceCategory.COMPLIANCE) -> ScoredChunk:
    chunk = EmbeddedChunk(
        chunk_id=str(uuid.uuid4()),
        source_id=source_id,
        category=category,
        ingested_at=datetime.now(timezone.utc),
        text="secure coding guidance text",
        vector=[0.1, 0.2, 0.3],
    )
    return ScoredChunk(chunk=chunk, score=0.9)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _active_scope(assets: list[str] | None = None) -> TargetScope:
    now = _now()
    return TargetScope(
        scope_id="scope-1",
        assets=assets or ["org/repo"],
        created_at=now,
        expires_at=now + timedelta(hours=2),
    )


def _expired_scope(assets: list[str] | None = None) -> TargetScope:
    now = _now()
    return TargetScope(
        scope_id="scope-expired",
        assets=assets or ["org/repo"],
        created_at=now - timedelta(hours=3),
        expires_at=now - timedelta(hours=1),
    )


def _revoked_scope(assets: list[str] | None = None) -> TargetScope:
    now = _now()
    return TargetScope(
        scope_id="scope-revoked",
        assets=assets or ["org/repo"],
        created_at=now,
        expires_at=now + timedelta(hours=2),
        revoked=True,
    )


def _make_finding(asset: str | None = "org/repo") -> Finding:
    return Finding(
        finding_id=str(uuid.uuid4()),
        originating_event_id=str(uuid.uuid4()),
        agent_id="test_agent",
        severity=Severity.HIGH,
        title="SQL injection in user input handler",
        detail="User input is passed directly to SQL query without sanitization.",
        asset=asset,
        created_at=_now(),
    )


def _make_agent(
    chunks: list[ScoredChunk] | None = None,
    patch_content: str = "--- a/file.py\n+++ b/file.py\n-bad\n+good",
) -> tuple[DevSecOpsAgent, _InMemoryAuditLog, FakePullRequestClient, _StubLLMGateway]:
    audit_log = _InMemoryAuditLog()
    retriever = _StubRetriever(chunks)
    llm = _StubLLMGateway(patch_content)
    pr_client = FakePullRequestClient()
    agent = DevSecOpsAgent(
        retriever=retriever,
        llm_gateway=llm,
        audit_log=audit_log,
        pr_client=pr_client,
    )
    return agent, audit_log, pr_client, llm


# ---------------------------------------------------------------------------
# FakePullRequestClient tests
# ---------------------------------------------------------------------------


class TestFakePullRequestClient:
    def test_create_pr_returns_pull_request(self) -> None:
        client = FakePullRequestClient()
        pr = client.create_pr(
            repo="org/repo",
            title="Fix SQL injection",
            body="Description here",
            branch="fix/vuln-abc123",
            patch="--- a/f\n+++ b/f\n-bad\n+good",
        )
        assert isinstance(pr, PullRequest)

    def test_create_pr_sets_requires_review_true(self) -> None:
        """PRs must always require human review before merge (Req 10.3)."""
        client = FakePullRequestClient()
        pr = client.create_pr(
            repo="org/repo",
            title="Fix",
            body="Body",
            branch="fix/branch",
            patch="patch",
        )
        assert pr.requires_review is True

    def test_create_pr_records_in_created_prs(self) -> None:
        client = FakePullRequestClient()
        pr1 = client.create_pr("org/repo", "Fix 1", "Body 1", "fix/1", "patch1")
        pr2 = client.create_pr("org/repo", "Fix 2", "Body 2", "fix/2", "patch2")
        assert len(client.created_prs) == 2
        assert client.created_prs[0] is pr1
        assert client.created_prs[1] is pr2

    def test_create_pr_assigns_unique_pr_ids(self) -> None:
        client = FakePullRequestClient()
        prs = [client.create_pr("org/repo", "Fix", "Body", f"fix/{i}", "patch") for i in range(5)]
        ids = [pr.pr_id for pr in prs]
        assert len(ids) == len(set(ids)), "all pr_ids must be unique"

    def test_create_pr_stores_correct_repo(self) -> None:
        client = FakePullRequestClient()
        pr = client.create_pr("myorg/myrepo", "Fix", "Body", "fix/1", "patch")
        assert pr.repo == "myorg/myrepo"

    def test_create_pr_stores_correct_patch(self) -> None:
        client = FakePullRequestClient()
        patch = "--- a/main.py\n+++ b/main.py\n-vulnerable_line\n+safe_line"
        pr = client.create_pr("org/repo", "Fix", "Body", "fix/1", patch)
        assert pr.patch == patch


# ---------------------------------------------------------------------------
# DevSecOpsAgent.remediate — in-scope (Req 10.1-10.4)
# ---------------------------------------------------------------------------


class TestRemediateInScope:
    def test_remediate_returns_pull_request_for_in_scope_finding(self) -> None:
        """In-scope finding returns a PullRequest (Req 10.2)."""
        agent, _, _, _ = _make_agent()
        finding = _make_finding(asset="org/repo")
        scope = _active_scope(assets=["org/repo"])
        result = agent.remediate(finding, scope)
        assert isinstance(result, PullRequest)

    def test_remediate_pr_requires_review(self) -> None:
        """Opened PR must require human review before merge (Req 10.3)."""
        agent, _, _, _ = _make_agent()
        finding = _make_finding(asset="org/repo")
        scope = _active_scope(assets=["org/repo"])
        pr = agent.remediate(finding, scope)
        assert isinstance(pr, PullRequest)
        assert pr.requires_review is True

    def test_remediate_pr_references_repo(self) -> None:
        """The opened PR targets the correct repository."""
        agent, _, _, _ = _make_agent()
        finding = _make_finding(asset="org/my-service")
        scope = _active_scope(assets=["org/my-service"])
        pr = agent.remediate(finding, scope)
        assert isinstance(pr, PullRequest)
        assert pr.repo == "org/my-service"

    def test_remediate_pr_body_references_guidance_source_ids(self) -> None:
        """PR body must reference the guidance source IDs retrieved from RAG (Req 10.4)."""
        chunks = [
            _make_scored_chunk("compliance-src-1"),
            _make_scored_chunk("compliance-src-2"),
        ]
        agent, _, _, _ = _make_agent(chunks=chunks)
        finding = _make_finding(asset="org/repo")
        scope = _active_scope(assets=["org/repo"])
        pr = agent.remediate(finding, scope)
        assert isinstance(pr, PullRequest)
        assert "compliance-src-1" in pr.body
        assert "compliance-src-2" in pr.body

    def test_remediate_pr_body_contains_vulnerability_title(self) -> None:
        """PR body must describe the addressed vulnerability (Req 10.2)."""
        agent, _, _, _ = _make_agent()
        finding = _make_finding(asset="org/repo")
        scope = _active_scope(assets=["org/repo"])
        pr = agent.remediate(finding, scope)
        assert isinstance(pr, PullRequest)
        assert finding.title in pr.body

    def test_remediate_pr_body_contains_severity(self) -> None:
        """PR body should include the severity of the vulnerability."""
        agent, _, _, _ = _make_agent()
        finding = _make_finding(asset="org/repo")
        scope = _active_scope(assets=["org/repo"])
        pr = agent.remediate(finding, scope)
        assert isinstance(pr, PullRequest)
        assert finding.severity.value in pr.body

    def test_remediate_pr_patch_comes_from_llm(self) -> None:
        """The PR patch is generated by the LLM Gateway (Req 10.1)."""
        expected_patch = "--- a/auth.py\n+++ b/auth.py\n-unsafe\n+safe"
        agent, _, _, llm = _make_agent(patch_content=expected_patch)
        finding = _make_finding(asset="org/repo")
        scope = _active_scope(assets=["org/repo"])
        pr = agent.remediate(finding, scope)
        assert isinstance(pr, PullRequest)
        assert pr.patch == expected_patch

    def test_remediate_invokes_llm_with_finding_context(self) -> None:
        """LLM should be called with the finding detail as part of the prompt."""
        agent, _, _, llm = _make_agent()
        finding = _make_finding(asset="org/repo")
        scope = _active_scope(assets=["org/repo"])
        agent.remediate(finding, scope)
        assert len(llm.calls) == 1
        assert finding.detail in llm.calls[0].prompt

    def test_remediate_retrieves_compliance_guidance(self) -> None:
        """Agent must call retriever for COMPLIANCE guidance (Req 10.4)."""
        chunks = [_make_scored_chunk("owasp-top10-sec-coding")]
        agent, _, _, _ = _make_agent(chunks=chunks)
        finding = _make_finding(asset="org/repo")
        scope = _active_scope(assets=["org/repo"])
        pr = agent.remediate(finding, scope)
        assert isinstance(pr, PullRequest)
        assert "owasp-top10-sec-coding" in pr.body

    def test_remediate_emits_pr_opened_audit_record(self) -> None:
        """A PR_OPENED audit record must be appended (Req 12.1)."""
        agent, audit_log, _, _ = _make_agent()
        finding = _make_finding(asset="org/repo")
        scope = _active_scope(assets=["org/repo"])
        agent.remediate(finding, scope)
        pr_opened = [r for r in audit_log.read_all() if r.action == AuditAction.PR_OPENED]
        assert len(pr_opened) == 1

    def test_remediate_pr_opened_audit_contains_pr_id(self) -> None:
        """PR_OPENED audit record detail must contain pr_id (Req 12.1)."""
        agent, audit_log, pr_client, _ = _make_agent()
        finding = _make_finding(asset="org/repo")
        scope = _active_scope(assets=["org/repo"])
        pr = agent.remediate(finding, scope)
        assert isinstance(pr, PullRequest)
        pr_opened_records = [r for r in audit_log.read_all() if r.action == AuditAction.PR_OPENED]
        assert pr_opened_records[0].detail["pr_id"] == pr.pr_id

    def test_remediate_pr_opened_audit_contains_context_refs(self) -> None:
        """PR_OPENED audit record detail must contain context_refs (Req 12.1)."""
        chunks = [_make_scored_chunk("guidance-ref-1")]
        agent, audit_log, _, _ = _make_agent(chunks=chunks)
        finding = _make_finding(asset="org/repo")
        scope = _active_scope(assets=["org/repo"])
        agent.remediate(finding, scope)
        pr_opened_records = [r for r in audit_log.read_all() if r.action == AuditAction.PR_OPENED]
        assert "guidance-ref-1" in pr_opened_records[0].detail["context_refs"]

    def test_remediate_pr_opened_audit_does_not_leak_patch(self) -> None:
        """Audit record must NOT contain the raw patch (security requirement)."""
        patch = "super_secret_patch_content_xyz"
        agent, audit_log, _, _ = _make_agent(patch_content=patch)
        finding = _make_finding(asset="org/repo")
        scope = _active_scope(assets=["org/repo"])
        agent.remediate(finding, scope)
        pr_opened_records = [r for r in audit_log.read_all() if r.action == AuditAction.PR_OPENED]
        detail_str = str(pr_opened_records[0].detail)
        assert patch not in detail_str

    def test_remediate_pr_stored_in_fake_client(self) -> None:
        """FakePullRequestClient records the opened PR."""
        agent, _, pr_client, _ = _make_agent()
        finding = _make_finding(asset="org/repo")
        scope = _active_scope(assets=["org/repo"])
        pr = agent.remediate(finding, scope)
        assert isinstance(pr, PullRequest)
        assert len(pr_client.created_prs) == 1
        assert pr_client.created_prs[0].pr_id == pr.pr_id


# ---------------------------------------------------------------------------
# DevSecOpsAgent.remediate — out-of-scope (Req 10.5)
# ---------------------------------------------------------------------------


class TestRemediateOutOfScope:
    def test_remediate_declines_when_asset_not_in_scope(self) -> None:
        """Asset not in scope → Declination returned (Req 10.5)."""
        agent, _, _, _ = _make_agent()
        finding = _make_finding(asset="org/other-repo")
        scope = _active_scope(assets=["org/repo"])
        result = agent.remediate(finding, scope)
        assert isinstance(result, Declination)

    def test_remediate_declines_when_scope_expired(self) -> None:
        """Expired scope → Declination returned (Req 10.5)."""
        agent, _, _, _ = _make_agent()
        finding = _make_finding(asset="org/repo")
        scope = _expired_scope(assets=["org/repo"])
        result = agent.remediate(finding, scope)
        assert isinstance(result, Declination)

    def test_remediate_declines_when_scope_revoked(self) -> None:
        """Revoked scope → Declination returned (Req 10.5)."""
        agent, _, _, _ = _make_agent()
        finding = _make_finding(asset="org/repo")
        scope = _revoked_scope(assets=["org/repo"])
        result = agent.remediate(finding, scope)
        assert isinstance(result, Declination)

    def test_remediate_declines_when_finding_has_no_asset(self) -> None:
        """Finding with no asset → Declination returned (Req 10.5)."""
        agent, _, _, _ = _make_agent()
        finding = _make_finding(asset=None)
        scope = _active_scope(assets=["org/repo"])
        result = agent.remediate(finding, scope)
        assert isinstance(result, Declination)

    def test_remediate_declination_carries_finding_id(self) -> None:
        agent, _, _, _ = _make_agent()
        finding = _make_finding(asset="out-of-scope-repo")
        scope = _active_scope(assets=["org/repo"])
        result = agent.remediate(finding, scope)
        assert isinstance(result, Declination)
        assert result.finding_id == finding.finding_id

    def test_remediate_declination_has_non_empty_reason(self) -> None:
        agent, _, _, _ = _make_agent()
        finding = _make_finding(asset="org/other-repo")
        scope = _active_scope(assets=["org/repo"])
        result = agent.remediate(finding, scope)
        assert isinstance(result, Declination)
        assert result.reason

    def test_remediate_emits_remediation_declined_audit_record(self) -> None:
        """REMEDIATION_DECLINED audit record must be appended (Req 10.5, 12.1)."""
        agent, audit_log, _, _ = _make_agent()
        finding = _make_finding(asset="org/other-repo")
        scope = _active_scope(assets=["org/repo"])
        agent.remediate(finding, scope)
        declined = [r for r in audit_log.read_all() if r.action == AuditAction.REMEDIATION_DECLINED]
        assert len(declined) == 1

    def test_remediate_declined_audit_record_targets_asset(self) -> None:
        """REMEDIATION_DECLINED audit record must target the finding's asset."""
        agent, audit_log, _, _ = _make_agent()
        finding = _make_finding(asset="org/other-repo")
        scope = _active_scope(assets=["org/repo"])
        agent.remediate(finding, scope)
        declined = [r for r in audit_log.read_all() if r.action == AuditAction.REMEDIATION_DECLINED]
        assert declined[0].target == "org/other-repo"

    def test_remediate_declined_does_not_open_pr(self) -> None:
        """No PR should be opened when declining (Req 10.5)."""
        agent, _, pr_client, _ = _make_agent()
        finding = _make_finding(asset="org/other-repo")
        scope = _active_scope(assets=["org/repo"])
        agent.remediate(finding, scope)
        assert pr_client.created_prs == []

    def test_remediate_declined_does_not_call_llm(self) -> None:
        """LLM should not be called when declining."""
        agent, _, _, llm = _make_agent()
        finding = _make_finding(asset="org/other-repo")
        scope = _active_scope(assets=["org/repo"])
        agent.remediate(finding, scope)
        assert llm.calls == []

    def test_remediate_declined_no_pr_opened_audit_record(self) -> None:
        """PR_OPENED audit record must NOT be emitted on declination."""
        agent, audit_log, _, _ = _make_agent()
        finding = _make_finding(asset="org/other-repo")
        scope = _active_scope(assets=["org/repo"])
        agent.remediate(finding, scope)
        pr_opened = [r for r in audit_log.read_all() if r.action == AuditAction.PR_OPENED]
        assert pr_opened == []


# ---------------------------------------------------------------------------
# Property 23: In-scope findings yield review-required PRs with fix,
#              description, and guidance
# Feature: autonomous-cyber-defense-platform, Property 23: In-scope findings yield review-required PRs with fix, description, and guidance
# ---------------------------------------------------------------------------

import uuid as _uuid
from datetime import datetime as _datetime, timezone as _timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from acdp.agents.devsecops_agent import (
    DevSecOpsAgent,
    FakePullRequestClient,
    PullRequest,
    Declination,
)
from acdp.models import (
    AuditRecord,
    EmbeddedChunk,
    Finding,
    LLMRequest,
    LLMResponse,
    ScoredChunk,
    Severity,
    SourceCategory,
    TargetScope,
)

# ---------------------------------------------------------------------------
# Stubs (local copies to avoid import-order issues in the module)
# ---------------------------------------------------------------------------


class _Prop23AuditLog:
    """In-memory audit log for Property 23."""

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


class _Prop23Retriever:
    """Retriever stub that always returns at least one compliance chunk.

    Returns a deterministic ``ScoredChunk`` tagged with a known source-id so
    the property can assert the PR body references it.
    """

    GUIDANCE_SOURCE_ID = "property23-compliance-guidance"

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        category: str | None = None,
    ) -> list[ScoredChunk]:
        chunk = EmbeddedChunk(
            chunk_id="prop23-chunk-1",
            source_id=self.GUIDANCE_SOURCE_ID,
            category=SourceCategory.COMPLIANCE,
            ingested_at=_datetime(2024, 1, 1, tzinfo=_timezone.utc),
            text="Always validate and sanitise input before processing.",
            vector=[0.1, 0.2, 0.3],
        )
        return [ScoredChunk(chunk=chunk, score=0.95)]


class _Prop23LLMGateway:
    """LLM Gateway stub that returns a non-empty, deterministic patch."""

    # A valid unified-diff patch that is non-empty.
    PATCH = "--- a/app.py\n+++ b/app.py\n@@ -1,3 +1,3 @@\n-unsafe_call(user_input)\n+safe_call(sanitize(user_input))\n"

    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(model="stub-llm", content=self.PATCH)

    def embed(self, text: str, model: str | None = None) -> list[float]:
        return [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Hypothesis strategies for Property 23
# ---------------------------------------------------------------------------

# Non-empty printable identifier alphabet (no whitespace).
_p23_identifier = st.text(
    alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x7E),
    min_size=1,
    max_size=24,
)

# Repository name: simple "org/repo" format, both parts non-empty and
# printable without slashes or special characters so the name is a valid
# asset identifier.
_p23_repo_part = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=1,
    max_size=20,
)


@st.composite
def _p23_repo_names(draw: st.DrawFn) -> str:
    """Generate repo names of the form ``org/repo`` — valid asset identifiers."""
    org = draw(_p23_repo_part)
    repo = draw(_p23_repo_part)
    return f"{org}/{repo}"


# A fixed far-future expiry so the scope is always active during the test.
_P23_CREATED_AT = _datetime(2024, 1, 1, tzinfo=_timezone.utc)
_P23_EXPIRES_AT = _datetime(2099, 12, 31, 23, 59, 59, tzinfo=_timezone.utc)

# Printable text for finding fields.
_p23_text = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    min_size=1,
    max_size=120,
)


@st.composite
def _p23_in_scope_finding_and_scope(draw: st.DrawFn) -> dict:
    """Generate a (Finding, TargetScope) pair where the finding is always in scope.

    * The scope is active (not expired, not revoked).
    * The finding's asset is one of the scope's assets.
    * The finding has a non-empty title and detail so the PR body is meaningful.
    """
    # Draw at least one repo name that will be both in the scope assets list
    # and used as the finding's asset.
    repo = draw(_p23_repo_names())
    # Optionally include additional repos in the scope.
    extra_repos = draw(st.lists(_p23_repo_names(), min_size=0, max_size=4))
    all_repos = [repo] + extra_repos

    scope = TargetScope(
        scope_id=draw(_p23_identifier),
        assets=all_repos,
        created_at=_P23_CREATED_AT,
        expires_at=_P23_EXPIRES_AT,
        revoked=False,
    )

    finding = Finding(
        finding_id=draw(_p23_identifier),
        originating_event_id=draw(_p23_identifier),
        agent_id=draw(_p23_identifier),
        severity=draw(st.sampled_from(list(Severity))),
        title=draw(_p23_text),
        detail=draw(_p23_text),
        asset=repo,
        context_refs=draw(st.lists(_p23_identifier, min_size=0, max_size=3)),
        created_at=_datetime(2024, 6, 1, tzinfo=_timezone.utc),
    )

    return {"finding": finding, "scope": scope}


# ---------------------------------------------------------------------------
# Property 23 test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(scenario=_p23_in_scope_finding_and_scope())
def test_property23_in_scope_findings_yield_review_required_prs(
    scenario: dict,
) -> None:
    """**Validates: Requirements 10.1, 10.2, 10.3, 10.4**

    For any vulnerability Finding whose asset is a repository within a valid,
    active TargetScope, DevSecOpsAgent.remediate() SHALL:

    - Return a PullRequest (not a Declination) [Req 10.2]
    - The PR SHALL contain a non-empty patch/fix [Req 10.1]
    - The PR body SHALL be non-empty and reference the vulnerability [Req 10.2]
    - The PR body SHALL reference at least one secure-coding guidance source ID
      retrieved from the RAG Core [Req 10.4]
    - The PR SHALL require human review before merge (requires_review=True) [Req 10.3]
    """
    finding: Finding = scenario["finding"]
    scope: TargetScope = scenario["scope"]

    audit_log = _Prop23AuditLog()
    retriever = _Prop23Retriever()
    llm = _Prop23LLMGateway()
    pr_client = FakePullRequestClient()

    agent = DevSecOpsAgent(
        retriever=retriever,
        llm_gateway=llm,
        audit_log=audit_log,
        pr_client=pr_client,
    )

    result = agent.remediate(finding, scope)

    # Req 10.2: result MUST be a PullRequest, not a Declination.
    assert isinstance(result, PullRequest), (
        f"Expected PullRequest for in-scope finding (asset={finding.asset!r}, "
        f"scope assets={scope.assets!r}), got {type(result).__name__}: {result!r}"
    )

    pr: PullRequest = result

    # Req 10.1: the PR must contain a non-empty proposed fix/patch.
    assert pr.patch, (
        f"PR patch must be non-empty (Req 10.1), got empty string. "
        f"finding_id={finding.finding_id!r}"
    )

    # Req 10.2: the PR body must be non-empty and reference the vulnerability.
    assert pr.body, (
        f"PR body must be non-empty (Req 10.2). finding_id={finding.finding_id!r}"
    )
    # The title and severity should appear in the PR body.
    assert finding.title in pr.body, (
        f"PR body must describe the addressed vulnerability (Req 10.2). "
        f"Expected to find title={finding.title!r} in body. "
        f"finding_id={finding.finding_id!r}"
    )

    # Req 10.4: the PR body must reference the secure-coding guidance source ID
    # retrieved from the RAG Core.
    assert _Prop23Retriever.GUIDANCE_SOURCE_ID in pr.body, (
        f"PR body must reference RAG guidance source IDs (Req 10.4). "
        f"Expected {_Prop23Retriever.GUIDANCE_SOURCE_ID!r} in body. "
        f"finding_id={finding.finding_id!r}"
    )

    # Req 10.3: the PR must be in a state that requires human review before merge.
    assert pr.requires_review is True, (
        f"PR must require human review before merge (Req 10.3), "
        f"but requires_review={pr.requires_review!r}. "
        f"finding_id={finding.finding_id!r}"
    )
