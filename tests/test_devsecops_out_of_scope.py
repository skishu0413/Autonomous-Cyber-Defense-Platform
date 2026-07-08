"""Unit tests for DevSecOps Agent out-of-scope declination (Task 17.3).

Covers Requirement 10.5:
  IF a Finding references a repository outside the Target_Scope, THEN THE
  DevSecOps_Agent SHALL decline to act and record the decision in the
  Audit_Log.

Test cases:
1. Finding whose asset is not listed in the active scope's assets → Declination (Req 10.5)
2. Finding whose scope has expired → Declination (Req 10.5)
3. Finding whose scope has been revoked → Declination (Req 10.5)
4. Finding with no asset (None) → Declination (Req 10.5)
5. Declination carries the correct finding_id (Req 10.5)
6. Declination has a non-empty human-readable reason (Req 10.5)
7. Exactly one REMEDIATION_DECLINED audit record is appended (Req 10.5, 12.1)
8. REMEDIATION_DECLINED audit record targets the finding's asset (Req 10.5, 12.1)
9. REMEDIATION_DECLINED actor_id is 'devsecops_agent' (Req 12.1)
10. No PullRequest is opened (no PR_OPENED audit record) on declination (Req 10.5)
11. The LLM is NOT called on declination (no side effects) (Req 10.5)
12. Each declined call writes exactly one REMEDIATION_DECLINED record (isolation)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from acdp.agents.devsecops_agent import (
    Declination,
    DevSecOpsAgent,
    FakePullRequestClient,
    PullRequest,
)
from acdp.models import (
    AuditAction,
    AuditRecord,
    Finding,
    LLMRequest,
    LLMResponse,
    ScoredChunk,
    Severity,
    TargetScope,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _InMemoryAuditLog:
    """Minimal in-memory audit log for isolated unit tests."""

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


class _NullRetriever:
    """Retriever stub that always returns an empty result set."""

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        category: str | None = None,
    ) -> list[ScoredChunk]:
        return []


class _TrackingLLMGateway:
    """LLM Gateway stub that records calls so tests can assert it was NOT invoked."""

    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(model="stub", content="patch")

    def embed(self, text: str, model: str | None = None) -> list[float]:
        return [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _future(hours: int = 2) -> datetime:
    return _now() + timedelta(hours=hours)


def _past(hours: int = 2) -> datetime:
    return _now() - timedelta(hours=hours)


def _active_scope(assets: list[str] | None = None, scope_id: str = "scope-active") -> TargetScope:
    """Return a scope that is active (not expired, not revoked)."""
    return TargetScope(
        scope_id=scope_id,
        assets=assets or ["org/in-scope-repo"],
        created_at=_now(),
        expires_at=_future(hours=2),
        revoked=False,
    )


def _expired_scope(assets: list[str] | None = None) -> TargetScope:
    """Return a scope whose expiry is in the past."""
    return TargetScope(
        scope_id="scope-expired",
        assets=assets or ["org/repo"],
        created_at=_past(hours=3),
        expires_at=_past(hours=1),
        revoked=False,
    )


def _revoked_scope(assets: list[str] | None = None) -> TargetScope:
    """Return a scope that is active timewise but has been revoked."""
    return TargetScope(
        scope_id="scope-revoked",
        assets=assets or ["org/repo"],
        created_at=_now(),
        expires_at=_future(hours=2),
        revoked=True,
    )


def _make_finding(asset: str | None = "org/out-of-scope-repo") -> Finding:
    return Finding(
        finding_id=str(uuid.uuid4()),
        originating_event_id=str(uuid.uuid4()),
        agent_id="test_agent",
        severity=Severity.HIGH,
        title="SQL injection in query builder",
        detail="User input flows directly into an unsanitized SQL query.",
        asset=asset,
        created_at=_now(),
    )


def _make_agent() -> tuple[DevSecOpsAgent, _InMemoryAuditLog, FakePullRequestClient, _TrackingLLMGateway]:
    """Construct a DevSecOpsAgent backed by in-memory test doubles."""
    audit_log = _InMemoryAuditLog()
    retriever = _NullRetriever()
    llm = _TrackingLLMGateway()
    pr_client = FakePullRequestClient()
    agent = DevSecOpsAgent(
        retriever=retriever,
        llm_gateway=llm,
        audit_log=audit_log,
        pr_client=pr_client,
    )
    return agent, audit_log, pr_client, llm


# ---------------------------------------------------------------------------
# Requirement 10.5 — out-of-scope declination (asset not in scope)
# ---------------------------------------------------------------------------


class TestOutOfScopeDeclination:
    """Finding referencing a repository not listed in the active scope's assets.

    Requirement 10.5: IF a Finding references a repository outside the
    Target_Scope, THEN THE DevSecOps_Agent SHALL decline to act and record
    the decision in the Audit_Log.
    """

    def test_returns_declination_not_pull_request(self) -> None:
        """remediate() returns a Declination when the asset is not in scope (Req 10.5)."""
        agent, _, _, _ = _make_agent()
        finding = _make_finding(asset="org/out-of-scope-repo")
        scope = _active_scope(assets=["org/in-scope-repo"])

        result = agent.remediate(finding, scope)

        assert isinstance(result, Declination), (
            f"Expected Declination for out-of-scope finding, got {type(result).__name__}"
        )

    def test_declination_carries_correct_finding_id(self) -> None:
        """The Declination must reference the declined finding's ID (Req 10.5)."""
        agent, _, _, _ = _make_agent()
        finding = _make_finding(asset="org/out-of-scope-repo")
        scope = _active_scope(assets=["org/in-scope-repo"])

        result = agent.remediate(finding, scope)

        assert isinstance(result, Declination)
        assert result.finding_id == finding.finding_id, (
            f"Declination.finding_id={result.finding_id!r} != "
            f"finding.finding_id={finding.finding_id!r}"
        )

    def test_declination_has_non_empty_reason(self) -> None:
        """The Declination must include a human-readable reason (Req 10.5)."""
        agent, _, _, _ = _make_agent()
        finding = _make_finding(asset="org/other-repo")
        scope = _active_scope(assets=["org/in-scope-repo"])

        result = agent.remediate(finding, scope)

        assert isinstance(result, Declination)
        assert result.reason, "Declination.reason must be a non-empty string"

    def test_appends_exactly_one_remediation_declined_audit_record(self) -> None:
        """Exactly one REMEDIATION_DECLINED record must be appended (Req 10.5, 12.1)."""
        agent, audit_log, _, _ = _make_agent()
        finding = _make_finding(asset="org/out-of-scope-repo")
        scope = _active_scope(assets=["org/in-scope-repo"])

        agent.remediate(finding, scope)

        declined = [
            r for r in audit_log.read_all()
            if r.action == AuditAction.REMEDIATION_DECLINED
        ]
        assert len(declined) == 1, (
            f"Expected exactly 1 REMEDIATION_DECLINED record, got {len(declined)}"
        )

    def test_audit_record_targets_finding_asset(self) -> None:
        """REMEDIATION_DECLINED record target must be the finding's asset (Req 10.5, 12.1)."""
        agent, audit_log, _, _ = _make_agent()
        finding = _make_finding(asset="org/out-of-scope-repo")
        scope = _active_scope(assets=["org/in-scope-repo"])

        agent.remediate(finding, scope)

        declined = [
            r for r in audit_log.read_all()
            if r.action == AuditAction.REMEDIATION_DECLINED
        ]
        assert declined[0].target == "org/out-of-scope-repo", (
            "REMEDIATION_DECLINED target must match the finding's asset"
        )

    def test_audit_record_actor_id_is_devsecops_agent(self) -> None:
        """REMEDIATION_DECLINED actor_id must be 'devsecops_agent' (Req 12.1)."""
        agent, audit_log, _, _ = _make_agent()
        finding = _make_finding(asset="org/out-of-scope-repo")
        scope = _active_scope(assets=["org/in-scope-repo"])

        agent.remediate(finding, scope)

        declined = [
            r for r in audit_log.read_all()
            if r.action == AuditAction.REMEDIATION_DECLINED
        ]
        assert declined[0].actor_id == "devsecops_agent"

    def test_no_pull_request_opened(self) -> None:
        """No PR is opened when declining — FakePullRequestClient stays empty (Req 10.5)."""
        agent, _, pr_client, _ = _make_agent()
        finding = _make_finding(asset="org/out-of-scope-repo")
        scope = _active_scope(assets=["org/in-scope-repo"])

        agent.remediate(finding, scope)

        assert pr_client.created_prs == [], (
            "No PR should be opened when the finding is out of scope"
        )

    def test_no_pr_opened_audit_record(self) -> None:
        """No PR_OPENED audit record must be written on declination (Req 10.5)."""
        agent, audit_log, _, _ = _make_agent()
        finding = _make_finding(asset="org/out-of-scope-repo")
        scope = _active_scope(assets=["org/in-scope-repo"])

        agent.remediate(finding, scope)

        pr_opened = [
            r for r in audit_log.read_all()
            if r.action == AuditAction.PR_OPENED
        ]
        assert pr_opened == [], "PR_OPENED must not be written when declining"

    def test_llm_not_called_on_declination(self) -> None:
        """LLM Gateway must NOT be called when the finding is out of scope (Req 10.5)."""
        agent, _, _, llm = _make_agent()
        finding = _make_finding(asset="org/out-of-scope-repo")
        scope = _active_scope(assets=["org/in-scope-repo"])

        agent.remediate(finding, scope)

        assert llm.calls == [], (
            "LLM must not be called on an out-of-scope declination"
        )


# ---------------------------------------------------------------------------
# Requirement 10.5 — expired scope treated as out-of-scope
# ---------------------------------------------------------------------------


class TestExpiredScopeDeclination:
    """Expired scope is treated as out-of-scope — agent must decline.

    Requirement 10.5 applies: the agent declines even when the finding's asset
    is listed in the scope's assets, because the scope has expired.
    """

    def test_expired_scope_returns_declination(self) -> None:
        """remediate() returns a Declination when the scope is expired (Req 10.5)."""
        agent, _, _, _ = _make_agent()
        finding = _make_finding(asset="org/repo")
        scope = _expired_scope(assets=["org/repo"])

        result = agent.remediate(finding, scope)

        assert isinstance(result, Declination), (
            "Expired scope must produce a Declination, not a PullRequest"
        )

    def test_expired_scope_appends_remediation_declined_audit_record(self) -> None:
        """REMEDIATION_DECLINED record is appended when scope has expired (Req 10.5)."""
        agent, audit_log, _, _ = _make_agent()
        finding = _make_finding(asset="org/repo")
        scope = _expired_scope(assets=["org/repo"])

        agent.remediate(finding, scope)

        declined = [
            r for r in audit_log.read_all()
            if r.action == AuditAction.REMEDIATION_DECLINED
        ]
        assert len(declined) == 1

    def test_expired_scope_no_pr_opened(self) -> None:
        """No PR is opened when the scope is expired (Req 10.5)."""
        agent, _, pr_client, _ = _make_agent()
        finding = _make_finding(asset="org/repo")
        scope = _expired_scope(assets=["org/repo"])

        agent.remediate(finding, scope)

        assert pr_client.created_prs == []

    def test_just_expired_scope_is_declined(self) -> None:
        """A scope that expired 1 second ago is treated as out-of-scope (Req 10.5)."""
        scope = TargetScope(
            scope_id="scope-just-expired",
            assets=["org/repo"],
            created_at=_past(hours=1),
            expires_at=_now() - timedelta(seconds=1),
        )
        agent, audit_log, _, _ = _make_agent()
        finding = _make_finding(asset="org/repo")

        result = agent.remediate(finding, scope)

        assert isinstance(result, Declination)
        declined = [
            r for r in audit_log.read_all()
            if r.action == AuditAction.REMEDIATION_DECLINED
        ]
        assert len(declined) == 1


# ---------------------------------------------------------------------------
# Requirement 10.5 — revoked scope treated as out-of-scope
# ---------------------------------------------------------------------------


class TestRevokedScopeDeclination:
    """Revoked scope is treated as out-of-scope — agent must decline.

    Requirement 10.5 applies even when the scope is not time-expired, but
    has been explicitly revoked by the operator.
    """

    def test_revoked_scope_returns_declination(self) -> None:
        """remediate() returns a Declination when the scope is revoked (Req 10.5)."""
        agent, _, _, _ = _make_agent()
        finding = _make_finding(asset="org/repo")
        scope = _revoked_scope(assets=["org/repo"])

        result = agent.remediate(finding, scope)

        assert isinstance(result, Declination), (
            "Revoked scope must produce a Declination, not a PullRequest"
        )

    def test_revoked_scope_appends_remediation_declined_audit_record(self) -> None:
        """REMEDIATION_DECLINED record is appended when scope is revoked (Req 10.5)."""
        agent, audit_log, _, _ = _make_agent()
        finding = _make_finding(asset="org/repo")
        scope = _revoked_scope(assets=["org/repo"])

        agent.remediate(finding, scope)

        declined = [
            r for r in audit_log.read_all()
            if r.action == AuditAction.REMEDIATION_DECLINED
        ]
        assert len(declined) == 1

    def test_revoked_scope_no_pr_opened(self) -> None:
        """No PR is opened when the scope is revoked (Req 10.5)."""
        agent, _, pr_client, _ = _make_agent()
        finding = _make_finding(asset="org/repo")
        scope = _revoked_scope(assets=["org/repo"])

        agent.remediate(finding, scope)

        assert pr_client.created_prs == []

    def test_revoked_scope_no_llm_call(self) -> None:
        """LLM Gateway must NOT be invoked when the scope is revoked (Req 10.5)."""
        agent, _, _, llm = _make_agent()
        finding = _make_finding(asset="org/repo")
        scope = _revoked_scope(assets=["org/repo"])

        agent.remediate(finding, scope)

        assert llm.calls == []


# ---------------------------------------------------------------------------
# Requirement 10.5 — finding with no asset (None)
# ---------------------------------------------------------------------------


class TestNoAssetDeclination:
    """Finding with asset=None has no repository to check — agent must decline.

    Requirement 10.5: no asset means the repository cannot be within any
    Target_Scope; the agent declines and records the decision.
    """

    def test_no_asset_returns_declination(self) -> None:
        """remediate() returns a Declination when finding.asset is None (Req 10.5)."""
        agent, _, _, _ = _make_agent()
        finding = _make_finding(asset=None)
        scope = _active_scope(assets=["org/in-scope-repo"])

        result = agent.remediate(finding, scope)

        assert isinstance(result, Declination), (
            "Finding with no asset must produce a Declination"
        )

    def test_no_asset_appends_remediation_declined_audit_record(self) -> None:
        """REMEDIATION_DECLINED record is appended when finding.asset is None (Req 10.5)."""
        agent, audit_log, _, _ = _make_agent()
        finding = _make_finding(asset=None)
        scope = _active_scope(assets=["org/in-scope-repo"])

        agent.remediate(finding, scope)

        declined = [
            r for r in audit_log.read_all()
            if r.action == AuditAction.REMEDIATION_DECLINED
        ]
        assert len(declined) == 1

    def test_no_asset_declination_carries_finding_id(self) -> None:
        """Declination.finding_id must match the finding's ID even with no asset (Req 10.5)."""
        agent, _, _, _ = _make_agent()
        finding = _make_finding(asset=None)
        scope = _active_scope(assets=["org/in-scope-repo"])

        result = agent.remediate(finding, scope)

        assert isinstance(result, Declination)
        assert result.finding_id == finding.finding_id


# ---------------------------------------------------------------------------
# Isolation: each declined call is independent
# ---------------------------------------------------------------------------


class TestDeclinationIsolation:
    """Each declined call produces exactly one audit record with no cross-call pollution."""

    def test_two_declined_calls_produce_two_audit_records(self) -> None:
        """Two separate out-of-scope remediations each write one REMEDIATION_DECLINED record."""
        agent, audit_log, _, _ = _make_agent()
        scope = _active_scope(assets=["org/in-scope-repo"])

        agent.remediate(_make_finding(asset="org/repo-a"), scope)
        agent.remediate(_make_finding(asset="org/repo-b"), scope)

        declined = [
            r for r in audit_log.read_all()
            if r.action == AuditAction.REMEDIATION_DECLINED
        ]
        assert len(declined) == 2, (
            "Each declined call must produce its own REMEDIATION_DECLINED record"
        )

    def test_declined_call_does_not_prevent_subsequent_in_scope_remediation(
        self,
    ) -> None:
        """A declination must not block a subsequent in-scope remediation from succeeding."""
        # We need the LLM to return something for the in-scope path; reuse _TrackingLLMGateway
        # but wire a retriever that returns nothing (the PR will still be opened).
        audit_log = _InMemoryAuditLog()
        retriever = _NullRetriever()
        llm = _TrackingLLMGateway()
        pr_client = FakePullRequestClient()
        agent = DevSecOpsAgent(
            retriever=retriever,
            llm_gateway=llm,
            audit_log=audit_log,
            pr_client=pr_client,
        )
        scope = _active_scope(assets=["org/in-scope-repo"])

        # First call: out-of-scope → Declination
        result_declined = agent.remediate(_make_finding(asset="org/other-repo"), scope)
        assert isinstance(result_declined, Declination)

        # Second call: in-scope → PullRequest
        result_pr = agent.remediate(_make_finding(asset="org/in-scope-repo"), scope)
        assert isinstance(result_pr, PullRequest), (
            "In-scope remediation following a declination must still produce a PullRequest"
        )
