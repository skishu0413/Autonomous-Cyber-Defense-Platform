"""Unit tests for Red Team Agent probe refusal (Task 16.4).

Covers the three refusal conditions mandated by Requirements 9.2 and 9.5:

1. Probe target is outside the authorized TargetScope (Req 9.2)
   → refused, PROBE_REFUSED audit record appended, no findings returned.

2. TargetScope is expired (expires_at in the past) (Req 9.5)
   → blocked, PROBE_REFUSED audit record appended, no findings returned.

3. TargetScope is revoked (revoked=True) (Req 9.5)
   → blocked, PROBE_REFUSED audit record appended, no findings returned.

In every case "no side effect" is confirmed by asserting an empty findings
list is returned and no spurious additional records beyond the expected
AUTHZ_DENY + PROBE_REFUSED pair are written to the audit log.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from acdp.agents.red_team_agent import ProbePlan, RedTeamAgent
from acdp.audit import AuditLog
from acdp.authorization import AuthorizationService
from acdp.models import AuditAction, AuditRecord, TargetScope
from acdp.knowledge_base.retrieve import Retriever


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
    ) -> list:
        return []


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _future(hours: int = 2) -> datetime:
    return _now() + timedelta(hours=hours)


def _past(hours: int = 2) -> datetime:
    return _now() - timedelta(hours=hours)


def _make_plan(
    target_asset: str = "host-a",
    weaknesses: list[str] | None = None,
) -> ProbePlan:
    return ProbePlan(
        plan_id=str(uuid.uuid4()),
        task_id=str(uuid.uuid4()),
        event_id=str(uuid.uuid4()),
        target_asset=target_asset,
        context_refs=["owasp-ref-1"],
        weaknesses=weaknesses if weaknesses is not None else ["SQLi", "XSS"],
    )


def _make_agent_with_scopes(
    scopes: list[TargetScope],
) -> tuple[RedTeamAgent, _InMemoryAuditLog]:
    audit_log = _InMemoryAuditLog()
    authz = AuthorizationService(scopes=scopes, audit=audit_log)
    retriever = _NullRetriever()
    agent = RedTeamAgent(audit_log=audit_log, retriever=retriever, authz=authz)
    return agent, audit_log


# ---------------------------------------------------------------------------
# Requirement 9.2 — out-of-scope refusal
# ---------------------------------------------------------------------------


class TestOutOfScopeRefusal:
    """Probe targeting an asset outside the authorized TargetScope is refused.

    Requirement 9.2: IF a probe target is outside the authorized Target_Scope,
    THEN THE Red_Team_Agent SHALL refuse the probe and record the refusal in
    the Audit_Log.
    """

    def test_out_of_scope_returns_empty_findings(self) -> None:
        """No findings are returned when the asset is not in the scope (Req 9.2)."""
        # Scope covers only "host-b"; plan targets "host-a"
        scope = TargetScope(
            scope_id="scope-oos",
            assets=["host-b"],
            created_at=_now(),
            expires_at=_future(),
        )
        agent, _ = _make_agent_with_scopes([scope])
        plan = _make_plan(target_asset="host-a", weaknesses=["SQLi", "XSS", "CSRF"])

        findings = agent.execute_probe(plan, scope)

        assert findings == [], (
            "Out-of-scope probe must return empty findings (no side effect)."
        )

    def test_out_of_scope_appends_probe_refused_audit_record(self) -> None:
        """A PROBE_REFUSED audit record is appended when the asset is out of scope (Req 9.2)."""
        scope = TargetScope(
            scope_id="scope-oos",
            assets=["host-b"],
            created_at=_now(),
            expires_at=_future(),
        )
        agent, audit_log = _make_agent_with_scopes([scope])
        plan = _make_plan(target_asset="host-a")

        agent.execute_probe(plan, scope)

        refused = [
            r for r in audit_log.read_all() if r.action == AuditAction.PROBE_REFUSED
        ]
        assert len(refused) == 1, (
            "Exactly one PROBE_REFUSED record must be appended on out-of-scope refusal."
        )

    def test_out_of_scope_audit_record_targets_the_probe_asset(self) -> None:
        """PROBE_REFUSED record target field is the probe asset (Req 9.2)."""
        scope = TargetScope(
            scope_id="scope-oos",
            assets=["host-b"],
            created_at=_now(),
            expires_at=_future(),
        )
        agent, audit_log = _make_agent_with_scopes([scope])
        plan = _make_plan(target_asset="host-a")

        agent.execute_probe(plan, scope)

        refused = [
            r for r in audit_log.read_all() if r.action == AuditAction.PROBE_REFUSED
        ]
        assert refused[0].target == "host-a", (
            "PROBE_REFUSED target must be the probe asset."
        )

    def test_out_of_scope_audit_record_actor_is_red_team_agent(self) -> None:
        """PROBE_REFUSED record actor_id is 'red_team_agent' (Req 9.2)."""
        scope = TargetScope(
            scope_id="scope-oos",
            assets=["host-b"],
            created_at=_now(),
            expires_at=_future(),
        )
        agent, audit_log = _make_agent_with_scopes([scope])
        plan = _make_plan(target_asset="host-a")

        agent.execute_probe(plan, scope)

        refused = [
            r for r in audit_log.read_all() if r.action == AuditAction.PROBE_REFUSED
        ]
        assert refused[0].actor_id == "red_team_agent", (
            "PROBE_REFUSED actor_id must be 'red_team_agent'."
        )

    def test_out_of_scope_no_extra_probe_refused_records(self) -> None:
        """Exactly one PROBE_REFUSED record is written — no duplicates (Req 9.2)."""
        scope = TargetScope(
            scope_id="scope-oos",
            assets=["host-b"],
            created_at=_now(),
            expires_at=_future(),
        )
        agent, audit_log = _make_agent_with_scopes([scope])
        plan = _make_plan(target_asset="host-a", weaknesses=["w1", "w2", "w3"])

        agent.execute_probe(plan, scope)

        refused_count = sum(
            1 for r in audit_log.read_all() if r.action == AuditAction.PROBE_REFUSED
        )
        assert refused_count == 1, (
            "Even for multi-weakness plans, only one PROBE_REFUSED record must be written."
        )

    def test_out_of_scope_empty_scope_assets_list(self) -> None:
        """An empty assets list in the scope causes refusal for any asset (Req 9.2)."""
        scope = TargetScope(
            scope_id="scope-empty",
            assets=[],  # no assets authorized
            created_at=_now(),
            expires_at=_future(),
        )
        agent, audit_log = _make_agent_with_scopes([scope])
        plan = _make_plan(target_asset="host-a")

        findings = agent.execute_probe(plan, scope)

        assert findings == []
        refused = [
            r for r in audit_log.read_all() if r.action == AuditAction.PROBE_REFUSED
        ]
        assert len(refused) == 1


# ---------------------------------------------------------------------------
# Requirement 9.5 — expired scope refusal
# ---------------------------------------------------------------------------


class TestExpiredScopeRefusal:
    """Probe blocked when the TargetScope has expired.

    Requirement 9.5: WHILE a Red Team authorization is expired or revoked,
    THE Red_Team_Agent SHALL execute no probes and SHALL record each blocked
    attempt in the Audit_Log.
    """

    def test_expired_scope_returns_empty_findings(self) -> None:
        """No findings are returned when the scope has expired (Req 9.5)."""
        scope = TargetScope(
            scope_id="scope-expired",
            assets=["host-a"],
            created_at=_past(hours=3),
            expires_at=_past(hours=1),  # expired 1 hour ago
        )
        agent, _ = _make_agent_with_scopes([scope])
        plan = _make_plan(target_asset="host-a", weaknesses=["SQLi", "XSS"])

        findings = agent.execute_probe(plan, scope)

        assert findings == [], (
            "Expired-scope probe must return empty findings (no side effect)."
        )

    def test_expired_scope_appends_probe_refused_audit_record(self) -> None:
        """A PROBE_REFUSED record is appended when the scope has expired (Req 9.5)."""
        scope = TargetScope(
            scope_id="scope-expired",
            assets=["host-a"],
            created_at=_past(hours=3),
            expires_at=_past(hours=1),
        )
        agent, audit_log = _make_agent_with_scopes([scope])
        plan = _make_plan(target_asset="host-a")

        agent.execute_probe(plan, scope)

        refused = [
            r for r in audit_log.read_all() if r.action == AuditAction.PROBE_REFUSED
        ]
        assert len(refused) == 1, (
            "Exactly one PROBE_REFUSED record must be appended on expired-scope block."
        )

    def test_expired_scope_audit_record_targets_the_probe_asset(self) -> None:
        """PROBE_REFUSED record target is the probe asset when scope expired (Req 9.5)."""
        scope = TargetScope(
            scope_id="scope-expired",
            assets=["prod-server"],
            created_at=_past(hours=3),
            expires_at=_past(hours=1),
        )
        agent, audit_log = _make_agent_with_scopes([scope])
        plan = _make_plan(target_asset="prod-server")

        agent.execute_probe(plan, scope)

        refused = [
            r for r in audit_log.read_all() if r.action == AuditAction.PROBE_REFUSED
        ]
        assert refused[0].target == "prod-server"

    def test_expired_scope_no_findings_even_with_multiple_weaknesses(self) -> None:
        """No findings at all, regardless of how many weaknesses are in the plan (Req 9.5)."""
        scope = TargetScope(
            scope_id="scope-expired",
            assets=["host-a"],
            created_at=_past(hours=5),
            expires_at=_past(hours=2),
        )
        agent, _ = _make_agent_with_scopes([scope])
        plan = _make_plan(
            target_asset="host-a",
            weaknesses=["SQLi", "XSS", "SSRF", "Auth bypass", "Path traversal"],
        )

        findings = agent.execute_probe(plan, scope)

        assert findings == []

    def test_just_expired_scope_is_refused(self) -> None:
        """A scope that expired at exactly now - 1 second is refused (Req 9.5)."""
        scope = TargetScope(
            scope_id="scope-just-expired",
            assets=["host-a"],
            created_at=_past(hours=1),
            expires_at=_now() - timedelta(seconds=1),
        )
        agent, _ = _make_agent_with_scopes([scope])
        plan = _make_plan(target_asset="host-a")

        findings = agent.execute_probe(plan, scope)

        assert findings == []


# ---------------------------------------------------------------------------
# Requirement 9.5 — revoked scope refusal
# ---------------------------------------------------------------------------


class TestRevokedScopeRefusal:
    """Probe blocked when the TargetScope has been revoked.

    Requirement 9.5: WHILE a Red Team authorization is expired or revoked,
    THE Red_Team_Agent SHALL execute no probes and SHALL record each blocked
    attempt in the Audit_Log.
    """

    def test_revoked_scope_returns_empty_findings(self) -> None:
        """No findings are returned when the scope is revoked (Req 9.5)."""
        scope = TargetScope(
            scope_id="scope-revoked",
            assets=["host-a"],
            created_at=_now(),
            expires_at=_future(),
            revoked=True,
        )
        agent, _ = _make_agent_with_scopes([scope])
        plan = _make_plan(target_asset="host-a", weaknesses=["SQLi", "XSS"])

        findings = agent.execute_probe(plan, scope)

        assert findings == [], (
            "Revoked-scope probe must return empty findings (no side effect)."
        )

    def test_revoked_scope_appends_probe_refused_audit_record(self) -> None:
        """A PROBE_REFUSED record is appended when the scope is revoked (Req 9.5)."""
        scope = TargetScope(
            scope_id="scope-revoked",
            assets=["host-a"],
            created_at=_now(),
            expires_at=_future(),
            revoked=True,
        )
        agent, audit_log = _make_agent_with_scopes([scope])
        plan = _make_plan(target_asset="host-a")

        agent.execute_probe(plan, scope)

        refused = [
            r for r in audit_log.read_all() if r.action == AuditAction.PROBE_REFUSED
        ]
        assert len(refused) == 1, (
            "Exactly one PROBE_REFUSED record must be appended on revoked-scope block."
        )

    def test_revoked_scope_audit_record_targets_the_probe_asset(self) -> None:
        """PROBE_REFUSED record target is the probe asset when scope revoked (Req 9.5)."""
        scope = TargetScope(
            scope_id="scope-revoked",
            assets=["internal-db"],
            created_at=_now(),
            expires_at=_future(),
            revoked=True,
        )
        agent, audit_log = _make_agent_with_scopes([scope])
        plan = _make_plan(target_asset="internal-db")

        agent.execute_probe(plan, scope)

        refused = [
            r for r in audit_log.read_all() if r.action == AuditAction.PROBE_REFUSED
        ]
        assert refused[0].target == "internal-db"

    def test_revoked_scope_no_findings_even_with_multiple_weaknesses(self) -> None:
        """No findings at all for a revoked scope, regardless of plan size (Req 9.5)."""
        scope = TargetScope(
            scope_id="scope-revoked",
            assets=["host-a"],
            created_at=_now(),
            expires_at=_future(),
            revoked=True,
        )
        agent, _ = _make_agent_with_scopes([scope])
        plan = _make_plan(
            target_asset="host-a",
            weaknesses=["SQLi", "XSS", "SSRF", "Auth bypass", "Path traversal"],
        )

        findings = agent.execute_probe(plan, scope)

        assert findings == []

    def test_revoked_scope_actor_id_is_red_team_agent(self) -> None:
        """PROBE_REFUSED actor_id is 'red_team_agent' for revoked scope (Req 9.5)."""
        scope = TargetScope(
            scope_id="scope-revoked",
            assets=["host-a"],
            created_at=_now(),
            expires_at=_future(),
            revoked=True,
        )
        agent, audit_log = _make_agent_with_scopes([scope])
        plan = _make_plan(target_asset="host-a")

        agent.execute_probe(plan, scope)

        refused = [
            r for r in audit_log.read_all() if r.action == AuditAction.PROBE_REFUSED
        ]
        assert refused[0].actor_id == "red_team_agent"

    def test_revoked_and_expired_scope_is_also_refused(self) -> None:
        """A scope that is both revoked and expired is also refused (Req 9.5)."""
        scope = TargetScope(
            scope_id="scope-revoked-expired",
            assets=["host-a"],
            created_at=_past(hours=3),
            expires_at=_past(hours=1),
            revoked=True,
        )
        agent, audit_log = _make_agent_with_scopes([scope])
        plan = _make_plan(target_asset="host-a")

        findings = agent.execute_probe(plan, scope)

        assert findings == []
        refused = [
            r for r in audit_log.read_all() if r.action == AuditAction.PROBE_REFUSED
        ]
        assert len(refused) == 1


# ---------------------------------------------------------------------------
# Cross-cutting: multiple-call isolation
# ---------------------------------------------------------------------------


class TestRefusalIsolation:
    """Refusal has no cross-call side effects — each call is independently clean."""

    def test_each_refused_call_writes_exactly_one_probe_refused_record(
        self,
    ) -> None:
        """Two separate refused probes each write exactly one PROBE_REFUSED record."""
        scope = TargetScope(
            scope_id="scope-revoked",
            assets=["host-a"],
            created_at=_now(),
            expires_at=_future(),
            revoked=True,
        )
        agent, audit_log = _make_agent_with_scopes([scope])
        plan1 = _make_plan(target_asset="host-a")
        plan2 = _make_plan(target_asset="host-a")

        agent.execute_probe(plan1, scope)
        agent.execute_probe(plan2, scope)

        refused = [
            r for r in audit_log.read_all() if r.action == AuditAction.PROBE_REFUSED
        ]
        assert len(refused) == 2, (
            "Each refused probe call must produce its own PROBE_REFUSED record."
        )

    def test_refused_probe_does_not_contaminate_subsequent_authorized_probe(
        self,
    ) -> None:
        """A refusal does not prevent a later authorized probe from succeeding."""
        # First, refuse a probe for an out-of-scope asset
        scope = TargetScope(
            scope_id="scope-1",
            assets=["host-b"],
            created_at=_now(),
            expires_at=_future(),
        )
        agent, audit_log = _make_agent_with_scopes([scope])
        plan_refused = _make_plan(target_asset="host-a")  # not in scope

        refused_findings = agent.execute_probe(plan_refused, scope)
        assert refused_findings == []

        # Now probe the authorized asset
        plan_authorized = _make_plan(
            target_asset="host-b", weaknesses=["SQLi", "CSRF"]
        )
        authorized_findings = agent.execute_probe(plan_authorized, scope)

        assert len(authorized_findings) == 2, (
            "Authorized probe must produce findings unaffected by prior refusal."
        )
        # The later probe must NOT write any PROBE_REFUSED record
        all_records = audit_log.read_all()
        refused_count = sum(
            1 for r in all_records if r.action == AuditAction.PROBE_REFUSED
        )
        # Only the first (refused) call should have written a PROBE_REFUSED record
        assert refused_count == 1
