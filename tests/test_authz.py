"""Tests for the target scope registry (Req 11.3).

Covers Property 2 (every defined scope is recorded with a populated
``expires_at``) plus unit tests for audit-before-act recording and active-scope
filtering.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from acdp.audit import JsonlAuditLog
from acdp.exceptions import AuditWriteError
from acdp.models import ActionRequest, AuditAction, AuditRecord, TargetScope
from tests.strategies import action_requests, eval_times, target_scopes


def _registry(tmp: str) -> "ScopeRegistry":  # noqa: F821 - imported lazily below
    from acdp.authorization import ScopeRegistry

    return ScopeRegistry(JsonlAuditLog(Path(tmp) / "audit.jsonl"))


class _FailingAuditLog:
    """An AuditLog whose ``append`` always fails, to test audit-before-act."""

    def append(self, record: AuditRecord) -> AuditRecord:
        raise AuditWriteError("simulated audit write failure")

    def read_all(self) -> list[AuditRecord]:
        return []


# Feature: autonomous-cyber-defense-platform, Property 2: Every scope is
# recorded with an expiration — For any target scope defined by an operator,
# the stored scope SHALL have a populated ``expires_at`` value.
# Validates: Requirements 11.3
@settings(max_examples=200)
@given(scopes=st.lists(target_scopes(), max_size=8))
def test_every_defined_scope_is_recorded_with_expiration(
    scopes: list[TargetScope],
) -> None:
    from acdp.authorization import ScopeRegistry

    with tempfile.TemporaryDirectory() as tmp:
        audit_log = JsonlAuditLog(Path(tmp) / "audit.jsonl")
        registry = ScopeRegistry(audit_log)

        for scope in scopes:
            returned = registry.define_scope(scope)
            # define_scope returns the recorded scope unchanged...
            assert returned == scope
            # ...and the recorded scope always carries a populated expiration.
            assert returned.expires_at is not None
            assert isinstance(returned.expires_at, datetime)

        # Defining N scopes appends exactly N SCOPE_DEFINED audit records
        # (audit-before-act, Req 11.3), each targeting the defined scope.
        records = audit_log.read_all()
        assert len(records) == len(scopes)
        for record, scope in zip(records, scopes):
            assert record.action == AuditAction.SCOPE_DEFINED
            assert record.target == scope.scope_id


def test_define_scope_returns_scope_and_persists_it() -> None:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    scope = TargetScope(
        scope_id="s1",
        assets=["host-a", "host-b"],
        created_at=now,
        expires_at=now + timedelta(hours=1),
    )

    with tempfile.TemporaryDirectory() as tmp:
        registry = _registry(tmp)
        returned = registry.define_scope(scope)

        assert returned == scope
        # The scope is active while unexpired and not revoked.
        assert registry.get_active_scopes(now) == [scope]


def test_define_scope_audits_before_recording() -> None:
    """A failed audit append aborts the definition: nothing is recorded."""
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    scope = TargetScope(
        scope_id="s1",
        assets=["host-a"],
        created_at=now,
        expires_at=now + timedelta(hours=1),
    )
    from acdp.authorization import ScopeRegistry

    registry = ScopeRegistry(_FailingAuditLog())

    with pytest.raises(AuditWriteError):
        registry.define_scope(scope)

    # The scope must not have been recorded because the audit append failed.
    assert registry.get_active_scopes(now) == []


def test_get_active_scopes_excludes_expired_and_revoked() -> None:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    active = TargetScope(
        scope_id="active",
        assets=["a"],
        created_at=now,
        expires_at=now + timedelta(hours=1),
    )
    expired = TargetScope(
        scope_id="expired",
        assets=["b"],
        created_at=now - timedelta(hours=2),
        expires_at=now - timedelta(hours=1),
    )
    revoked = TargetScope(
        scope_id="revoked",
        assets=["c"],
        created_at=now,
        expires_at=now + timedelta(hours=1),
        revoked=True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        registry = _registry(tmp)
        registry.define_scope(active)
        registry.define_scope(expired)
        registry.define_scope(revoked)

        # Only the unexpired, unrevoked scope is active at ``now``.
        assert registry.get_active_scopes(now) == [active]


def test_get_active_scopes_empty_when_none_defined() -> None:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as tmp:
        registry = _registry(tmp)
        assert registry.get_active_scopes(now) == []


class _RecordingAuditLog:
    """An in-memory :class:`~acdp.audit.AuditLog` that records every append.

    Used as a spy so the authorization property can inspect exactly which
    decisions were written to the audit trail and assert that no side-effect
    action (probe/containment/PR) was ever recorded.
    """

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    def append(self, record: AuditRecord) -> AuditRecord:
        stored = record.model_copy(update={"seq": len(self.records) + 1})
        self.records.append(stored)
        return stored

    def read_all(self) -> list[AuditRecord]:
        return list(self.records)

    def last(self) -> AuditRecord:
        return self.records[-1]


# Audit actions that represent an actual external side effect (a probe,
# containment, or opened PR). A fail-closed DENY must never produce any of
# these — the only permitted record on a deny is the AUTHZ_DENY decision.
_SIDE_EFFECT_ACTIONS = frozenset(
    {
        AuditAction.PR_OPENED,
        AuditAction.PROBE_REFUSED,
        AuditAction.REMEDIATION_DECLINED,
    }
)


# Feature: autonomous-cyber-defense-platform, Property 1: Fail-closed, scope-gated authorization
# For any action request, any set of defined target scopes, and any evaluation
# time, the Authorization Service returns GRANT iff the target asset is a member
# of at least one scope that is neither expired nor revoked at that time; in
# every other case it returns DENY, no probe/containment/PR side effect occurs,
# and a scope-violation (AUTHZ_DENY) decision is recorded in the audit log.
# Validates: Requirements 8.3, 9.1, 9.2, 9.5, 10.5, 11.1, 11.2, 11.4
@settings(max_examples=200)
@given(
    req=action_requests(),
    scopes=st.lists(target_scopes(), max_size=8),
    at=eval_times(),
)
def test_authorization_fails_closed(
    req: ActionRequest,
    scopes: list[TargetScope],
    at: datetime,
) -> None:
    from acdp.authorization import AuthorizationService

    audit = _RecordingAuditLog()
    svc = AuthorizationService(scopes, audit=audit)

    decision = svc.authorize(req, at)

    # GRANT iff the requested asset is a member of at least one scope that is
    # neither expired nor revoked at the evaluation time; DENY in every other
    # case (out of scope, expired, revoked).
    in_valid_scope = any(
        scope.is_active(at) and req.asset in scope.assets for scope in scopes
    )
    assert decision.grant is in_valid_scope

    # Every decision is recorded in the audit log with the requesting agent and
    # target asset, exactly once (no other records are produced).
    assert len(audit.records) == 1
    recorded = audit.last()
    assert recorded.actor_id == req.agent_id
    assert recorded.target == req.asset

    if decision.grant:
        # A grant names the matching scope and is audited as AUTHZ_GRANT.
        assert decision.scope_id is not None
        assert any(
            s.scope_id == decision.scope_id
            and s.is_active(at)
            and req.asset in s.assets
            for s in scopes
        )
        assert recorded.action == AuditAction.AUTHZ_GRANT
    else:
        # Fail closed: no scope is named, a scope-violation (AUTHZ_DENY)
        # decision is recorded, and NO probe/containment/PR side effect occurs.
        assert decision.scope_id is None
        assert recorded.action == AuditAction.AUTHZ_DENY
        assert all(
            r.action not in _SIDE_EFFECT_ACTIONS for r in audit.records
        )


# Feature: autonomous-cyber-defense-platform, Property 3: Every authorization
# decision emits exactly one audit record with required fields — For any
# authorization request, exactly one audit record SHALL be appended containing
# the decision (grant/deny), the requesting agent identifier, and the target
# asset.
# Validates: Requirements 11.5
@settings(max_examples=200)
@given(
    req=action_requests(),
    scopes=st.lists(target_scopes(), max_size=8),
    at=eval_times(),
)
def test_authorization_emits_exactly_one_audit_record(
    req: ActionRequest,
    scopes: list[TargetScope],
    at: datetime,
) -> None:
    from acdp.authorization import AuthorizationService

    with tempfile.TemporaryDirectory() as tmp:
        audit_log = JsonlAuditLog(Path(tmp) / "audit.jsonl")
        svc = AuthorizationService(scopes, audit=audit_log)

        # Exactly-one is measured as the count delta of the append-only log
        # across a single authorize() call (Req 11.5).
        before = len(audit_log.read_all())
        decision = svc.authorize(req, at)
        after = audit_log.read_all()

        assert len(after) - before == 1

        # The single appended record carries the required fields: the decision
        # (grant/deny), the requesting agent identifier, and the target asset.
        recorded = after[-1]

        # Decision (grant/deny) — encoded both as the audit action and mirrored
        # in the record detail, and consistent with the returned decision.
        expected_action = (
            AuditAction.AUTHZ_GRANT if decision.grant else AuditAction.AUTHZ_DENY
        )
        assert recorded.action == expected_action
        assert recorded.outcome == ("grant" if decision.grant else "deny")
        assert recorded.detail["decision"] == ("grant" if decision.grant else "deny")

        # Requesting agent identifier.
        assert recorded.actor_id == req.agent_id
        assert recorded.detail["agent_id"] == req.agent_id

        # Target asset.
        assert recorded.target == req.asset
        assert recorded.detail["asset"] == req.asset
