"""Property-based tests for the append-only audit log (Req 12)."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from acdp.audit import JsonlAuditLog, guarded_action
from acdp.exceptions import AuditWriteError
from acdp.models import AuditAction, AuditRecord
from tests.strategies import audit_records


def _is_well_formed(record: AuditRecord) -> bool:
    """A record is well-formed iff it carries the four Req 12.1 fields."""
    return (
        record.timestamp is not None
        and record.actor_id is not None
        and record.action is not None
        and record.outcome is not None
    )


# Feature: autonomous-cyber-defense-platform, Property 4: Audit append is
# single-record and append-only — For any prior audit log state and any single
# action, the resulting log SHALL equal the prior log followed by exactly one
# new well-formed record (containing timestamp, actor identifier, action type,
# and outcome), with all prior records preserved unchanged and in order.
# Validates: Requirements 12.1, 12.2
@settings(max_examples=200)
@given(
    prior=st.lists(audit_records(), max_size=8),
    new_record=audit_records(),
)
def test_audit_append_is_single_record_and_append_only(
    prior: list[AuditRecord], new_record: AuditRecord
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = JsonlAuditLog(Path(tmp) / "audit.jsonl")

        # Establish the prior log state.
        for record in prior:
            log.append(record)
        prior_state = log.read_all()
        assert len(prior_state) == len(prior)

        # Perform the single action under test.
        stored = log.append(new_record)
        resulting_state = log.read_all()

        # Exactly one record was added — no more, no fewer.
        assert len(resulting_state) == len(prior_state) + 1

        # All prior records are preserved unchanged and in their original order.
        assert resulting_state[: len(prior_state)] == prior_state

        # The resulting log equals prior log followed by the newly stored record.
        assert resulting_state[-1] == stored

        # The appended record is well-formed (timestamp, actor, action, outcome)
        # and preserves the caller-supplied field values.
        assert _is_well_formed(stored)
        assert stored.timestamp == new_record.timestamp
        assert stored.actor_id == new_record.actor_id
        assert stored.action == new_record.action
        assert stored.outcome == new_record.outcome

        # A monotonic seq is assigned: the new record's seq is exactly one more
        # than the highest prior seq (or 1 for the first record).
        prior_max_seq = max((r.seq for r in prior_state), default=0)
        assert stored.seq == prior_max_seq + 1

        # Sequence ids across the whole log are strictly increasing (monotonic).
        seqs = [r.seq for r in resulting_state]
        assert all(earlier < later for earlier, later in zip(seqs, seqs[1:]))


class _FailingAuditLog:
    """An AuditLog whose ``append`` always fails, simulating a durable-write error.

    Used to verify the *audit-before-act* contract (Req 12.4): when the audit
    append cannot be committed, the guarded action must not run.
    """

    def __init__(self) -> None:
        self.append_calls = 0

    def append(self, record: AuditRecord) -> AuditRecord:
        self.append_calls += 1
        raise AuditWriteError("simulated audit write failure")

    def read_all(self) -> list[AuditRecord]:
        return []


def _sample_record() -> AuditRecord:
    return AuditRecord(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        actor_id="orchestrator",
        action=AuditAction.FINDING_RECORDED,
        outcome="pending",
    )


# Req 12.4: a failed audit append must halt the action — the wrapped side
# effect must not commit and the AuditWriteError must surface to the caller.
def test_guarded_action_not_committed_when_audit_append_fails() -> None:
    """When ``append`` raises, the wrapped action never runs and the error surfaces."""
    audit_log = _FailingAuditLog()

    committed: list[str] = []

    def action() -> str:
        committed.append("applied")
        return "committed"

    with pytest.raises(AuditWriteError):
        guarded_action(audit_log, _sample_record(), action)

    # The append was attempted...
    assert audit_log.append_calls == 1
    # ...but because it failed, the action's side effect must NOT have occurred.
    assert committed == []


def test_guarded_action_failure_leaves_real_log_unchanged() -> None:
    """A failed append via a real log leaves no record and does not run the action.

    Simulates the durable-write failure by pointing the JSONL log at a path
    inside a non-existent directory, so ``open(..., "a")`` raises and is
    wrapped as :class:`AuditWriteError`.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # Target a file under a directory that does not exist so opening it for
        # append fails with OSError, which JsonlAuditLog surfaces as
        # AuditWriteError. read_all on this path cleanly reports no records.
        unwritable = Path(tmp) / "missing_dir" / "audit.jsonl"
        log = JsonlAuditLog(unwritable)

        side_effect_ran = False

        def action() -> None:
            nonlocal side_effect_ran
            side_effect_ran = True

        with pytest.raises(AuditWriteError):
            guarded_action(log, _sample_record(), action)

        # The action must not have committed its side effect.
        assert side_effect_ran is False
        # And no record was persisted.
        assert log.read_all() == []
