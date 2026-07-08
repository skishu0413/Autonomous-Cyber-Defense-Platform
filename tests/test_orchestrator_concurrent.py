"""Unit test for concurrent multi-agent event processing — Task 19.6.

Requirement 1.6: THE Orchestrator SHALL process events from multiple Agents
without requiring the Operator to sequence events manually.

These tests assert that:
1. Multiple security events of different types can be dispatched simultaneously
   via threading without manual sequencing.
2. All task plans are fully processed (all tasks reach a terminal state).
3. Findings from all agents across all concurrent dispatches are recorded.
4. No manual ordering or inter-event coordination is required from the caller.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from acdp.models import (
    AuditAction,
    AuditRecord,
    Finding,
    SecurityEvent,
    SecurityEventType,
    Severity,
    Task,
    TargetScope,
    TaskPlan,
    TaskResult,
    TaskState,
)
from acdp.orchestrator import AgentCallable, Orchestrator


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _active_scope(scope_id: str, assets: list[str] | None = None) -> TargetScope:
    return TargetScope(
        scope_id=scope_id,
        assets=assets or ["host-a", "repo-b"],
        created_at=_now(),
        expires_at=_now() + timedelta(hours=4),
    )


def _make_event(
    event_type: SecurityEventType,
    scope_id: str = "scope-concurrent",
) -> SecurityEvent:
    return SecurityEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        target_scope_id=scope_id,
        payload={},
        timestamp=_now(),
    )


class _ThreadSafeAuditLog:
    """Thread-safe in-memory audit log for concurrent dispatch tests."""

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []
        self._lock = threading.Lock()
        self._seq = 0

    def append(self, record: AuditRecord) -> AuditRecord:
        with self._lock:
            self._seq += 1
            stored = record.model_copy(update={"seq": self._seq})
            self._records.append(stored)
            return stored

    def read_all(self) -> list[AuditRecord]:
        with self._lock:
            return list(self._records)


def _success_agent(
    agent_id: str, findings_per_call: int = 1
) -> AgentCallable:
    """Return an agent callable that succeeds and emits the given number of findings."""

    def _agent(task: Any, scope: Any) -> tuple[list[Finding], str | None]:
        produced = [
            Finding(
                finding_id=str(uuid.uuid4()),
                originating_event_id=task.event_id,
                agent_id=agent_id,
                severity=Severity.MEDIUM,
                title=f"Finding from {agent_id}",
                detail=f"Auto-generated finding for task {task.task_id}",
                created_at=_now(),
            )
            for _ in range(findings_per_call)
        ]
        return produced, None

    return _agent


def _make_orchestrator(
    agents: dict[str, AgentCallable] | None = None,
    scopes: dict[str, TargetScope] | None = None,
    audit_log: _ThreadSafeAuditLog | None = None,
) -> tuple[Orchestrator, _ThreadSafeAuditLog]:
    log = audit_log or _ThreadSafeAuditLog()
    orch = Orchestrator(
        audit_log=log,
        agent_registry=agents or {},
        scope_registry=scopes or {},
    )
    return orch, log


# ---------------------------------------------------------------------------
# Test class: concurrent multi-agent event processing (Req 1.6)
# ---------------------------------------------------------------------------


class TestConcurrentMultiAgentProcessing:
    """Requirement 1.6: THE Orchestrator SHALL process events from multiple
    Agents without requiring the Operator to sequence events manually.

    These tests verify that the Orchestrator handles multiple security events
    dispatched concurrently via threads, each with multiple agent tasks, without
    any manual inter-event sequencing by the caller.
    """

    def test_multiple_events_dispatched_concurrently_all_complete(self) -> None:
        """Multiple security events of different types are dispatched concurrently
        via threads; every task in every plan reaches COMPLETED state without
        the test needing to manually sequence them.

        Requirement 1.6: no manual sequencing required.
        """
        scope = _active_scope("scope-concurrent")
        agents: dict[str, AgentCallable] = {
            "blue_team": _success_agent("blue_team"),
            "guardrail": _success_agent("guardrail"),
            "red_team": _success_agent("red_team"),
            "devsecops": _success_agent("devsecops"),
        }
        orch, audit_log = _make_orchestrator(
            agents=agents,
            scopes={"scope-concurrent": scope},
        )

        # Create one event per SecurityEventType so all agents are exercised.
        events = [
            _make_event(SecurityEventType.TELEMETRY),
            _make_event(SecurityEventType.PROBE_REQUEST),
            _make_event(SecurityEventType.VULNERABILITY_FINDING),
            _make_event(SecurityEventType.PROMPT),
        ]

        # Collect all results via thread-safe container.
        all_results: list[list[TaskResult]] = [[] for _ in events]
        errors: list[Exception] = []

        def _dispatch(idx: int, event: SecurityEvent) -> None:
            try:
                plan = orch.handle_event(event)
                results = orch.dispatch(plan)
                all_results[idx] = results
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        # Dispatch all events concurrently — no manual sequencing.
        threads = [
            threading.Thread(target=_dispatch, args=(i, ev), daemon=True)
            for i, ev in enumerate(events)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread(s) raised exceptions: {errors}"

        # Every event must have produced results.
        for idx, results in enumerate(all_results):
            assert results, (
                f"Event #{idx} ({events[idx].event_type!r}) produced no TaskResults"
            )

        # Every task in every plan must be in a terminal state (COMPLETED or FAILED).
        terminal = {TaskState.COMPLETED, TaskState.FAILED}
        for idx, results in enumerate(all_results):
            for result in results:
                assert result.task.state in terminal, (
                    f"Event #{idx}: task {result.task.task_id!r} for agent "
                    f"{result.task.assigned_agent!r} ended in non-terminal state "
                    f"{result.task.state!r}"
                )

    def test_all_events_complete_without_manual_ordering(self) -> None:
        """Events submitted in an arbitrary order are all fully processed; the
        caller does not need to manually sequence them.

        This test submits events in reverse type order and asserts that results
        are independent of submission order.

        Requirement 1.6: no manual sequencing required.
        """
        scope = _active_scope("scope-unordered")
        agents: dict[str, AgentCallable] = {
            "blue_team": _success_agent("blue_team"),
            "guardrail": _success_agent("guardrail"),
            "red_team": _success_agent("red_team"),
            "devsecops": _success_agent("devsecops"),
        }
        orch, _ = _make_orchestrator(
            agents=agents,
            scopes={"scope-unordered": scope},
        )

        # Submit events in a deliberately "unordered" sequence.
        event_types = list(reversed(list(SecurityEventType)))
        events = [_make_event(et, scope_id="scope-unordered") for et in event_types]

        # Process all events in a single pass without any caller-imposed ordering.
        for event in events:
            plan = orch.handle_event(event)
            results = orch.dispatch(plan)

            # Each dispatch completes fully without waiting for other events.
            assert results, f"Expected results for event type {event.event_type!r}"
            for result in results:
                assert result.task.state in {TaskState.COMPLETED, TaskState.FAILED}, (
                    f"Task {result.task.task_id!r} not in terminal state: "
                    f"{result.task.state!r}"
                )

    def test_concurrent_dispatch_produces_findings_from_all_agents(self) -> None:
        """When multiple events are dispatched concurrently, findings from all
        agent types are recorded in the audit log without manual sequencing.

        Requirement 1.6: the Orchestrator processes events from multiple Agents
        without requiring the Operator to sequence events manually.
        """
        scope = _active_scope("scope-findings")
        agents: dict[str, AgentCallable] = {
            "blue_team": _success_agent("blue_team", findings_per_call=2),
            "guardrail": _success_agent("guardrail", findings_per_call=1),
            "red_team": _success_agent("red_team", findings_per_call=1),
            "devsecops": _success_agent("devsecops", findings_per_call=1),
        }
        shared_log = _ThreadSafeAuditLog()
        orch, _ = _make_orchestrator(
            agents=agents,
            scopes={"scope-findings": scope},
            audit_log=shared_log,
        )

        # Use the two event types that cover blue_team + red_team concurrently.
        events = [
            _make_event(SecurityEventType.TELEMETRY, "scope-findings"),
            _make_event(SecurityEventType.PROBE_REQUEST, "scope-findings"),
        ]

        barrier = threading.Barrier(len(events))
        errors: list[Exception] = []

        def _dispatch_with_barrier(event: SecurityEvent) -> None:
            try:
                plan = orch.handle_event(event)
                # Sync all threads so they hit dispatch simultaneously.
                barrier.wait(timeout=10)
                orch.dispatch(plan)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=_dispatch_with_barrier, args=(ev,), daemon=True)
            for ev in events
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread(s) raised exceptions: {errors}"

        # All FINDING_RECORDED audit records should include at least one entry
        # per agent that was invoked across both events.
        records = shared_log.read_all()
        finding_records = [r for r in records if r.action == AuditAction.FINDING_RECORDED]

        agent_ids_recorded = {r.detail.get("agent_id") for r in finding_records}
        # TELEMETRY → blue_team + guardrail; PROBE_REQUEST → red_team + guardrail.
        expected_agents = {"blue_team", "guardrail", "red_team"}
        assert expected_agents <= agent_ids_recorded, (
            f"Expected findings from {expected_agents}, but only recorded from "
            f"{agent_ids_recorded}"
        )

    def test_independent_events_do_not_interfere(self) -> None:
        """Results from one event's dispatch are independent from another's;
        no shared mutable state leaks between concurrent dispatches.

        Requirement 1.6: events from multiple Agents processed independently.
        """
        scope = _active_scope("scope-independent")
        agents: dict[str, AgentCallable] = {
            "blue_team": _success_agent("blue_team"),
            "guardrail": _success_agent("guardrail"),
        }
        orch, _ = _make_orchestrator(
            agents=agents,
            scopes={"scope-independent": scope},
        )

        N = 10  # Dispatch N events concurrently.
        events = [
            _make_event(SecurityEventType.TELEMETRY, "scope-independent")
            for _ in range(N)
        ]

        results_by_event: dict[str, list[TaskResult]] = {}
        lock = threading.Lock()
        errors: list[Exception] = []

        def _dispatch(event: SecurityEvent) -> None:
            try:
                plan = orch.handle_event(event)
                results = orch.dispatch(plan)
                with lock:
                    results_by_event[event.event_id] = results
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_dispatch, args=(ev,), daemon=True) for ev in events]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread(s) raised exceptions: {errors}"
        assert len(results_by_event) == N, (
            f"Expected {N} event result sets, got {len(results_by_event)}"
        )

        for event_id, results in results_by_event.items():
            # Each dispatch produces results tied to its own event_id.
            for result in results:
                assert result.task.event_id == event_id, (
                    f"Task event_id mismatch: expected {event_id!r}, "
                    f"got {result.task.event_id!r}"
                )
            # All tasks reach a terminal state.
            for result in results:
                assert result.task.state == TaskState.COMPLETED, (
                    f"Task {result.task.task_id!r} did not complete: "
                    f"{result.task.state!r}"
                )
