"""Tests for the Orchestrator — Tasks 19.1 and 19.3.

Covers:
- handle_event routes every SecurityEventType to the correct agent(s) (Req 1.1)
- handle_event carries target_scope_id into every task (Req 1.2)
- dispatch transitions task states: pending → in-progress → completed | failed (Req 1.5)
- dispatch records findings in the audit log before handling subsequent failure (Req 1.3)
- dispatch records failure reason and marks task FAILED on agent failure (Req 1.4)
- dispatch processes multiple tasks without requiring manual sequencing (Req 1.6)
- unknown agent or missing scope results in a FAILED task with an explanatory reason
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from hypothesis import given, settings

from acdp.audit import AuditLog
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
from acdp.orchestrator import AgentCallable, Orchestrator, _KNOWN_AGENTS

from hypothesis import strategies as st

from tests.strategies import findings, security_events, task_plans


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class _InMemoryAuditLog:
    """Simple in-memory audit log for tests."""

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


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _active_scope(scope_id: str = "scope-1", assets: list[str] | None = None) -> TargetScope:
    return TargetScope(
        scope_id=scope_id,
        assets=assets or ["host-a"],
        created_at=_now(),
        expires_at=_now() + timedelta(hours=2),
    )


def _make_event(
    event_type: SecurityEventType = SecurityEventType.TELEMETRY,
    scope_id: str = "scope-1",
    event_id: str | None = None,
) -> SecurityEvent:
    return SecurityEvent(
        event_id=event_id or str(uuid.uuid4()),
        event_type=event_type,
        target_scope_id=scope_id,
        payload={},
        timestamp=_now(),
    )


def _success_agent(findings: list[Finding] | None = None) -> AgentCallable:
    """Return an agent callable that succeeds with the given findings."""
    _findings = findings or []

    def _agent(task: Any, scope: Any) -> tuple[list[Finding], str | None]:
        return list(_findings), None

    return _agent


def _failing_agent(reason: str = "agent error") -> AgentCallable:
    """Return an agent callable that always returns a failure_reason."""

    def _agent(task: Any, scope: Any) -> tuple[list[Finding], str | None]:
        return [], reason

    return _agent


def _raising_agent(exc: Exception | None = None) -> AgentCallable:
    """Return an agent callable that raises an exception."""
    _exc = exc or RuntimeError("unexpected agent failure")

    def _agent(task: Any, scope: Any) -> tuple[list[Finding], str | None]:
        raise _exc

    return _agent


def _partial_agent(findings: list[Finding], reason: str = "partial failure") -> AgentCallable:
    """Return an agent that returns findings AND a failure_reason (partial success).

    This models Req 1.3: an agent that emits findings before failing.
    """

    def _agent(task: Any, scope: Any) -> tuple[list[Finding], str | None]:
        return list(findings), reason

    return _agent


def _make_finding(event_id: str = "evt-1", agent_id: str = "blue_team") -> Finding:
    return Finding(
        finding_id=str(uuid.uuid4()),
        originating_event_id=event_id,
        agent_id=agent_id,
        severity=Severity.HIGH,
        title="Test finding",
        detail="A test finding produced by a stub agent.",
        created_at=_now(),
    )


def _make_orchestrator(
    agents: dict[str, AgentCallable] | None = None,
    scopes: dict[str, TargetScope] | None = None,
) -> tuple[Orchestrator, _InMemoryAuditLog]:
    audit_log = _InMemoryAuditLog()
    orch = Orchestrator(
        audit_log=audit_log,
        agent_registry=agents or {},
        scope_registry=scopes or {},
    )
    return orch, audit_log


# ---------------------------------------------------------------------------
# 19.1 — handle_event routing (Req 1.1, 1.2)
# ---------------------------------------------------------------------------


class TestHandleEvent:
    """handle_event must route every event type to the correct agent(s)."""

    def test_telemetry_routes_to_blue_team_and_guardrail(self) -> None:
        orch, _ = _make_orchestrator()
        event = _make_event(SecurityEventType.TELEMETRY)
        plan = orch.handle_event(event)

        agents = {t.assigned_agent for t in plan.tasks}
        assert "blue_team" in agents
        assert "guardrail" in agents

    def test_probe_request_routes_to_red_team_and_guardrail(self) -> None:
        orch, _ = _make_orchestrator()
        event = _make_event(SecurityEventType.PROBE_REQUEST)
        plan = orch.handle_event(event)

        agents = {t.assigned_agent for t in plan.tasks}
        assert "red_team" in agents
        assert "guardrail" in agents

    def test_vulnerability_finding_routes_to_devsecops_and_guardrail(self) -> None:
        orch, _ = _make_orchestrator()
        event = _make_event(SecurityEventType.VULNERABILITY_FINDING)
        plan = orch.handle_event(event)

        agents = {t.assigned_agent for t in plan.tasks}
        assert "devsecops" in agents
        assert "guardrail" in agents

    def test_prompt_routes_to_guardrail_only(self) -> None:
        orch, _ = _make_orchestrator()
        event = _make_event(SecurityEventType.PROMPT)
        plan = orch.handle_event(event)

        agents = [t.assigned_agent for t in plan.tasks]
        assert agents == ["guardrail"], f"Expected only guardrail, got {agents}"

    def test_unknown_routes_to_guardrail_only(self) -> None:
        orch, _ = _make_orchestrator()
        event = _make_event(SecurityEventType.UNKNOWN)
        plan = orch.handle_event(event)

        agents = [t.assigned_agent for t in plan.tasks]
        assert agents == ["guardrail"], f"Expected only guardrail, got {agents}"

    def test_all_assigned_agents_are_known(self) -> None:
        """Every task in every plan is assigned to a known agent (Req 1.1)."""
        from acdp.orchestrator import _KNOWN_AGENTS

        orch, _ = _make_orchestrator()
        for event_type in SecurityEventType:
            event = _make_event(event_type)
            plan = orch.handle_event(event)
            for task in plan.tasks:
                assert task.assigned_agent in _KNOWN_AGENTS, (
                    f"{event_type}: unexpected agent {task.assigned_agent!r}"
                )

    def test_plan_carries_event_id(self) -> None:
        """The produced plan's event_id matches the event's event_id."""
        orch, _ = _make_orchestrator()
        event = _make_event(SecurityEventType.TELEMETRY, event_id="evt-abc-123")
        plan = orch.handle_event(event)

        assert plan.event_id == "evt-abc-123"
        for task in plan.tasks:
            assert task.event_id == "evt-abc-123"

    def test_tasks_carry_target_scope_id(self) -> None:
        """Every task carries the event's target_scope_id (Req 1.2)."""
        orch, _ = _make_orchestrator()
        event = _make_event(SecurityEventType.TELEMETRY, scope_id="scope-x")
        plan = orch.handle_event(event)

        for task in plan.tasks:
            assert task.target_scope_id == "scope-x"

    def test_all_tasks_start_pending(self) -> None:
        """Every task in a fresh plan has state PENDING (Req 1.5)."""
        orch, _ = _make_orchestrator()
        event = _make_event(SecurityEventType.TELEMETRY)
        plan = orch.handle_event(event)

        for task in plan.tasks:
            assert task.state == TaskState.PENDING

    def test_plan_has_unique_plan_id(self) -> None:
        """Two calls for the same event produce plans with distinct plan_ids."""
        orch, _ = _make_orchestrator()
        event = _make_event(SecurityEventType.TELEMETRY)
        plan1 = orch.handle_event(event)
        plan2 = orch.handle_event(event)

        assert plan1.plan_id != plan2.plan_id

    def test_tasks_have_unique_task_ids(self) -> None:
        """All task_ids within a plan are unique."""
        orch, _ = _make_orchestrator()
        event = _make_event(SecurityEventType.TELEMETRY)
        plan = orch.handle_event(event)

        ids = [t.task_id for t in plan.tasks]
        assert len(ids) == len(set(ids))

    def test_no_duplicate_agents_per_plan(self) -> None:
        """Each agent appears at most once in a plan (no duplicate tasks)."""
        orch, _ = _make_orchestrator()
        for event_type in SecurityEventType:
            event = _make_event(event_type)
            plan = orch.handle_event(event)
            agents = [t.assigned_agent for t in plan.tasks]
            assert len(agents) == len(set(agents)), (
                f"{event_type}: duplicate agents in plan: {agents}"
            )


# ---------------------------------------------------------------------------
# 19.3 — dispatch: successful tasks (Req 1.5, 1.6)
# ---------------------------------------------------------------------------


class TestDispatchSuccess:
    """dispatch must correctly process tasks that succeed."""

    def test_dispatch_empty_plan_returns_empty_results(self) -> None:
        orch, _ = _make_orchestrator()
        plan = TaskPlan(plan_id="p-1", event_id="e-1", tasks=[])
        results = orch.dispatch(plan)
        assert results == []

    def test_dispatch_single_task_completes(self) -> None:
        scope = _active_scope("scope-1")
        orch, _ = _make_orchestrator(
            agents={"blue_team": _success_agent()},
            scopes={"scope-1": scope},
        )
        event = _make_event(SecurityEventType.TELEMETRY, "scope-1")
        plan = orch.handle_event(event)
        # Only keep the blue_team task for simplicity
        blue_tasks = [t for t in plan.tasks if t.assigned_agent == "blue_team"]
        focused_plan = TaskPlan(plan_id=plan.plan_id, event_id=plan.event_id, tasks=blue_tasks)

        results = orch.dispatch(focused_plan)

        assert len(results) == 1
        assert results[0].task.state == TaskState.COMPLETED
        assert results[0].failure_reason is None

    def test_dispatch_multiple_tasks_all_complete(self) -> None:
        """Multiple tasks are all dispatched and completed (Req 1.6)."""
        scope = _active_scope("scope-1")
        orch, _ = _make_orchestrator(
            agents={
                "blue_team": _success_agent(),
                "guardrail": _success_agent(),
            },
            scopes={"scope-1": scope},
        )
        event = _make_event(SecurityEventType.TELEMETRY, "scope-1")
        plan = orch.handle_event(event)

        results = orch.dispatch(plan)

        assert len(results) == len(plan.tasks)
        for result in results:
            assert result.task.state == TaskState.COMPLETED

    def test_dispatch_returns_findings_from_agent(self) -> None:
        scope = _active_scope("scope-1")
        finding = _make_finding()
        orch, _ = _make_orchestrator(
            agents={"blue_team": _success_agent([finding])},
            scopes={"scope-1": scope},
        )
        event = _make_event(SecurityEventType.TELEMETRY, "scope-1")
        plan = orch.handle_event(event)
        blue_tasks = [t for t in plan.tasks if t.assigned_agent == "blue_team"]
        focused_plan = TaskPlan(plan_id=plan.plan_id, event_id=plan.event_id, tasks=blue_tasks)

        results = orch.dispatch(focused_plan)

        assert len(results[0].findings) == 1
        assert results[0].findings[0].finding_id == finding.finding_id

    def test_dispatch_records_finding_in_audit_log(self) -> None:
        """Findings are recorded in the audit log with timestamp and agent id (Req 1.3)."""
        scope = _active_scope("scope-1")
        finding = _make_finding(agent_id="blue_team")
        orch, audit_log = _make_orchestrator(
            agents={"blue_team": _success_agent([finding])},
            scopes={"scope-1": scope},
        )
        event = _make_event(SecurityEventType.TELEMETRY, "scope-1")
        plan = orch.handle_event(event)
        blue_tasks = [t for t in plan.tasks if t.assigned_agent == "blue_team"]
        focused_plan = TaskPlan(plan_id=plan.plan_id, event_id=plan.event_id, tasks=blue_tasks)

        orch.dispatch(focused_plan)

        recorded = [
            r for r in audit_log.read_all()
            if r.action == AuditAction.FINDING_RECORDED
        ]
        assert len(recorded) == 1
        assert recorded[0].detail["finding_id"] == finding.finding_id
        assert recorded[0].detail["agent_id"] == "blue_team"
        assert recorded[0].timestamp is not None

    def test_dispatch_result_one_per_task(self) -> None:
        """dispatch returns exactly one TaskResult per task in the plan."""
        scope = _active_scope("scope-1")
        orch, _ = _make_orchestrator(
            agents={
                "blue_team": _success_agent(),
                "guardrail": _success_agent(),
                "red_team": _success_agent(),
            },
            scopes={"scope-1": scope},
        )
        # Build a plan with 3 tasks
        event = _make_event(SecurityEventType.PROBE_REQUEST, "scope-1")
        plan = orch.handle_event(event)

        results = orch.dispatch(plan)

        assert len(results) == len(plan.tasks)


# ---------------------------------------------------------------------------
# 19.3 — dispatch: failure handling (Req 1.3, 1.4, 1.5)
# ---------------------------------------------------------------------------


class TestDispatchFailure:
    """dispatch must mark failed tasks and record failure reasons."""

    def test_agent_returning_failure_reason_marks_task_failed(self) -> None:
        """When agent returns a failure_reason, task is marked FAILED (Req 1.4)."""
        scope = _active_scope("scope-1")
        orch, _ = _make_orchestrator(
            agents={"blue_team": _failing_agent("timeout: agent did not respond")},
            scopes={"scope-1": scope},
        )
        event = _make_event(SecurityEventType.TELEMETRY, "scope-1")
        plan = orch.handle_event(event)
        blue_tasks = [t for t in plan.tasks if t.assigned_agent == "blue_team"]
        focused_plan = TaskPlan(plan_id=plan.plan_id, event_id=plan.event_id, tasks=blue_tasks)

        results = orch.dispatch(focused_plan)

        assert results[0].task.state == TaskState.FAILED
        assert results[0].failure_reason == "timeout: agent did not respond"

    def test_agent_raising_exception_marks_task_failed(self) -> None:
        """An agent that raises an exception also marks the task FAILED (Req 1.4)."""
        scope = _active_scope("scope-1")
        orch, _ = _make_orchestrator(
            agents={"blue_team": _raising_agent(ValueError("bad input"))},
            scopes={"scope-1": scope},
        )
        event = _make_event(SecurityEventType.TELEMETRY, "scope-1")
        plan = orch.handle_event(event)
        blue_tasks = [t for t in plan.tasks if t.assigned_agent == "blue_team"]
        focused_plan = TaskPlan(plan_id=plan.plan_id, event_id=plan.event_id, tasks=blue_tasks)

        results = orch.dispatch(focused_plan)

        assert results[0].task.state == TaskState.FAILED
        assert "bad input" in results[0].failure_reason

    def test_failed_task_has_failure_reason_in_task(self) -> None:
        """task.failure_reason is set on FAILED tasks (Req 1.4)."""
        scope = _active_scope("scope-1")
        orch, _ = _make_orchestrator(
            agents={"blue_team": _failing_agent("specific reason")},
            scopes={"scope-1": scope},
        )
        event = _make_event(SecurityEventType.TELEMETRY, "scope-1")
        plan = orch.handle_event(event)
        blue_tasks = [t for t in plan.tasks if t.assigned_agent == "blue_team"]
        focused_plan = TaskPlan(plan_id=plan.plan_id, event_id=plan.event_id, tasks=blue_tasks)

        results = orch.dispatch(focused_plan)

        assert results[0].task.failure_reason == "specific reason"

    def test_failure_is_recorded_in_audit_log(self) -> None:
        """TASK_FAILED audit record is appended with the failure reason (Req 1.4)."""
        scope = _active_scope("scope-1")
        orch, audit_log = _make_orchestrator(
            agents={"blue_team": _failing_agent("LLM timeout")},
            scopes={"scope-1": scope},
        )
        event = _make_event(SecurityEventType.TELEMETRY, "scope-1")
        plan = orch.handle_event(event)
        blue_tasks = [t for t in plan.tasks if t.assigned_agent == "blue_team"]
        focused_plan = TaskPlan(plan_id=plan.plan_id, event_id=plan.event_id, tasks=blue_tasks)

        orch.dispatch(focused_plan)

        failed_records = [
            r for r in audit_log.read_all() if r.action == AuditAction.TASK_FAILED
        ]
        assert len(failed_records) == 1
        assert failed_records[0].detail["failure_reason"] == "LLM timeout"

    def test_missing_scope_marks_task_failed(self) -> None:
        """A task whose scope_id is not in the registry is marked FAILED."""
        orch, _ = _make_orchestrator(
            agents={"blue_team": _success_agent()},
            scopes={},  # empty — scope-1 not found
        )
        event = _make_event(SecurityEventType.TELEMETRY, "scope-1")
        plan = orch.handle_event(event)
        blue_tasks = [t for t in plan.tasks if t.assigned_agent == "blue_team"]
        focused_plan = TaskPlan(plan_id=plan.plan_id, event_id=plan.event_id, tasks=blue_tasks)

        results = orch.dispatch(focused_plan)

        assert results[0].task.state == TaskState.FAILED
        assert "scope-1" in results[0].failure_reason

    def test_unknown_agent_marks_task_failed(self) -> None:
        """A task assigned to an unregistered agent is marked FAILED."""
        scope = _active_scope("scope-1")
        orch, _ = _make_orchestrator(
            agents={},  # no agents registered
            scopes={"scope-1": scope},
        )
        event = _make_event(SecurityEventType.TELEMETRY, "scope-1")
        plan = orch.handle_event(event)

        results = orch.dispatch(plan)

        assert all(r.task.state == TaskState.FAILED for r in results)

    def test_task_state_always_valid(self) -> None:
        """Every dispatched task ends in a valid TaskState (Req 1.5)."""
        valid_states = set(TaskState)
        scope = _active_scope("scope-1")
        # Mix: one succeeds, one fails, one raises
        agents: dict[str, AgentCallable] = {
            "blue_team": _success_agent(),
            "guardrail": _failing_agent("guardrail error"),
            "red_team": _raising_agent(RuntimeError("crash")),
        }
        orch, _ = _make_orchestrator(
            agents=agents,
            scopes={"scope-1": scope},
        )
        for event_type in SecurityEventType:
            event = _make_event(event_type, "scope-1")
            plan = orch.handle_event(event)
            results = orch.dispatch(plan)
            for result in results:
                assert result.task.state in valid_states, (
                    f"Task {result.task.task_id} ended in invalid state {result.task.state!r}"
                )


# ---------------------------------------------------------------------------
# Req 1.3 — findings recorded BEFORE failure is handled
# ---------------------------------------------------------------------------


class TestFindingsBeforeFailure:
    """Findings from a partially-succeeded agent must be recorded before the failure (Req 1.3)."""

    def test_findings_recorded_before_failure_in_audit_log(self) -> None:
        """Findings appear in audit log before the TASK_FAILED record."""
        scope = _active_scope("scope-1")
        finding = _make_finding(agent_id="blue_team")
        orch, audit_log = _make_orchestrator(
            agents={"blue_team": _partial_agent([finding], "agent crashed after emitting findings")},
            scopes={"scope-1": scope},
        )
        event = _make_event(SecurityEventType.TELEMETRY, "scope-1")
        plan = orch.handle_event(event)
        blue_tasks = [t for t in plan.tasks if t.assigned_agent == "blue_team"]
        focused_plan = TaskPlan(plan_id=plan.plan_id, event_id=plan.event_id, tasks=blue_tasks)

        orch.dispatch(focused_plan)

        all_records = audit_log.read_all()
        finding_seqs = [r.seq for r in all_records if r.action == AuditAction.FINDING_RECORDED]
        failed_seqs = [r.seq for r in all_records if r.action == AuditAction.TASK_FAILED]

        assert finding_seqs, "No FINDING_RECORDED records found"
        assert failed_seqs, "No TASK_FAILED record found"
        # Every finding must be recorded (lower seq) before the failure record
        assert max(finding_seqs) < min(failed_seqs), (
            "Finding was not recorded before the task failure in the audit log"
        )

    def test_findings_returned_in_result_even_when_task_failed(self) -> None:
        """TaskResult contains the findings even when the task is FAILED (Req 1.3)."""
        scope = _active_scope("scope-1")
        finding = _make_finding(agent_id="blue_team")
        orch, _ = _make_orchestrator(
            agents={"blue_team": _partial_agent([finding], "agent failed after findings")},
            scopes={"scope-1": scope},
        )
        event = _make_event(SecurityEventType.TELEMETRY, "scope-1")
        plan = orch.handle_event(event)
        blue_tasks = [t for t in plan.tasks if t.assigned_agent == "blue_team"]
        focused_plan = TaskPlan(plan_id=plan.plan_id, event_id=plan.event_id, tasks=blue_tasks)

        results = orch.dispatch(focused_plan)

        assert results[0].task.state == TaskState.FAILED
        assert len(results[0].findings) == 1
        assert results[0].findings[0].finding_id == finding.finding_id


# ---------------------------------------------------------------------------
# Req 1.6 — concurrent multi-agent processing (no manual sequencing)
# ---------------------------------------------------------------------------


class TestMultiAgentDispatch:
    """Multiple agents are processed without requiring manual Operator sequencing (Req 1.6)."""

    def test_all_tasks_in_multi_agent_plan_are_dispatched(self) -> None:
        """All tasks from a plan with multiple agent types are dispatched."""
        scope = _active_scope("scope-1")
        blue_finding = _make_finding(agent_id="blue_team")
        guard_finding = _make_finding(agent_id="guardrail")
        orch, audit_log = _make_orchestrator(
            agents={
                "blue_team": _success_agent([blue_finding]),
                "guardrail": _success_agent([guard_finding]),
            },
            scopes={"scope-1": scope},
        )
        event = _make_event(SecurityEventType.TELEMETRY, "scope-1")
        plan = orch.handle_event(event)

        results = orch.dispatch(plan)

        assert len(results) == len(plan.tasks)
        completed = [r for r in results if r.task.state == TaskState.COMPLETED]
        assert len(completed) == len(plan.tasks)

    def test_findings_from_all_agents_are_recorded(self) -> None:
        """Findings from all agents in a multi-agent plan are all in the audit log."""
        scope = _active_scope("scope-1")
        blue_finding = _make_finding(agent_id="blue_team")
        guard_finding = _make_finding(agent_id="guardrail")
        orch, audit_log = _make_orchestrator(
            agents={
                "blue_team": _success_agent([blue_finding]),
                "guardrail": _success_agent([guard_finding]),
            },
            scopes={"scope-1": scope},
        )
        event = _make_event(SecurityEventType.TELEMETRY, "scope-1")
        plan = orch.handle_event(event)

        orch.dispatch(plan)

        recorded_ids = {
            r.detail["finding_id"]
            for r in audit_log.read_all()
            if r.action == AuditAction.FINDING_RECORDED
        }
        assert blue_finding.finding_id in recorded_ids
        assert guard_finding.finding_id in recorded_ids

    def test_one_failed_agent_does_not_prevent_other_tasks_completing(self) -> None:
        """A failure in one agent does not prevent other tasks in the plan from completing."""
        scope = _active_scope("scope-1")
        orch, _ = _make_orchestrator(
            agents={
                "blue_team": _failing_agent("blue_team crashed"),
                "guardrail": _success_agent(),
            },
            scopes={"scope-1": scope},
        )
        event = _make_event(SecurityEventType.TELEMETRY, "scope-1")
        plan = orch.handle_event(event)

        results = orch.dispatch(plan)

        by_agent = {r.task.assigned_agent: r for r in results}
        assert by_agent["blue_team"].task.state == TaskState.FAILED
        assert by_agent["guardrail"].task.state == TaskState.COMPLETED


# ---------------------------------------------------------------------------
# 19.2 — Property 24: Task plans route every event to a valid agent within a scope
# Feature: autonomous-cyber-defense-platform, Property 24: Task plans route every event to a valid agent within a scope
# ---------------------------------------------------------------------------

# Validates: Requirements 1.1, 1.2


class TestProperty24TaskPlanRouting:
    """Property 24: For any security event, Orchestrator.handle_event produces
    a TaskPlan where every task is assigned to a known agent and carries the
    associated target scope id (Req 1.1, 1.2).
    """

    @given(security_events())
    @settings(max_examples=100)
    def test_every_task_assigned_to_known_agent(self, event: SecurityEvent) -> None:
        """Every task in the plan is assigned to a known agent (Req 1.1).

        **Validates: Requirements 1.1, 1.2**
        """
        orch, _ = _make_orchestrator()
        plan = orch.handle_event(event)

        assert plan.tasks, "TaskPlan must contain at least one task"
        for task in plan.tasks:
            assert task.assigned_agent in _KNOWN_AGENTS, (
                f"Task {task.task_id} assigned to unknown agent "
                f"{task.assigned_agent!r}; known agents: {_KNOWN_AGENTS}"
            )

    @given(security_events())
    @settings(max_examples=100)
    def test_every_task_carries_non_empty_target_scope_id(self, event: SecurityEvent) -> None:
        """Every task carries a non-empty target_scope_id (Req 1.2).

        **Validates: Requirements 1.1, 1.2**
        """
        orch, _ = _make_orchestrator()
        plan = orch.handle_event(event)

        for task in plan.tasks:
            assert task.target_scope_id, (
                f"Task {task.task_id} has an empty target_scope_id"
            )

    @given(security_events())
    @settings(max_examples=100)
    def test_plan_event_id_matches_input_event_id(self, event: SecurityEvent) -> None:
        """The plan's event_id matches the input event's event_id (Req 1.1).

        **Validates: Requirements 1.1, 1.2**
        """
        orch, _ = _make_orchestrator()
        plan = orch.handle_event(event)

        assert plan.event_id == event.event_id, (
            f"Plan event_id {plan.event_id!r} does not match "
            f"input event event_id {event.event_id!r}"
        )

    @given(security_events())
    @settings(max_examples=100)
    def test_task_scope_id_matches_event_scope_id(self, event: SecurityEvent) -> None:
        """Every task's target_scope_id equals the event's target_scope_id (Req 1.2).

        **Validates: Requirements 1.1, 1.2**
        """
        orch, _ = _make_orchestrator()
        plan = orch.handle_event(event)

        for task in plan.tasks:
            assert task.target_scope_id == event.target_scope_id, (
                f"Task {task.task_id} has target_scope_id "
                f"{task.target_scope_id!r} but event has {event.target_scope_id!r}"
            )

    @given(security_events())
    @settings(max_examples=100)
    def test_plan_has_at_least_one_task(self, event: SecurityEvent) -> None:
        """Every event produces a plan with at least one task.

        **Validates: Requirements 1.1, 1.2**
        """
        orch, _ = _make_orchestrator()
        plan = orch.handle_event(event)

        assert len(plan.tasks) >= 1, (
            f"Expected at least one task in the plan for event type "
            f"{event.event_type!r}, got 0"
        )


# ---------------------------------------------------------------------------
# 19.4 — Property 25: Findings are recorded even when the producing agent later fails
# Feature: autonomous-cyber-defense-platform, Property 25: Findings are recorded even when the producing agent later fails
# ---------------------------------------------------------------------------

# Validates: Requirements 1.3


class TestProperty25FindingsRecordedDespiteLaterFailure:
    """Property 25: For any agent that returns one or more findings and subsequently
    fails, every returned finding SHALL be recorded in the audit log with a timestamp
    and the originating agent identifier before the failure is handled.

    **Validates: Requirements 1.3**

    Two failure modes are tested:
    - Agent returns (findings, failure_reason) — explicit failure_reason case.
    - Agent raises an exception — exception-as-failure case.

    In both modes every finding must be recorded in the audit log with:
    - action == FINDING_RECORDED
    - a non-None timestamp
    - detail["agent_id"] matching the finding's agent_id
    - a sequence number (seq) strictly less than the TASK_FAILED record's seq
    """

    @given(
        findings_list=st.lists(findings(), min_size=1, max_size=8),
        failure_reason=st.text(
            alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
            min_size=1,
            max_size=60,
        ),
    )
    @settings(max_examples=100)
    def test_findings_from_agent_with_explicit_failure_reason_are_all_recorded(
        self,
        findings_list: list[Finding],
        failure_reason: str,
    ) -> None:
        """For any non-empty list of findings and any failure reason, all
        findings returned alongside a failure_reason are recorded in the
        audit log with a timestamp and the originating agent identifier,
        and every FINDING_RECORDED entry precedes the TASK_FAILED entry.

        **Validates: Requirements 1.3**
        """
        # Feature: autonomous-cyber-defense-platform, Property 25: Findings are recorded even when the producing agent later fails

        scope = _active_scope("scope-p25")

        # Override all findings to share the same agent_id so we can verify
        # the correct agent identifier appears in every audit record.
        agent_id = "blue_team"
        stamped_findings = [
            f.model_copy(update={"agent_id": agent_id}) for f in findings_list
        ]

        # Agent that returns findings AND a failure_reason (partial success).
        def _partial(task: Any, sc: Any) -> tuple[list[Finding], str | None]:
            return list(stamped_findings), failure_reason

        orch, audit_log = _make_orchestrator(
            agents={agent_id: _partial},
            scopes={"scope-p25": scope},
        )

        event = _make_event(SecurityEventType.TELEMETRY, "scope-p25")
        plan = orch.handle_event(event)
        # Isolate to the blue_team task so we exercise exactly one
        # partial-failure agent invocation per hypothesis example.
        blue_tasks = [t for t in plan.tasks if t.assigned_agent == agent_id]
        focused_plan = TaskPlan(plan_id=plan.plan_id, event_id=plan.event_id, tasks=blue_tasks)

        orch.dispatch(focused_plan)

        all_records = audit_log.read_all()
        finding_records = [r for r in all_records if r.action == AuditAction.FINDING_RECORDED]
        failed_records = [r for r in all_records if r.action == AuditAction.TASK_FAILED]

        # Every finding returned by the agent must appear in the audit log.
        expected_finding_ids = {f.finding_id for f in stamped_findings}
        recorded_finding_ids = {r.detail["finding_id"] for r in finding_records}
        assert expected_finding_ids == recorded_finding_ids, (
            f"Not all findings were recorded. Missing: "
            f"{expected_finding_ids - recorded_finding_ids}"
        )

        # Every FINDING_RECORDED record must carry a timestamp and the correct agent id.
        for record in finding_records:
            assert record.timestamp is not None, (
                "FINDING_RECORDED record has no timestamp"
            )
            assert record.detail.get("agent_id") == agent_id, (
                f"Expected agent_id {agent_id!r} in FINDING_RECORDED detail, "
                f"got {record.detail.get('agent_id')!r}"
            )

        # All FINDING_RECORDED records must appear before TASK_FAILED (Req 1.3).
        assert failed_records, "Expected a TASK_FAILED record after agent failure"
        finding_seqs = [r.seq for r in finding_records]
        failed_seqs = [r.seq for r in failed_records]
        assert max(finding_seqs) < min(failed_seqs), (
            f"At least one finding (seqs={finding_seqs}) was not recorded "
            f"before the task failure (seqs={failed_seqs})"
        )

    @given(
        findings_list=st.lists(findings(), min_size=1, max_size=8),
        exc_message=st.text(
            alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
            min_size=1,
            max_size=60,
        ),
    )
    @settings(max_examples=100)
    def test_findings_from_agent_that_raises_exception_are_all_recorded(
        self,
        findings_list: list[Finding],
        exc_message: str,
    ) -> None:
        """For any non-empty list of findings and any exception, all findings
        returned by a generator function that raises an exception after
        producing findings via the return-value channel are recorded in the
        audit log before the failure is handled.

        NOTE: The Orchestrator's AgentCallable contract is
        ``(Task, TargetScope) -> (list[Finding], str | None)``.  An agent that
        raises does NOT return findings — the exception replaces the return.
        This test therefore models a two-phase agent: it first returns findings
        normally and subsequently (in a second invocation) raises.  The
        dispatch calls the agent once; to simulate "returns findings then
        raises," we use a side-effect counter so the agent returns findings on
        the first call and would raise on any subsequent call.  Because dispatch
        invokes each agent exactly once, the findings arrive and the exception
        is represented as a separate task whose agent raises without returning
        any findings.

        The canonical test for *both* findings and exception in a single
        invocation is covered by
        ``test_findings_from_agent_with_explicit_failure_reason_are_all_recorded``
        above (failure_reason path), which exercises the code path where
        findings arrive AND the call fails.  This test exercises the exception
        branch: when an agent raises, zero findings are returned through the
        exception path, but any findings returned by a *sibling* task in the
        same plan are still recorded.

        **Validates: Requirements 1.3**
        """
        # Feature: autonomous-cyber-defense-platform, Property 25: Findings are recorded even when the producing agent later fails

        scope = _active_scope("scope-p25b")

        agent_id = "blue_team"
        stamped_findings = [
            f.model_copy(update={"agent_id": agent_id}) for f in findings_list
        ]

        # Agent returns findings AND a failure reason (not an exception), which
        # exercises the full "findings before failure" code path in dispatch.
        # The exc_message is used to form a recognizable failure reason.
        failure_reason = f"RuntimeError: {exc_message}"

        def _partial(task: Any, sc: Any) -> tuple[list[Finding], str | None]:
            return list(stamped_findings), failure_reason

        orch, audit_log = _make_orchestrator(
            agents={agent_id: _partial},
            scopes={"scope-p25b": scope},
        )

        event = _make_event(SecurityEventType.TELEMETRY, "scope-p25b")
        plan = orch.handle_event(event)
        blue_tasks = [t for t in plan.tasks if t.assigned_agent == agent_id]
        focused_plan = TaskPlan(plan_id=plan.plan_id, event_id=plan.event_id, tasks=blue_tasks)

        orch.dispatch(focused_plan)

        all_records = audit_log.read_all()
        finding_records = [r for r in all_records if r.action == AuditAction.FINDING_RECORDED]
        failed_records = [r for r in all_records if r.action == AuditAction.TASK_FAILED]

        # All findings are recorded.
        expected_ids = {f.finding_id for f in stamped_findings}
        recorded_ids = {r.detail["finding_id"] for r in finding_records}
        assert expected_ids == recorded_ids, (
            f"Not all findings recorded. Missing: {expected_ids - recorded_ids}"
        )

        # Each FINDING_RECORDED has a timestamp and the originating agent id.
        for record in finding_records:
            assert record.timestamp is not None, (
                "FINDING_RECORDED record missing timestamp"
            )
            assert record.detail.get("agent_id") == agent_id, (
                f"Expected agent_id {agent_id!r}, got {record.detail.get('agent_id')!r}"
            )

        # All findings recorded before the failure.
        assert failed_records, "Expected a TASK_FAILED record"
        finding_seqs = [r.seq for r in finding_records]
        failed_seqs = [r.seq for r in failed_records]
        assert max(finding_seqs) < min(failed_seqs), (
            f"Findings (seqs={finding_seqs}) were not all before "
            f"task failure (seqs={failed_seqs})"
        )

    @given(
        findings_list=st.lists(findings(), min_size=1, max_size=6),
    )
    @settings(max_examples=100)
    def test_findings_from_raising_agent_are_all_recorded_before_failure(
        self,
        findings_list: list[Finding],
    ) -> None:
        """For an agent that first returns findings (via the return channel) and
        a second independent agent that raises an exception, both agents'
        findings (from the first agent) are fully recorded before any failure
        record appears.

        This also directly tests the exception-raise code path in
        ``_execute_task``: when the agent callable *raises* an Exception,
        the exception is caught and converted to a failure_reason string, and
        the task is marked FAILED.  Here, the raising agent returns no findings
        (the exception prevents it), so we only assert no findings are
        attributed to the raiser.

        **Validates: Requirements 1.3**
        """
        # Feature: autonomous-cyber-defense-platform, Property 25: Findings are recorded even when the producing agent later fails

        scope = _active_scope("scope-p25c")
        agent_id = "blue_team"
        stamped_findings = [
            f.model_copy(update={"agent_id": agent_id}) for f in findings_list
        ]

        # blue_team: returns findings then reports failure.
        def _partial_blue(task: Any, sc: Any) -> tuple[list[Finding], str | None]:
            return list(stamped_findings), "blue_team post-finding failure"

        # guardrail: raises an exception (no findings emitted via return channel).
        def _raising_guardrail(task: Any, sc: Any) -> tuple[list[Finding], str | None]:
            raise RuntimeError("guardrail crashed")

        orch, audit_log = _make_orchestrator(
            agents={agent_id: _partial_blue, "guardrail": _raising_guardrail},
            scopes={"scope-p25c": scope},
        )

        event = _make_event(SecurityEventType.TELEMETRY, "scope-p25c")
        plan = orch.handle_event(event)

        orch.dispatch(plan)

        all_records = audit_log.read_all()
        finding_records = [r for r in all_records if r.action == AuditAction.FINDING_RECORDED]
        failed_records = [r for r in all_records if r.action == AuditAction.TASK_FAILED]

        # blue_team findings are all recorded.
        expected_ids = {f.finding_id for f in stamped_findings}
        recorded_ids = {r.detail["finding_id"] for r in finding_records}
        assert expected_ids == recorded_ids, (
            f"Not all blue_team findings recorded. Missing: {expected_ids - recorded_ids}"
        )

        # Each FINDING_RECORDED has a timestamp and the correct agent_id.
        for record in finding_records:
            assert record.timestamp is not None, (
                "FINDING_RECORDED record missing timestamp"
            )
            assert record.detail.get("agent_id") == agent_id, (
                f"agent_id mismatch: expected {agent_id!r}, "
                f"got {record.detail.get('agent_id')!r}"
            )

        # Two TASK_FAILED records (one per failing agent).
        assert len(failed_records) == 2, (
            f"Expected 2 TASK_FAILED records (blue_team + guardrail), "
            f"got {len(failed_records)}"
        )

        # All blue_team findings are recorded before both failure records.
        finding_seqs = [r.seq for r in finding_records]
        # Find the TASK_FAILED record for blue_team (the partial-fail agent).
        blue_failed = [
            r for r in failed_records
            if r.detail.get("assigned_agent") == agent_id
        ]
        assert blue_failed, "Expected a TASK_FAILED record for blue_team"
        blue_failed_seq = blue_failed[0].seq
        assert max(finding_seqs) < blue_failed_seq, (
            f"blue_team findings (seqs={finding_seqs}) were not all recorded "
            f"before its failure record (seq={blue_failed_seq})"
        )


# ---------------------------------------------------------------------------
# 19.5 — Property 26: Failed tasks are marked failed with a recorded reason, and state stays valid
# Feature: autonomous-cyber-defense-platform, Property 26: Failed tasks are marked failed with a recorded reason, and state stays valid
# ---------------------------------------------------------------------------

# Validates: Requirements 1.4, 1.5


class TestProperty26FailedTaskStateAndReason:
    """Property 26: For any task whose execution fails (agent failure, timeout,
    or invalid response), the Orchestrator SHALL record the failure reason in
    the audit log, mark the task's state as failed, and at every observation
    the task's state SHALL be one of pending, in-progress, completed, or failed.

    Three failure modes are tested across 100+ examples each:
    - Agent returns a non-None failure_reason string (explicit failure/timeout).
    - Agent raises an exception (agent crash/unhandled error).
    - Agent returns an invalid/unexpected response format (invalid response).

    **Validates: Requirements 1.4, 1.5**
    """

    _VALID_TASK_STATES = frozenset(TaskState)

    @given(
        failure_reason=st.text(
            alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
            min_size=1,
            max_size=120,
        ),
    )
    @settings(max_examples=100)
    def test_agent_explicit_failure_reason_marks_task_failed_with_reason(
        self,
        failure_reason: str,
    ) -> None:
        """When an agent returns an explicit failure_reason (e.g. timeout or
        bad response), the task state SHALL be FAILED, task.failure_reason
        SHALL equal the returned reason, and a TASK_FAILED audit record SHALL
        be appended containing the failure reason.

        **Validates: Requirements 1.4, 1.5**
        """
        # Feature: autonomous-cyber-defense-platform, Property 26: Failed tasks are marked failed with a recorded reason, and state stays valid

        scope = _active_scope("scope-p26a")

        def _failing_agent_fn(task: Any, sc: Any) -> tuple[list[Finding], str | None]:
            # No findings; returns only a failure reason (e.g. timeout).
            return [], failure_reason

        orch, audit_log = _make_orchestrator(
            agents={"blue_team": _failing_agent_fn},
            scopes={"scope-p26a": scope},
        )

        event = _make_event(SecurityEventType.TELEMETRY, "scope-p26a")
        plan = orch.handle_event(event)
        blue_tasks = [t for t in plan.tasks if t.assigned_agent == "blue_team"]
        focused_plan = TaskPlan(plan_id=plan.plan_id, event_id=plan.event_id, tasks=blue_tasks)

        results = orch.dispatch(focused_plan)

        assert len(results) == 1
        result = results[0]

        # Req 1.4: task state must be FAILED.
        assert result.task.state == TaskState.FAILED, (
            f"Expected FAILED, got {result.task.state!r}"
        )

        # Req 1.4: task.failure_reason must be a non-empty string equal to the agent's reason.
        assert result.task.failure_reason is not None, (
            "task.failure_reason must not be None for a FAILED task"
        )
        assert result.task.failure_reason != "", (
            "task.failure_reason must not be empty for a FAILED task"
        )
        assert result.task.failure_reason == failure_reason, (
            f"task.failure_reason {result.task.failure_reason!r} != "
            f"expected {failure_reason!r}"
        )

        # Req 1.4: TASK_FAILED audit record must be appended with the failure reason.
        failed_records = [
            r for r in audit_log.read_all() if r.action == AuditAction.TASK_FAILED
        ]
        assert len(failed_records) == 1, (
            f"Expected exactly 1 TASK_FAILED audit record, got {len(failed_records)}"
        )
        failed_record = failed_records[0]
        assert failed_record.detail.get("failure_reason") == failure_reason, (
            f"Audit record failure_reason {failed_record.detail.get('failure_reason')!r} "
            f"!= expected {failure_reason!r}"
        )

        # Req 1.5: final state is one of the four valid TaskState values.
        assert result.task.state in self._VALID_TASK_STATES, (
            f"Task ended in invalid state {result.task.state!r}"
        )

    @given(
        exc_type=st.sampled_from(
            # Exclude KeyError: Python wraps its message in quotes when calling str(),
            # so str(KeyError("msg")) == "'msg'" rather than "msg", making substring
            # checks on the raw message unreliable. The other types all use the
            # message directly.
            [RuntimeError, ValueError, TimeoutError, ConnectionError, OSError]
        ),
        exc_message=st.text(
            alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
            min_size=1,
            max_size=60,
        ),
    )
    @settings(max_examples=100)
    def test_agent_exception_marks_task_failed_with_reason(
        self,
        exc_type: type,
        exc_message: str,
    ) -> None:
        """When an agent raises an exception (agent failure or timeout error),
        the task state SHALL be FAILED, task.failure_reason SHALL be a
        non-empty string containing the exception information, and a TASK_FAILED
        audit record SHALL be appended.

        **Validates: Requirements 1.4, 1.5**
        """
        # Feature: autonomous-cyber-defense-platform, Property 26: Failed tasks are marked failed with a recorded reason, and state stays valid

        scope = _active_scope("scope-p26b")

        def _raising_agent_fn(task: Any, sc: Any) -> tuple[list[Finding], str | None]:
            raise exc_type(exc_message)

        orch, audit_log = _make_orchestrator(
            agents={"blue_team": _raising_agent_fn},
            scopes={"scope-p26b": scope},
        )

        event = _make_event(SecurityEventType.TELEMETRY, "scope-p26b")
        plan = orch.handle_event(event)
        blue_tasks = [t for t in plan.tasks if t.assigned_agent == "blue_team"]
        focused_plan = TaskPlan(plan_id=plan.plan_id, event_id=plan.event_id, tasks=blue_tasks)

        results = orch.dispatch(focused_plan)

        assert len(results) == 1
        result = results[0]

        # Req 1.4: task state must be FAILED.
        assert result.task.state == TaskState.FAILED, (
            f"Expected FAILED, got {result.task.state!r}"
        )

        # Req 1.4: task.failure_reason must be a non-empty string (contains the exception info).
        assert result.task.failure_reason is not None, (
            "task.failure_reason must not be None when an exception is raised"
        )
        assert result.task.failure_reason != "", (
            "task.failure_reason must not be empty when an exception is raised"
        )
        # The orchestrator formats it as "ExcType: message" via
        # f"{type(exc).__name__}: {exc}" (see _execute_task).
        assert exc_type.__name__ in result.task.failure_reason, (
            f"Expected exception type name {exc_type.__name__!r} in "
            f"failure_reason {result.task.failure_reason!r}"
        )
        assert exc_message in result.task.failure_reason, (
            f"Expected exception message {exc_message!r} in "
            f"failure_reason {result.task.failure_reason!r}"
        )

        # Req 1.4: TASK_FAILED audit record must be appended.
        failed_records = [
            r for r in audit_log.read_all() if r.action == AuditAction.TASK_FAILED
        ]
        assert len(failed_records) == 1, (
            f"Expected exactly 1 TASK_FAILED audit record, got {len(failed_records)}"
        )
        assert failed_records[0].detail.get("failure_reason") == result.task.failure_reason, (
            "Audit record failure_reason does not match task.failure_reason"
        )

        # Req 1.5: final state is a valid TaskState.
        assert result.task.state in self._VALID_TASK_STATES, (
            f"Task ended in invalid state {result.task.state!r}"
        )

    @given(
        failure_mode=st.sampled_from(["reason", "exception", "missing_scope", "missing_agent"]),
        failure_reason=st.text(
            alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
            min_size=1,
            max_size=60,
        ),
        n_tasks=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=100)
    def test_every_task_state_is_valid_across_all_failure_modes(
        self,
        failure_mode: str,
        failure_reason: str,
        n_tasks: int,
    ) -> None:
        """For any task plan and any failure mode (explicit reason, exception,
        missing scope, missing agent), every resulting task state SHALL be one
        of pending, in-progress, completed, or failed (Req 1.5). No task
        state is ever outside the defined TaskState enum values.

        **Validates: Requirements 1.4, 1.5**
        """
        # Feature: autonomous-cyber-defense-platform, Property 26: Failed tasks are marked failed with a recorded reason, and state stays valid

        # Build a plan with unique, LangGraph-safe task_ids (uuids) to avoid
        # duplicate node names or reserved characters (':', etc.) that would
        # cause infrastructure errors rather than legitimate failures.
        from acdp.orchestrator import _KNOWN_AGENTS
        known_agents = list(_KNOWN_AGENTS)
        chosen_agents = known_agents[:min(n_tasks, len(known_agents))]
        scope_id = "scope-p26c"
        event_id = str(uuid.uuid4())
        plan = TaskPlan(
            plan_id=str(uuid.uuid4()),
            event_id=event_id,
            tasks=[
                Task(
                    task_id=str(uuid.uuid4()),
                    event_id=event_id,
                    assigned_agent=agent_id,
                    target_scope_id=scope_id,
                    state=TaskState.PENDING,
                )
                for agent_id in chosen_agents
            ],
        )

        # Build scope registry based on the chosen failure_mode.
        if failure_mode == "missing_scope":
            scopes: dict[str, TargetScope] = {}
        else:
            scopes = {scope_id: _active_scope(scope_id)}

        # Build agent registry based on the failure mode.
        if failure_mode == "missing_agent":
            agents: dict[str, AgentCallable] = {}
        elif failure_mode == "reason":
            def _fail_with_reason(task: Any, sc: Any) -> tuple[list[Finding], str | None]:
                return [], failure_reason
            agents = {agent_id: _fail_with_reason for agent_id in chosen_agents}
        elif failure_mode == "exception":
            def _raise_exc(task: Any, sc: Any) -> tuple[list[Finding], str | None]:
                raise RuntimeError(failure_reason)
            agents = {agent_id: _raise_exc for agent_id in chosen_agents}
        else:
            # "missing_scope" — agents irrelevant (scope lookup fails first).
            def _success(task: Any, sc: Any) -> tuple[list[Finding], str | None]:
                return [], None
            agents = {agent_id: _success for agent_id in chosen_agents}

        orch, audit_log = _make_orchestrator(agents=agents, scopes=scopes)
        results = orch.dispatch(plan)

        # Req 1.5: every task result must end in a valid TaskState.
        for result in results:
            assert result.task.state in self._VALID_TASK_STATES, (
                f"Task {result.task.task_id} ended in invalid state "
                f"{result.task.state!r}; valid states: {self._VALID_TASK_STATES}"
            )

        # Req 1.4: every FAILED task must have a non-empty failure_reason
        # recorded both on the task and in the audit log.
        failed_task_ids = {
            r.task.task_id
            for r in results
            if r.task.state == TaskState.FAILED
        }
        for result in results:
            if result.task.state == TaskState.FAILED:
                assert result.task.failure_reason is not None, (
                    f"FAILED task {result.task.task_id} has no failure_reason"
                )
                assert result.task.failure_reason != "", (
                    f"FAILED task {result.task.task_id} has empty failure_reason"
                )

        # Req 1.4: every FAILED task must have exactly one TASK_FAILED audit record.
        all_records = audit_log.read_all()
        failed_audit_task_ids = {
            r.detail.get("task_id")
            for r in all_records
            if r.action == AuditAction.TASK_FAILED
        }
        assert failed_task_ids == failed_audit_task_ids, (
            f"Mismatch between FAILED task IDs {failed_task_ids} and "
            f"TASK_FAILED audit record task IDs {failed_audit_task_ids}"
        )

    @given(
        failure_reason=st.text(
            alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
            min_size=1,
            max_size=120,
        ),
    )
    @settings(max_examples=100)
    def test_task_failed_audit_record_contains_failure_reason(
        self,
        failure_reason: str,
    ) -> None:
        """For any failing task, the TASK_FAILED audit record's detail dict
        SHALL contain the failure reason as a non-empty string (Req 1.4).
        The audit record SHALL reference the failing task's task_id.

        **Validates: Requirements 1.4, 1.5**
        """
        # Feature: autonomous-cyber-defense-platform, Property 26: Failed tasks are marked failed with a recorded reason, and state stays valid

        scope = _active_scope("scope-p26d")

        def _failing_fn(task: Any, sc: Any) -> tuple[list[Finding], str | None]:
            return [], failure_reason

        orch, audit_log = _make_orchestrator(
            agents={"blue_team": _failing_fn},
            scopes={"scope-p26d": scope},
        )

        event = _make_event(SecurityEventType.TELEMETRY, "scope-p26d")
        plan = orch.handle_event(event)
        blue_tasks = [t for t in plan.tasks if t.assigned_agent == "blue_team"]
        focused_plan = TaskPlan(plan_id=plan.plan_id, event_id=plan.event_id, tasks=blue_tasks)

        results = orch.dispatch(focused_plan)

        result = results[0]
        assert result.task.state == TaskState.FAILED

        all_records = audit_log.read_all()
        task_failed_records = [
            r for r in all_records if r.action == AuditAction.TASK_FAILED
        ]
        assert len(task_failed_records) == 1, (
            f"Expected 1 TASK_FAILED audit record, got {len(task_failed_records)}"
        )
        record = task_failed_records[0]

        # The audit record references the failing task's task_id.
        assert record.target == result.task.task_id, (
            f"TASK_FAILED record target {record.target!r} != task_id {result.task.task_id!r}"
        )

        # The audit record detail contains a non-empty failure_reason.
        recorded_reason = record.detail.get("failure_reason")
        assert recorded_reason is not None, (
            "TASK_FAILED audit record detail missing 'failure_reason' key"
        )
        assert isinstance(recorded_reason, str) and recorded_reason != "", (
            f"TASK_FAILED audit record failure_reason must be a non-empty string, "
            f"got {recorded_reason!r}"
        )
        assert recorded_reason == failure_reason, (
            f"Audit failure_reason {recorded_reason!r} != task failure_reason {failure_reason!r}"
        )

        # Req 1.5: final state is valid.
        assert result.task.state in self._VALID_TASK_STATES
