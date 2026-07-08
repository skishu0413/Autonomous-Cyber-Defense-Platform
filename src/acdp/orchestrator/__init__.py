"""LangGraph-based Orchestrator: planning and dispatch (Layer 4).

The :class:`Orchestrator` performs two duties:

1. :meth:`Orchestrator.handle_event` — given a :class:`~acdp.models.SecurityEvent`,
   produce a :class:`~acdp.models.TaskPlan` that routes the event to the
   responsible agent(s). The routing rules are:

   * ``TELEMETRY`` → ``blue_team``
   * ``PROBE_REQUEST`` → ``red_team``
   * ``VULNERABILITY_FINDING`` → ``devsecops``
   * ``PROMPT`` and ``UNKNOWN`` → ``guardrail``

   Every event is *also* always routed to ``guardrail`` as the safety layer
   (Req 1.1, 1.2).

2. :meth:`Orchestrator.dispatch` — execute a :class:`~acdp.models.TaskPlan`
   via a LangGraph :class:`~langgraph.graph.StateGraph`. Each task runs as a
   separate node; the typed graph state tracks each task's
   :class:`~acdp.models.TaskState` (``pending → in-progress → completed |
   failed``). Findings are recorded in the :class:`~acdp.audit.AuditLog` as
   they arrive — including findings from an agent that subsequently fails —
   before the failure is handled (Req 1.3, 1.4, 1.5, 1.6).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, TypedDict

from langgraph.graph import StateGraph, END

from acdp.audit import AuditLog
from acdp.models import (
    AuditAction,
    AuditRecord,
    Finding,
    SecurityEvent,
    SecurityEventType,
    Task,
    TaskPlan,
    TaskResult,
    TaskState,
    TargetScope,
)

__all__ = ["Orchestrator", "AgentCallable"]

# Type alias for the agent dispatch callable: receives a Task and its
# TargetScope and returns (findings, failure_reason). The findings list may
# be non-empty even when failure_reason is set (agent returned partial results
# before failing), which is the Req 1.3 "record findings before handling
# failure" case.
AgentCallable = Callable[[Task, TargetScope], tuple[list[Finding], str | None]]


# ---------------------------------------------------------------------------
# Routing map: SecurityEventType -> list[agent identifiers]
# ---------------------------------------------------------------------------

# Primary agents responsible for each event type (Req 1.1). The guardrail
# agent is always included as the safety layer.
_EVENT_TYPE_TO_AGENTS: dict[SecurityEventType, list[str]] = {
    SecurityEventType.TELEMETRY: ["blue_team", "guardrail"],
    SecurityEventType.PROBE_REQUEST: ["red_team", "guardrail"],
    SecurityEventType.VULNERABILITY_FINDING: ["devsecops", "guardrail"],
    SecurityEventType.PROMPT: ["guardrail"],
    SecurityEventType.UNKNOWN: ["guardrail"],
}

# Canonical set of known agent identifiers.
_KNOWN_AGENTS = frozenset(["guardrail", "blue_team", "red_team", "devsecops"])


# ---------------------------------------------------------------------------
# LangGraph state type
# ---------------------------------------------------------------------------

class _DispatchState(TypedDict):
    """Typed state threaded through the LangGraph dispatch graph.

    ``task_results`` accumulates :class:`~acdp.models.TaskResult` objects
    as each task node completes (successfully or with failure). The state is
    intentionally kept flat and serializable so LangGraph can checkpoint it.
    """
    task_results: list[TaskResult]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Central planning and dispatch component (Layer 4 — Orchestration).

    The Orchestrator is the single entry point for security events. It produces
    task plans and dispatches them to agents via a LangGraph state machine.

    Constructor Args:
        audit_log: Append-only log; finding-recorded and task-failed events are
            written here during dispatch (Req 1.3, 1.4, 12.1).
        agent_registry: A mapping from agent identifier (e.g. ``"blue_team"``) to
            an :data:`AgentCallable`. Each callable executes the task and returns
            ``(findings, failure_reason)``. Use this injection point to wire in
            the real agent implementations or lightweight stubs for testing.
        scope_registry: A mapping from ``target_scope_id`` to
            :class:`~acdp.models.TargetScope`. Used to resolve the scope for
            each task during dispatch. When a scope_id is not found, the task is
            marked failed.
        actor_id: Actor identifier recorded on audit records; defaults to
            ``"orchestrator"``.
    """

    def __init__(
        self,
        audit_log: AuditLog,
        agent_registry: dict[str, AgentCallable],
        scope_registry: dict[str, TargetScope],
        *,
        actor_id: str = "orchestrator",
    ) -> None:
        self._audit_log = audit_log
        self._agent_registry = agent_registry
        self._scope_registry = scope_registry
        self._actor_id = actor_id

    # ------------------------------------------------------------------
    # 19.1 Event handling and planning  (Req 1.1, 1.2)
    # ------------------------------------------------------------------

    def handle_event(self, event: SecurityEvent) -> TaskPlan:
        """Create a :class:`~acdp.models.TaskPlan` routing ``event`` to agent(s).

        Routes the event to the responsible agent(s) based on
        :attr:`~acdp.models.SecurityEvent.event_type` (Req 1.1). Every task in
        the plan carries the event's ``target_scope_id`` so the dispatching agent
        can enforce scope constraints (Req 1.2).

        Returns the produced :class:`~acdp.models.TaskPlan`.
        """
        agent_ids = _EVENT_TYPE_TO_AGENTS.get(
            event.event_type, ["guardrail"]
        )
        # De-duplicate while preserving order (guardrail is already included
        # in the routing map for every type, but explicit PROMPT events should
        # not create two guardrail tasks).
        seen: set[str] = set()
        unique_agents: list[str] = []
        for aid in agent_ids:
            if aid not in seen:
                seen.add(aid)
                unique_agents.append(aid)

        tasks = [
            Task(
                task_id=str(uuid.uuid4()),
                event_id=event.event_id,
                assigned_agent=agent_id,
                target_scope_id=event.target_scope_id,
                state=TaskState.PENDING,
            )
            for agent_id in unique_agents
        ]

        return TaskPlan(
            plan_id=str(uuid.uuid4()),
            event_id=event.event_id,
            tasks=tasks,
        )

    # ------------------------------------------------------------------
    # 19.3 LangGraph dispatch state machine  (Req 1.3, 1.4, 1.5, 1.6)
    # ------------------------------------------------------------------

    def dispatch(self, plan: TaskPlan) -> list[TaskResult]:
        """Dispatch every task in ``plan`` via a LangGraph state machine.

        Each task is represented as a LangGraph node. The typed graph state
        tracks each task's result as it arrives. Task state transitions follow
        the lifecycle: ``pending → in-progress → completed | failed`` (Req 1.5).

        Findings returned by an agent are recorded in the audit log *before*
        any subsequent failure for that task is handled (Req 1.3). Failed tasks
        have their ``failure_reason`` set and state = ``FAILED`` (Req 1.4).
        Multiple tasks run as independent graph nodes, allowing the platform to
        process multiple agent responses without requiring the Operator to
        sequence them manually (Req 1.6).

        Returns a list of :class:`~acdp.models.TaskResult` — one per task in
        the plan.
        """
        if not plan.tasks:
            return []

        graph = self._build_dispatch_graph(plan)
        initial_state: _DispatchState = {"task_results": []}
        final_state = graph.invoke(initial_state)
        return final_state["task_results"]

    # ------------------------------------------------------------------
    # Internal: LangGraph graph construction
    # ------------------------------------------------------------------

    def _build_dispatch_graph(self, plan: TaskPlan) -> Any:
        """Build a LangGraph StateGraph with one node per task in ``plan``.

        Nodes are executed in definition order; each accumulates its
        :class:`~acdp.models.TaskResult` into the shared state. Because
        LangGraph runs the graph sequentially by default for a single-threaded
        StateGraph, every task is processed without requiring manual Operator
        sequencing (Req 1.6).
        """
        builder: StateGraph = StateGraph(_DispatchState)

        # Build one node closure per task so each node captures its own task
        # reference rather than a shared loop variable.
        node_names: list[str] = []
        for task in plan.tasks:
            node_name = f"task_{task.task_id}"
            node_fn = self._make_task_node(task)
            builder.add_node(node_name, node_fn)
            node_names.append(node_name)

        # Chain the nodes sequentially: START → node_0 → node_1 → … → END.
        # This guarantees Req 1.6 (no manual sequencing needed) by having the
        # graph engine handle execution order.
        if node_names:
            builder.set_entry_point(node_names[0])
            for i in range(len(node_names) - 1):
                builder.add_edge(node_names[i], node_names[i + 1])
            builder.add_edge(node_names[-1], END)

        return builder.compile()

    def _make_task_node(
        self, task: Task
    ) -> Callable[[_DispatchState], _DispatchState]:
        """Return a LangGraph node function that dispatches ``task``.

        The closure captures ``task`` by value so each node invocation operates
        on its own task object, ensuring independent state transitions.
        """
        # Snapshot task fields so this closure is independent of the original
        # Task object (avoids shared-state issues if the caller mutates it).
        task_snapshot = task.model_copy()

        def _node(state: _DispatchState) -> _DispatchState:
            result = self._execute_task(task_snapshot)
            # Accumulate results (LangGraph merges returned state fragments).
            return {"task_results": state["task_results"] + [result]}

        return _node

    def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task and return its :class:`~acdp.models.TaskResult`.

        Lifecycle:
        1. Mark the task ``IN_PROGRESS``.
        2. Resolve the target scope from the scope registry.
        3. Invoke the assigned agent; capture any findings returned before
           an error, and record them in the audit log immediately (Req 1.3).
        4. Record the failure in the audit log and mark the task ``FAILED``
           when the agent raises or returns a failure_reason (Req 1.4).
        5. Mark the task ``COMPLETED`` on success.

        Task state is always one of the four valid values (Req 1.5).
        """
        now = datetime.now(timezone.utc)

        # Req 1.5: transition to IN_PROGRESS before invoking the agent.
        in_progress_task = task.model_copy(update={"state": TaskState.IN_PROGRESS})

        # Resolve the scope; if missing, fail immediately (no scope → unsafe).
        scope = self._scope_registry.get(in_progress_task.target_scope_id)
        if scope is None:
            failure_reason = (
                f"target_scope_id {in_progress_task.target_scope_id!r} not found "
                "in scope registry"
            )
            failed_task = in_progress_task.model_copy(
                update={"state": TaskState.FAILED, "failure_reason": failure_reason}
            )
            self._record_task_failed(failed_task, failure_reason, now)
            return TaskResult(task=failed_task, findings=[], failure_reason=failure_reason)

        # Resolve the agent callable; if unknown, fail immediately.
        agent_fn = self._agent_registry.get(in_progress_task.assigned_agent)
        if agent_fn is None:
            failure_reason = (
                f"agent {in_progress_task.assigned_agent!r} not found in agent registry"
            )
            failed_task = in_progress_task.model_copy(
                update={"state": TaskState.FAILED, "failure_reason": failure_reason}
            )
            self._record_task_failed(failed_task, failure_reason, now)
            return TaskResult(task=failed_task, findings=[], failure_reason=failure_reason)

        # Invoke the agent. Collect any findings it emits *before* checking for
        # failure, so findings from a partially-succeeded call are always recorded
        # (Req 1.3).
        findings: list[Finding] = []
        failure_reason: str | None = None
        try:
            findings, failure_reason = agent_fn(in_progress_task, scope)
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"

        # Req 1.3: record every finding that arrived, including those from an
        # agent that subsequently failed, BEFORE handling the failure.
        for finding in findings:
            self._record_finding(finding, now)

        if failure_reason is not None:
            # Req 1.4: record the failure reason and mark the task FAILED.
            failed_task = in_progress_task.model_copy(
                update={"state": TaskState.FAILED, "failure_reason": failure_reason}
            )
            self._record_task_failed(failed_task, failure_reason, now)
            return TaskResult(
                task=failed_task,
                findings=findings,
                failure_reason=failure_reason,
            )

        # Success: mark the task COMPLETED.
        completed_task = in_progress_task.model_copy(
            update={"state": TaskState.COMPLETED}
        )
        return TaskResult(task=completed_task, findings=findings, failure_reason=None)

    # ------------------------------------------------------------------
    # Audit helpers
    # ------------------------------------------------------------------

    def _record_finding(self, finding: Finding, timestamp: datetime) -> None:
        """Append a FINDING_RECORDED audit record for ``finding`` (Req 1.3, 12.1)."""
        record = AuditRecord(
            timestamp=timestamp,
            actor_id=self._actor_id,
            action=AuditAction.FINDING_RECORDED,
            outcome="recorded",
            target=finding.finding_id,
            detail={
                "finding_id": finding.finding_id,
                "agent_id": finding.agent_id,
                "originating_event_id": finding.originating_event_id,
                "severity": finding.severity.value,
                "title": finding.title,
            },
        )
        self._audit_log.append(record)

    def _record_task_failed(
        self, task: Task, failure_reason: str, timestamp: datetime
    ) -> None:
        """Append a TASK_FAILED audit record for ``task`` (Req 1.4, 12.1)."""
        record = AuditRecord(
            timestamp=timestamp,
            actor_id=self._actor_id,
            action=AuditAction.TASK_FAILED,
            outcome="failed",
            target=task.task_id,
            detail={
                "task_id": task.task_id,
                "event_id": task.event_id,
                "assigned_agent": task.assigned_agent,
                "target_scope_id": task.target_scope_id,
                "failure_reason": failure_reason,
            },
        )
        self._audit_log.append(record)
