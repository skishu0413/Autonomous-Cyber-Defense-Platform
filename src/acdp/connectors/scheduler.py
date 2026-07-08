"""Scheduler — asyncio-based cron/interval probe scheduler.

Uses ``asyncio.sleep()`` exclusively (no APScheduler dependency). Supports two
scheduling modes:

* ``interval`` — sleep ``interval_seconds`` between runs
* ``cron``     — parse a five-field cron expression with the stdlib-only
                  ``_CronParser`` and sleep until the next matching UTC time

Per-target probe flow:
  1. AuthorizationService.authorize(...)  → DENY → audit + skip
  2. RedTeamAgent.plan_probe(...)
  3. RedTeamAgent.execute_probe(...)
  4. For each Finding: audit FINDING_RECORDED (+ optional DevSecOps remediation)
  5. Audit next-run-at record
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from acdp.audit import AuditLog
from acdp.connectors.base import BaseConnector
from acdp.connectors.config import SchedulerConfig
from acdp.models import (
    ActionRequest,
    AuditAction,
    AuditRecord,
    TargetScope,
)

if TYPE_CHECKING:
    from acdp.agents.devsecops_agent import DevSecOpsAgent
    from acdp.agents.red_team_agent import RedTeamAgent
    from acdp.authorization import AuthorizationService

__all__ = ["Scheduler"]


# ---------------------------------------------------------------------------
# Minimal cron parser (stdlib only)
# ---------------------------------------------------------------------------

class _CronParser:
    """Parse a five-field cron expression and compute the next UTC fire time.

    Fields: minute  hour  day-of-month  month  day-of-week
    Supports ``*`` (any), ``*/n`` (step), individual values, and comma-separated
    lists. Does NOT support ``L``, ``#``, or ``?`` (Quartz extensions).
    """

    def __init__(self, expression: str) -> None:
        parts = expression.strip().split()
        if len(parts) != 5:
            raise ValueError(
                f"Cron expression must have exactly 5 fields, got: {expression!r}"
            )
        self._minute = self._parse_field(parts[0], 0, 59)
        self._hour = self._parse_field(parts[1], 0, 23)
        self._dom = self._parse_field(parts[2], 1, 31)
        self._month = self._parse_field(parts[3], 1, 12)
        self._dow = self._parse_field(parts[4], 0, 6)

    @staticmethod
    def _parse_field(field: str, lo: int, hi: int) -> frozenset[int]:
        """Return the set of matching values for a single cron field."""
        result: set[int] = set()
        for part in field.split(","):
            part = part.strip()
            if part == "*":
                result.update(range(lo, hi + 1))
            elif part.startswith("*/"):
                step = int(part[2:])
                result.update(range(lo, hi + 1, step))
            elif "-" in part and "/" in part:
                range_part, step_str = part.split("/")
                start, end = range_part.split("-")
                result.update(range(int(start), int(end) + 1, int(step_str)))
            elif "-" in part:
                start, end = part.split("-")
                result.update(range(int(start), int(end) + 1))
            else:
                result.add(int(part))
        return frozenset(result)

    def next_fire(self, after: datetime) -> datetime:
        """Return the next UTC datetime after ``after`` that matches the expression."""
        # Truncate to the next minute (seconds and microseconds are not relevant)
        dt = after.replace(second=0, microsecond=0, tzinfo=timezone.utc) + timedelta(minutes=1)

        # Search up to 4 years ahead to avoid infinite loops on impossible expressions
        limit = dt + timedelta(days=366 * 4)

        while dt < limit:
            if dt.month not in self._month:
                # Skip to the first day of the next matching month
                dt = dt.replace(day=1, hour=0, minute=0)
                dt += timedelta(days=32)
                dt = dt.replace(day=1)
                continue
            if dt.day not in self._dom:
                dt = dt.replace(hour=0, minute=0) + timedelta(days=1)
                continue
            if dt.weekday() not in self._dow:
                # datetime.weekday() returns 0=Monday…6=Sunday
                # cron uses 0=Sunday…6=Saturday
                cron_dow = (dt.weekday() + 1) % 7
                if cron_dow not in self._dow:
                    dt = dt.replace(hour=0, minute=0) + timedelta(days=1)
                    continue
            if dt.hour not in self._hour:
                dt = dt.replace(minute=0) + timedelta(hours=1)
                continue
            if dt.minute not in self._minute:
                dt += timedelta(minutes=1)
                continue
            return dt

        raise ValueError(f"No next fire time found for cron expression within 4 years")


# ---------------------------------------------------------------------------
# Scheduler connector
# ---------------------------------------------------------------------------

class Scheduler(BaseConnector):
    """Asyncio-based probe scheduler for the Red Team / DevSecOps pipeline.

    Constructor Args:
        config: SchedulerConfig instance.
        red_team_agent: RedTeamAgent to plan and execute probes.
        devsecops_agent: DevSecOpsAgent for optional auto-remediation.
        authz_service: AuthorizationService for authorization gating.
        audit_log: Shared AuditLog instance from the Platform.
        scope: TargetScope used for authorization and probe execution.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        red_team_agent: "RedTeamAgent",
        devsecops_agent: "DevSecOpsAgent",
        authz_service: "AuthorizationService",
        audit_log: AuditLog,
        scope: TargetScope,
    ) -> None:
        self._config = config
        self._red_team = red_team_agent
        self._devsecops = devsecops_agent
        self._authz = authz_service
        self._audit_log = audit_log
        self._scope = scope
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the scheduling loop as an asyncio background task."""
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Signal the loop to stop and wait for it to finish."""
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    # ------------------------------------------------------------------
    # Scheduling loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Main scheduling loop — runs probes then sleeps until the next fire."""
        while not self._stop_event.is_set():
            await self._run_all_targets()

            if self._stop_event.is_set():
                break

            sleep_seconds = self._compute_sleep()
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._stop_event.wait()),
                    timeout=sleep_seconds,
                )
                # If we get here, stop_event was set
                break
            except asyncio.TimeoutError:
                # Normal: the sleep elapsed without being interrupted
                pass

    def _compute_sleep(self) -> float:
        """Return seconds to sleep before the next scheduled run."""
        if self._config.mode == "interval":
            return self._config.interval_seconds

        if self._config.mode == "cron":
            if not self._config.cron_expression:
                return self._config.interval_seconds  # fallback
            parser = _CronParser(self._config.cron_expression)
            now = datetime.now(timezone.utc)
            next_fire = parser.next_fire(now)
            delta = (next_fire - now).total_seconds()
            return max(0.0, delta)

        # Unknown mode — fallback to interval
        return self._config.interval_seconds

    # ------------------------------------------------------------------
    # Per-run probe flow
    # ------------------------------------------------------------------

    async def _run_all_targets(self) -> None:
        """Execute the probe flow for all configured targets in one scheduled run."""
        from acdp.agents.red_team_agent import ProbeTask
        from acdp.agents.devsecops_agent import PullRequest, Declination

        now = datetime.now(timezone.utc)

        for target in self._config.targets:
            if self._stop_event.is_set():
                break

            try:
                await self._probe_target(target, now)
            except Exception as exc:
                # Unhandled exception per-probe: write TASK_FAILED and continue
                try:
                    self._audit_log.append(AuditRecord(
                        timestamp=datetime.now(timezone.utc),
                        actor_id="scheduler",
                        action=AuditAction.TASK_FAILED,
                        outcome="probe_exception",
                        target=target,
                        detail={"error": str(exc), "target": target},
                    ))
                except Exception:
                    pass

        # Write next-run-at audit record after completing all targets
        next_sleep = self._compute_sleep()
        next_run_at = (datetime.now(timezone.utc)
                       + __import__("datetime").timedelta(seconds=next_sleep)).isoformat()
        try:
            self._audit_log.append(AuditRecord(
                timestamp=datetime.now(timezone.utc),
                actor_id="scheduler",
                action=AuditAction.FINDING_RECORDED,
                outcome="next_run_scheduled",
                detail={"next_run_at": next_run_at},
            ))
        except Exception:
            pass

    async def _probe_target(self, target: str, now: datetime) -> None:
        """Run the full probe flow for a single target."""
        from acdp.agents.red_team_agent import ProbeTask
        from acdp.agents.devsecops_agent import PullRequest, Declination

        # Step 1: Authorization gate
        authz_req = ActionRequest(
            agent_id="scheduler",
            asset=target,
            action="probe",
        )
        decision = self._authz.authorize(authz_req, at=now)

        if not decision.grant:
            # Denied — audit AUTHZ_DENY and skip to next target.
            # AuthorizationService already wrote the AUTHZ_DENY record via guarded_action.
            return

        # Step 2: Plan probe
        task = ProbeTask(
            task_id=str(uuid.uuid4()),
            event_id=str(uuid.uuid4()),
            target_asset=target,
            description=f"Scheduled probe of {target}",
        )
        plan = self._red_team.plan_probe(task)

        # Step 3: Execute probe
        findings = self._red_team.execute_probe(plan, self._scope)

        # Step 4: Audit each finding
        for finding in findings:
            self._audit_log.append(AuditRecord(
                timestamp=datetime.now(timezone.utc),
                actor_id="scheduler",
                action=AuditAction.FINDING_RECORDED,
                outcome="finding",
                target=target,
                detail={
                    "finding_id": finding.finding_id,
                    "plan_id": plan.plan_id,
                    "severity": finding.severity.value,
                },
            ))

            # Step 5: Auto-remediate if configured
            if self._config.auto_remediate:
                try:
                    result = self._devsecops.remediate(finding, self._scope)
                    if isinstance(result, PullRequest):
                        self._audit_log.append(AuditRecord(
                            timestamp=datetime.now(timezone.utc),
                            actor_id="scheduler",
                            action=AuditAction.PR_OPENED,
                            outcome="pr_opened",
                            target=target,
                            detail={
                                "finding_id": finding.finding_id,
                                "pr_id": result.pr_id,
                            },
                        ))
                    elif isinstance(result, Declination):
                        self._audit_log.append(AuditRecord(
                            timestamp=datetime.now(timezone.utc),
                            actor_id="scheduler",
                            action=AuditAction.REMEDIATION_DECLINED,
                            outcome="declined",
                            target=target,
                            detail={
                                "finding_id": finding.finding_id,
                                "reason": result.reason,
                            },
                        ))
                except Exception as exc:
                    self._audit_log.append(AuditRecord(
                        timestamp=datetime.now(timezone.utc),
                        actor_id="scheduler",
                        action=AuditAction.TASK_FAILED,
                        outcome="remediation_error",
                        target=target,
                        detail={
                            "finding_id": finding.finding_id,
                            "error": str(exc),
                        },
                    ))
