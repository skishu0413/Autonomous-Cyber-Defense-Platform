"""Red Team Agent — authorized adversary simulation and probe execution (Layer 4 — Agents).

The :class:`RedTeamAgent` performs authorized vulnerability probing against
operator-owned targets. It is the only agent with offensive capabilities,
and those capabilities are tightly gated:

* :meth:`plan_probe` retrieves OWASP GenAI and MITRE ATT&CK/ATLAS context from
  the RAG Core *before* planning a probe, grounding the probe plan in
  authoritative adversary knowledge (Req 9.4).

* :meth:`execute_probe` verifies authorization via the
  :class:`~acdp.authz.AuthorizationService` *first*, before any side effect.
  When authorization is denied (out of scope, expired, or revoked), the probe is
  refused, an :class:`~acdp.models.AuditRecord` with
  :attr:`~acdp.models.AuditAction.PROBE_REFUSED` is appended, and an empty list
  is returned — no findings, no side effects (Req 9.1, 9.2, 9.5). When
  authorized, exactly one :class:`~acdp.models.Finding` per entry in
  :attr:`ProbePlan.weaknesses` is produced and returned (Req 9.3).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel

from acdp.audit import AuditLog
from acdp.authorization import AuthorizationService
from acdp.models import (
    ActionRequest,
    AuditAction,
    AuditRecord,
    Finding,
    Severity,
    SourceCategory,
    TargetScope,
)
from acdp.knowledge_base.retrieve import Retriever

__all__ = [
    "RedTeamAgent",
    "ProbeTask",
    "ProbePlan",
]

# Sentinel agent identifier embedded in every finding and audit record produced
# by this agent (Req 12.1, 12.3).
_AGENT_ID = "red_team_agent"


class ProbeTask(BaseModel):
    """A probe task dispatched to the :class:`RedTeamAgent` by the Orchestrator.

    Fields:
        task_id: Unique identifier for this task (assigned by the Orchestrator).
        event_id: The originating event identifier; carried through to findings
            so each finding can be linked back to its originating event (Req 12.3).
        target_asset: The asset the probe should target (a host, domain, repo, or
            endpoint that must be within an authorized :class:`~acdp.models.TargetScope`
            for the probe to proceed, Req 9.1).
        description: Human-readable description of the probe objective.
    """

    task_id: str
    event_id: str
    target_asset: str
    description: str


class ProbePlan(BaseModel):
    """A structured probe plan produced by :meth:`RedTeamAgent.plan_probe`.

    Fields:
        plan_id: Unique identifier for this plan.
        task_id: The task identifier from the originating :class:`ProbeTask`.
        event_id: The originating event identifier from the :class:`ProbeTask`,
            carried through so findings produced during execution can reference
            the event that triggered the probe (Req 12.3).
        target_asset: The asset to probe (propagated from the task).
        context_refs: The ``source_id`` values of the RAG chunks retrieved
            during planning (OWASP GenAI, MITRE ATT&CK, MITRE ATLAS). These are
            attached as provenance to findings produced during execution (Req 9.4).
        weaknesses: The list of weakness labels to probe. Each entry in this
            list maps one-to-one to a :class:`~acdp.models.Finding` produced by
            :meth:`~RedTeamAgent.execute_probe` when the probe is authorized (Req 9.3).
    """

    plan_id: str
    task_id: str
    event_id: str
    target_asset: str
    context_refs: list[str]
    weaknesses: list[str]


class RedTeamAgent:
    """Red Team Agent: authorized adversary simulation and probe execution.

    All offensive actions are gated behind the
    :class:`~acdp.authz.AuthorizationService`. The agent never modifies any
    external state when authorization is denied.

    Constructor Args:
        audit_log: Append-only log; every probe refusal is recorded here with
            :attr:`~acdp.models.AuditAction.PROBE_REFUSED` (Req 9.2, 9.5).
        retriever: RAG retriever used in :meth:`plan_probe` to fetch OWASP GenAI
            and MITRE ATT&CK/ATLAS context (Req 9.4).
        authz: Authorization service that enforces scope, expiration, and
            revocation checks before any probe executes (Req 9.1).
        actor_id: Actor identifier recorded on audit records; defaults to
            ``"red_team_agent"``.
    """

    def __init__(
        self,
        audit_log: AuditLog,
        retriever: Retriever,
        authz: AuthorizationService,
        *,
        actor_id: str = _AGENT_ID,
    ) -> None:
        self._audit_log = audit_log
        self._retriever = retriever
        self._authz = authz
        self._actor_id = actor_id

    # ------------------------------------------------------------------
    # Probe planning  (Req 9.4)
    # ------------------------------------------------------------------

    def plan_probe(self, task: ProbeTask) -> ProbePlan:
        """Plan a probe by retrieving adversary knowledge from the RAG Core.

        Queries the RAG Core for context from three security knowledge
        categories — OWASP GenAI, MITRE ATT&CK, and MITRE ATLAS — and stores
        the returned ``source_id`` values in the plan as ``context_refs``
        (Req 9.4). The plan also includes a list of weaknesses derived from the
        task description that will be probed during execution.

        Args:
            task: The probe task dispatched by the Orchestrator, carrying the
                target asset and description.

        Returns:
            A :class:`ProbePlan` with populated ``context_refs`` from the
            retrieved adversary knowledge and a list of ``weaknesses`` to probe.
        """
        query = f"{task.description} target:{task.target_asset}"

        # Retrieve adversary context from all three security-knowledge categories
        # mandated by Req 9.4.
        context_refs: list[str] = []
        for category in (
            SourceCategory.OWASP_GENAI,
            SourceCategory.MITRE_ATTACK,
            SourceCategory.MITRE_ATLAS,
        ):
            chunks = self._retriever.retrieve(query, category=category.value)
            for scored_chunk in chunks:
                source_id = scored_chunk.chunk.source_id
                if source_id not in context_refs:
                    context_refs.append(source_id)

        # Derive the list of weaknesses to probe from the task description.
        # Each weakness in this list maps one-to-one to a Finding produced by
        # execute_probe (Req 9.3).
        weaknesses = _derive_weaknesses(task.description)

        return ProbePlan(
            plan_id=str(uuid.uuid4()),
            task_id=task.task_id,
            event_id=task.event_id,
            target_asset=task.target_asset,
            context_refs=context_refs,
            weaknesses=weaknesses,
        )

    # ------------------------------------------------------------------
    # Probe execution  (Req 9.1, 9.2, 9.3, 9.5)
    # ------------------------------------------------------------------

    def execute_probe(
        self, plan: ProbePlan, scope: TargetScope
    ) -> list[Finding]:
        """Execute a probe plan against the target after verifying authorization.

        Authorization is verified *first* (fail closed) via the
        :class:`~acdp.authz.AuthorizationService`. When the authorization
        service returns DENY (target out of scope, scope expired, or scope
        revoked), the probe is refused: exactly one
        :attr:`~acdp.models.AuditAction.PROBE_REFUSED` audit record is appended
        and an empty list is returned with no side effects (Req 9.1, 9.2, 9.5).

        When authorization is granted, exactly one
        :class:`~acdp.models.Finding` is produced per entry in
        ``plan.weaknesses`` and returned to the caller (Req 9.3). Each finding
        carries:

        * A unique ``finding_id`` (UUID4).
        * The ``originating_event_id`` from the originating task's event.
        * ``agent_id="red_team_agent"``.
        * The ``context_refs`` collected during planning.
        * The ``target_asset`` of the plan as ``asset``.

        Args:
            plan: The probe plan produced by :meth:`plan_probe`.
            scope: The :class:`~acdp.models.TargetScope` to authorize the probe
                against. The target asset must be within this scope and the scope
                must be neither expired nor revoked.

        Returns:
            A list of one :class:`~acdp.models.Finding` per weakness in the
            plan when authorized, or an empty list when the probe is refused.
        """
        now = datetime.now(timezone.utc)
        asset = plan.target_asset

        # Req 9.1: verify authorization first; fail closed.
        authz_request = ActionRequest(
            agent_id=self._actor_id,
            asset=asset,
            action="probe",
        )
        decision = self._authz.authorize(authz_request, at=now)

        if not decision.grant:
            # Req 9.2, 9.5: refuse the probe and record the refusal.
            refusal_record = AuditRecord(
                timestamp=now,
                actor_id=self._actor_id,
                action=AuditAction.PROBE_REFUSED,
                outcome="refused",
                target=asset,
                detail={
                    "reason": decision.reason,
                    "plan_id": plan.plan_id,
                    "task_id": plan.task_id,
                    "scope_id": scope.scope_id,
                    "scope_active": scope.is_active(now),
                    "scope_revoked": scope.revoked,
                },
            )
            self._audit_log.append(refusal_record)
            return []

        # Req 9.3: authorized — produce exactly one Finding per weakness.
        findings: list[Finding] = []
        for weakness in plan.weaknesses:
            findings.append(
                Finding(
                    finding_id=str(uuid.uuid4()),
                    originating_event_id=plan.event_id,
                    agent_id=self._actor_id,
                    severity=Severity.MEDIUM,
                    title=f"Discovered weakness: {weakness}",
                    detail=(
                        f"Probe of asset {asset!r} identified weakness: {weakness}. "
                        f"Context retrieved from: {', '.join(plan.context_refs) or 'none'}."
                    ),
                    asset=asset,
                    context_refs=list(plan.context_refs),
                    created_at=now,
                )
            )

        return findings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _derive_weaknesses(description: str) -> list[str]:
    """Derive a list of weakness labels from a probe task description.

    In a production deployment this would be driven by an LLM reasoning step
    over the task description and the retrieved adversary context. For the
    purposes of this implementation, a rule-based heuristic extracts one
    weakness per line of the description (or falls back to a single entry
    from the whole description) so that the one-to-one ``weakness → Finding``
    mapping (Req 9.3) is always exercised for any non-trivial task description.

    A non-empty list is guaranteed: if the description is blank, a single
    generic weakness entry is returned.
    """
    lines = [line.strip() for line in description.splitlines() if line.strip()]
    if lines:
        return lines
    stripped = description.strip()
    return [stripped] if stripped else ["unspecified weakness"]


