"""Unit tests for the Red Team Agent (Task 16.1).

Covers:
- plan_probe retrieves OWASP GenAI and MITRE ATT&CK/ATLAS context from RAG
  and stores source_ids in the plan's context_refs (Req 9.4)
- execute_probe verifies authorization before acting (Req 9.1)
- execute_probe refuses and audits when out of scope, expired, or revoked (Req 9.2, 9.5)
- execute_probe produces exactly one Finding per weakness when authorized (Req 9.3)
- findings carry correct originating_event_id, agent_id, and context_refs
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Iterator

import pytest

from acdp.agents.red_team_agent import ProbePlan, ProbeTask, RedTeamAgent
from acdp.audit import AuditLog
from acdp.authorization import AuthorizationService
from acdp.models import (
    AuditAction,
    AuditRecord,
    EmbeddedChunk,
    Finding,
    ScoredChunk,
    Severity,
    SourceCategory,
    TargetScope,
)
from acdp.knowledge_base.retrieve import Retriever


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
    """Retriever stub that returns a configurable set of scored chunks.

    Chunks can be pre-seeded per category so plan_probe can be tested in
    isolation without a live vector store.
    """

    def __init__(self, chunks_by_category: dict[str, list[ScoredChunk]] | None = None) -> None:
        self._chunks: dict[str, list[ScoredChunk]] = chunks_by_category or {}

    def retrieve(
        self, query: str, top_k: int | None = None, category: str | None = None
    ) -> list[ScoredChunk]:
        if category is None:
            # Return everything
            result: list[ScoredChunk] = []
            for chunks in self._chunks.values():
                result.extend(chunks)
            return result
        return list(self._chunks.get(category, []))


def _make_scored_chunk(source_id: str, category: SourceCategory) -> ScoredChunk:
    chunk = EmbeddedChunk(
        chunk_id=str(uuid.uuid4()),
        source_id=source_id,
        category=category,
        ingested_at=datetime.now(timezone.utc),
        text="sample text",
        vector=[0.1, 0.2, 0.3],
    )
    return ScoredChunk(chunk=chunk, score=0.9)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _future(hours: int = 1) -> datetime:
    return _now() + timedelta(hours=hours)


def _past(hours: int = 1) -> datetime:
    return _now() - timedelta(hours=hours)


def _active_scope(asset: str = "host-a") -> TargetScope:
    now = _now()
    return TargetScope(
        scope_id="scope-1",
        assets=[asset],
        created_at=now,
        expires_at=_future(2),
    )


def _expired_scope(asset: str = "host-a") -> TargetScope:
    now = _now()
    return TargetScope(
        scope_id="scope-expired",
        assets=[asset],
        created_at=_past(3),
        expires_at=_past(1),
    )


def _revoked_scope(asset: str = "host-a") -> TargetScope:
    now = _now()
    return TargetScope(
        scope_id="scope-revoked",
        assets=[asset],
        created_at=now,
        expires_at=_future(2),
        revoked=True,
    )


def _make_agent(
    scope: TargetScope | None = None,
    chunks_by_category: dict[str, list[ScoredChunk]] | None = None,
    scopes: list[TargetScope] | None = None,
) -> tuple[RedTeamAgent, _InMemoryAuditLog]:
    """Build a RedTeamAgent wired to an in-memory audit log and stub retriever."""
    audit_log = _InMemoryAuditLog()
    if scopes is None:
        scopes = [scope] if scope is not None else []
    authz = AuthorizationService(scopes=scopes, audit=audit_log)
    retriever = _StubRetriever(chunks_by_category)
    agent = RedTeamAgent(audit_log=audit_log, retriever=retriever, authz=authz)
    return agent, audit_log


def _make_task(target_asset: str = "host-a", description: str = "probe desc") -> ProbeTask:
    return ProbeTask(
        task_id=str(uuid.uuid4()),
        event_id=str(uuid.uuid4()),
        target_asset=target_asset,
        description=description,
    )


# ---------------------------------------------------------------------------
# plan_probe tests (Req 9.4)
# ---------------------------------------------------------------------------


class TestPlanProbe:
    def test_plan_probe_returns_probe_plan_with_task_id(self) -> None:
        agent, _ = _make_agent()
        task = _make_task()
        plan = agent.plan_probe(task)

        assert plan.task_id == task.task_id
        assert plan.target_asset == task.target_asset
        assert plan.event_id == task.event_id

    def test_plan_probe_has_unique_plan_id(self) -> None:
        agent, _ = _make_agent()
        task = _make_task()
        plan1 = agent.plan_probe(task)
        plan2 = agent.plan_probe(task)

        assert plan1.plan_id != plan2.plan_id

    def test_plan_probe_retrieves_owasp_genai_context(self) -> None:
        """plan_probe retrieves chunks from OWASP_GENAI category (Req 9.4)."""
        source_id = "owasp-src-1"
        chunks = {
            SourceCategory.OWASP_GENAI.value: [_make_scored_chunk(source_id, SourceCategory.OWASP_GENAI)],
        }
        agent, _ = _make_agent(chunks_by_category=chunks)
        task = _make_task()
        plan = agent.plan_probe(task)

        assert source_id in plan.context_refs

    def test_plan_probe_retrieves_mitre_attack_context(self) -> None:
        """plan_probe retrieves chunks from MITRE_ATTACK category (Req 9.4)."""
        source_id = "mitre-attack-src-1"
        chunks = {
            SourceCategory.MITRE_ATTACK.value: [
                _make_scored_chunk(source_id, SourceCategory.MITRE_ATTACK)
            ],
        }
        agent, _ = _make_agent(chunks_by_category=chunks)
        task = _make_task()
        plan = agent.plan_probe(task)

        assert source_id in plan.context_refs

    def test_plan_probe_retrieves_mitre_atlas_context(self) -> None:
        """plan_probe retrieves chunks from MITRE_ATLAS category (Req 9.4)."""
        source_id = "mitre-atlas-src-1"
        chunks = {
            SourceCategory.MITRE_ATLAS.value: [
                _make_scored_chunk(source_id, SourceCategory.MITRE_ATLAS)
            ],
        }
        agent, _ = _make_agent(chunks_by_category=chunks)
        task = _make_task()
        plan = agent.plan_probe(task)

        assert source_id in plan.context_refs

    def test_plan_probe_deduplicates_context_refs(self) -> None:
        """Same source_id from multiple chunks is included only once."""
        source_id = "shared-source"
        chunks = {
            SourceCategory.OWASP_GENAI.value: [
                _make_scored_chunk(source_id, SourceCategory.OWASP_GENAI),
                _make_scored_chunk(source_id, SourceCategory.OWASP_GENAI),
            ],
        }
        agent, _ = _make_agent(chunks_by_category=chunks)
        task = _make_task()
        plan = agent.plan_probe(task)

        assert plan.context_refs.count(source_id) == 1

    def test_plan_probe_collects_all_three_categories(self) -> None:
        """plan_probe fetches context from all three mandatory categories."""
        chunks = {
            SourceCategory.OWASP_GENAI.value: [
                _make_scored_chunk("owasp-1", SourceCategory.OWASP_GENAI)
            ],
            SourceCategory.MITRE_ATTACK.value: [
                _make_scored_chunk("attack-1", SourceCategory.MITRE_ATTACK)
            ],
            SourceCategory.MITRE_ATLAS.value: [
                _make_scored_chunk("atlas-1", SourceCategory.MITRE_ATLAS)
            ],
        }
        agent, _ = _make_agent(chunks_by_category=chunks)
        task = _make_task()
        plan = agent.plan_probe(task)

        assert "owasp-1" in plan.context_refs
        assert "attack-1" in plan.context_refs
        assert "atlas-1" in plan.context_refs

    def test_plan_probe_context_refs_empty_when_no_rag_results(self) -> None:
        """context_refs is empty when the RAG Core returns nothing (no error raised)."""
        agent, _ = _make_agent(chunks_by_category={})
        task = _make_task()
        plan = agent.plan_probe(task)

        assert plan.context_refs == []

    def test_plan_probe_weaknesses_non_empty_for_any_description(self) -> None:
        """plan_probe always produces at least one weakness."""
        for desc in ("", "  ", "SQL injection", "XSS\nCSRF\nAuth bypass"):
            agent, _ = _make_agent()
            task = _make_task(description=desc)
            plan = agent.plan_probe(task)
            assert len(plan.weaknesses) >= 1

    def test_plan_probe_multiline_description_yields_one_weakness_per_line(self) -> None:
        """Multi-line descriptions produce one weakness per non-blank line."""
        desc = "SQL injection\nXSS\nCSRF"
        agent, _ = _make_agent()
        task = _make_task(description=desc)
        plan = agent.plan_probe(task)

        assert len(plan.weaknesses) == 3
        assert plan.weaknesses == ["SQL injection", "XSS", "CSRF"]


# ---------------------------------------------------------------------------
# execute_probe — authorization refusal tests (Req 9.1, 9.2, 9.5)
# ---------------------------------------------------------------------------


class TestExecuteProbeRefusal:
    def _make_plan(
        self,
        target_asset: str = "host-a",
        weaknesses: list[str] | None = None,
    ) -> ProbePlan:
        return ProbePlan(
            plan_id=str(uuid.uuid4()),
            task_id=str(uuid.uuid4()),
            event_id=str(uuid.uuid4()),
            target_asset=target_asset,
            context_refs=["owasp-1"],
            weaknesses=weaknesses if weaknesses is not None else ["SQLi"],
        )

    def test_execute_probe_refused_when_no_scope_defined(self) -> None:
        """No scope defined → probe refused, empty findings returned."""
        agent, audit_log = _make_agent(scopes=[])
        plan = self._make_plan()
        scope = _active_scope()

        findings = agent.execute_probe(plan, scope)

        assert findings == []

    def test_execute_probe_refused_audits_probe_refused(self) -> None:
        """When refused, exactly one PROBE_REFUSED audit record is appended."""
        audit_log = _InMemoryAuditLog()
        authz = AuthorizationService(scopes=[], audit=audit_log)
        retriever = _StubRetriever()
        agent = RedTeamAgent(audit_log=audit_log, retriever=retriever, authz=authz)
        plan = self._make_plan()
        scope = _active_scope()

        agent.execute_probe(plan, scope)

        # The audit log gets: one AUTHZ_DENY from AuthorizationService, then one
        # PROBE_REFUSED from the agent itself.
        probe_refused_records = [
            r for r in audit_log.read_all() if r.action == AuditAction.PROBE_REFUSED
        ]
        assert len(probe_refused_records) == 1

    def test_execute_probe_refused_when_asset_out_of_scope(self) -> None:
        """Asset not in scope → refused, no findings."""
        scope = _active_scope(asset="host-b")  # plan targets host-a
        agent, _ = _make_agent(scopes=[scope])
        plan = self._make_plan(target_asset="host-a")

        findings = agent.execute_probe(plan, scope)

        assert findings == []

    def test_execute_probe_refused_when_scope_expired(self) -> None:
        """Expired scope → refused (Req 9.5), no findings."""
        scope = _expired_scope(asset="host-a")
        agent, _ = _make_agent(scopes=[scope])
        plan = self._make_plan(target_asset="host-a")

        findings = agent.execute_probe(plan, scope)

        assert findings == []

    def test_execute_probe_refused_when_scope_revoked(self) -> None:
        """Revoked scope → refused (Req 9.5), no findings."""
        scope = _revoked_scope(asset="host-a")
        agent, _ = _make_agent(scopes=[scope])
        plan = self._make_plan(target_asset="host-a")

        findings = agent.execute_probe(plan, scope)

        assert findings == []

    def test_execute_probe_refusal_audit_record_targets_asset(self) -> None:
        """PROBE_REFUSED audit record targets the probe asset."""
        audit_log = _InMemoryAuditLog()
        authz = AuthorizationService(scopes=[], audit=audit_log)
        retriever = _StubRetriever()
        agent = RedTeamAgent(audit_log=audit_log, retriever=retriever, authz=authz)
        plan = self._make_plan(target_asset="sensitive-host")
        scope = _active_scope(asset="other-host")

        agent.execute_probe(plan, scope)

        refused = [r for r in audit_log.read_all() if r.action == AuditAction.PROBE_REFUSED]
        assert len(refused) == 1
        assert refused[0].target == "sensitive-host"

    def test_execute_probe_refusal_audit_actor_id_is_agent(self) -> None:
        """PROBE_REFUSED audit record actor_id is the agent's actor_id."""
        audit_log = _InMemoryAuditLog()
        authz = AuthorizationService(scopes=[], audit=audit_log)
        retriever = _StubRetriever()
        agent = RedTeamAgent(
            audit_log=audit_log, retriever=retriever, authz=authz, actor_id="custom_red_agent"
        )
        plan = self._make_plan()
        scope = _active_scope()

        agent.execute_probe(plan, scope)

        refused = [r for r in audit_log.read_all() if r.action == AuditAction.PROBE_REFUSED]
        assert refused[0].actor_id == "custom_red_agent"


# ---------------------------------------------------------------------------
# execute_probe — authorized execution tests (Req 9.1, 9.3)
# ---------------------------------------------------------------------------


class TestExecuteProbeAuthorized:
    def _make_plan(
        self,
        target_asset: str = "host-a",
        weaknesses: list[str] | None = None,
        context_refs: list[str] | None = None,
        event_id: str | None = None,
        task_id: str | None = None,
    ) -> ProbePlan:
        return ProbePlan(
            plan_id=str(uuid.uuid4()),
            task_id=task_id if task_id is not None else str(uuid.uuid4()),
            event_id=event_id if event_id is not None else str(uuid.uuid4()),
            target_asset=target_asset,
            context_refs=context_refs if context_refs is not None else ["owasp-1"],
            weaknesses=weaknesses if weaknesses is not None else ["SQLi"],
        )

    def test_execute_probe_returns_one_finding_per_weakness(self) -> None:
        """Exactly one Finding per weakness in the plan (Req 9.3)."""
        scope = _active_scope(asset="host-a")
        agent, _ = _make_agent(scopes=[scope])
        plan = self._make_plan(weaknesses=["SQLi", "XSS", "CSRF"])

        findings = agent.execute_probe(plan, scope)

        assert len(findings) == 3

    def test_execute_probe_empty_weaknesses_returns_empty_findings(self) -> None:
        """A plan with no weaknesses yields no findings."""
        scope = _active_scope(asset="host-a")
        agent, _ = _make_agent(scopes=[scope])
        plan = self._make_plan(weaknesses=[])

        findings = agent.execute_probe(plan, scope)

        assert findings == []

    def test_execute_probe_findings_have_unique_ids(self) -> None:
        """Every finding in the result has a unique finding_id."""
        scope = _active_scope(asset="host-a")
        agent, _ = _make_agent(scopes=[scope])
        plan = self._make_plan(weaknesses=["SQLi", "XSS", "CSRF", "Auth bypass"])

        findings = agent.execute_probe(plan, scope)

        ids = [f.finding_id for f in findings]
        assert len(ids) == len(set(ids)), "finding_ids must be unique"

    def test_execute_probe_findings_carry_originating_event_id(self) -> None:
        """Each finding's originating_event_id matches the plan's event_id (Req 12.3)."""
        scope = _active_scope(asset="host-a")
        agent, _ = _make_agent(scopes=[scope])
        event_id = "evt-xyz-123"
        plan = self._make_plan(weaknesses=["SQLi", "XSS"], event_id=event_id)

        findings = agent.execute_probe(plan, scope)

        for f in findings:
            assert f.originating_event_id == event_id

    def test_execute_probe_findings_carry_agent_id(self) -> None:
        """Each finding's agent_id is 'red_team_agent'."""
        scope = _active_scope(asset="host-a")
        agent, _ = _make_agent(scopes=[scope])
        plan = self._make_plan(weaknesses=["SQLi"])

        findings = agent.execute_probe(plan, scope)

        for f in findings:
            assert f.agent_id == "red_team_agent"

    def test_execute_probe_findings_carry_context_refs_from_plan(self) -> None:
        """Each finding's context_refs matches the plan's context_refs."""
        scope = _active_scope(asset="host-a")
        agent, _ = _make_agent(scopes=[scope])
        ctx = ["owasp-1", "mitre-attack-2", "atlas-3"]
        plan = self._make_plan(weaknesses=["SQLi", "XSS"], context_refs=ctx)

        findings = agent.execute_probe(plan, scope)

        for f in findings:
            assert f.context_refs == ctx

    def test_execute_probe_findings_have_asset_set(self) -> None:
        """Each finding's asset matches the plan's target_asset."""
        scope = _active_scope(asset="web-server")
        agent, _ = _make_agent(scopes=[scope])
        plan = self._make_plan(target_asset="web-server", weaknesses=["SQLi"])

        findings = agent.execute_probe(plan, scope)

        for f in findings:
            assert f.asset == "web-server"

    def test_execute_probe_does_not_append_probe_refused_when_authorized(self) -> None:
        """No PROBE_REFUSED record is written when the probe is authorized."""
        audit_log = _InMemoryAuditLog()
        scope = _active_scope(asset="host-a")
        authz = AuthorizationService(scopes=[scope], audit=audit_log)
        retriever = _StubRetriever()
        agent = RedTeamAgent(audit_log=audit_log, retriever=retriever, authz=authz)
        plan = self._make_plan(target_asset="host-a", weaknesses=["SQLi"])

        agent.execute_probe(plan, scope)

        refused_records = [
            r for r in audit_log.read_all() if r.action == AuditAction.PROBE_REFUSED
        ]
        assert refused_records == []

    def test_execute_probe_findings_contain_weakness_in_title(self) -> None:
        """Each finding's title contains the weakness string."""
        scope = _active_scope(asset="host-a")
        agent, _ = _make_agent(scopes=[scope])
        weaknesses = ["SQL injection", "XSS via cookie"]
        plan = self._make_plan(weaknesses=weaknesses)

        findings = agent.execute_probe(plan, scope)

        for finding, weakness in zip(findings, weaknesses):
            assert weakness in finding.title


# ---------------------------------------------------------------------------
# Integration: plan_probe + execute_probe (Req 9.1, 9.3, 9.4)
# ---------------------------------------------------------------------------


class TestPlanThenExecute:
    def test_plan_then_execute_authorized(self) -> None:
        """Full plan → execute pipeline returns one finding per weakness."""
        rag_chunks = {
            SourceCategory.OWASP_GENAI.value: [
                _make_scored_chunk("owasp-1", SourceCategory.OWASP_GENAI)
            ],
            SourceCategory.MITRE_ATTACK.value: [
                _make_scored_chunk("attack-1", SourceCategory.MITRE_ATTACK)
            ],
            SourceCategory.MITRE_ATLAS.value: [
                _make_scored_chunk("atlas-1", SourceCategory.MITRE_ATLAS)
            ],
        }
        scope = _active_scope(asset="prod-host")
        agent, audit_log = _make_agent(
            scopes=[scope], chunks_by_category=rag_chunks
        )
        task = ProbeTask(
            task_id="task-1",
            event_id="event-1",
            target_asset="prod-host",
            description="SQL injection\nXSS\nCSRF",
        )

        plan = agent.plan_probe(task)
        findings = agent.execute_probe(plan, scope)

        # 3 weaknesses → 3 findings
        assert len(findings) == 3
        # All findings link to the originating event
        for f in findings:
            assert f.originating_event_id == "event-1"
        # All findings carry the RAG context refs
        for f in findings:
            assert "owasp-1" in f.context_refs
            assert "attack-1" in f.context_refs
            assert "atlas-1" in f.context_refs

    def test_plan_then_execute_refused_when_out_of_scope(self) -> None:
        """Full plan → execute returns no findings when asset not in scope."""
        scope = _active_scope(asset="other-host")
        agent, _ = _make_agent(scopes=[scope])
        task = _make_task(target_asset="prod-host")
        plan = agent.plan_probe(task)

        findings = agent.execute_probe(plan, scope)

        assert findings == []
