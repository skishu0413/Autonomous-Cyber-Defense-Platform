"""Property-based test for Red Team Agent — Property 21.

# Feature: autonomous-cyber-defense-platform, Property 21: Discovered weaknesses map one-to-one to reported findings
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings

from acdp.agents.red_team_agent import RedTeamAgent
from acdp.audit import AuditLog
from acdp.authorization import AuthorizationService
from acdp.models import AuditRecord, Finding

from tests.strategies import probe_plans_with_scopes


# ---------------------------------------------------------------------------
# Minimal in-memory stubs
# ---------------------------------------------------------------------------


class _InMemoryAuditLog:
    """Minimal in-memory audit log for property tests."""

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


class _EmptyRetriever:
    """Retriever stub that always returns an empty result (RAG not under test here)."""

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        category: str | None = None,
    ) -> list[Any]:
        return []


# ---------------------------------------------------------------------------
# Property 21: Discovered weaknesses map one-to-one to reported findings
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(scenario=probe_plans_with_scopes())
def test_property21_weakness_to_finding_one_to_one(
    scenario: dict[str, Any],
) -> None:
    """**Validates: Requirements 9.3**

    For any set of weaknesses discovered during an authorized probe, the Red
    Team Agent SHALL report exactly one finding per weakness to the Orchestrator.

    Asserts:
    - len(findings) == len(plan.weaknesses)  (one finding per weakness, no more, no less)
    - every returned item is a valid Finding instance
    """
    plan = scenario["plan"]
    scope = scenario["scope"]

    audit_log = _InMemoryAuditLog()
    authz = AuthorizationService(scopes=[scope], audit=audit_log)
    retriever = _EmptyRetriever()
    agent = RedTeamAgent(audit_log=audit_log, retriever=retriever, authz=authz)

    findings = agent.execute_probe(plan, scope)

    # One finding per weakness — the bijection property (Req 9.3).
    assert len(findings) == len(plan.weaknesses), (
        f"Expected {len(plan.weaknesses)} finding(s) for weaknesses "
        f"{plan.weaknesses!r}, got {len(findings)}"
    )

    # Every returned item must be a valid Finding instance.
    for finding in findings:
        assert isinstance(finding, Finding), (
            f"Expected a Finding instance, got {type(finding)!r}: {finding!r}"
        )
