"""Property-based test for Red Team Agent — Property 22.

# Feature: autonomous-cyber-defense-platform, Property 22: Probe planning retrieves adversary knowledge
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from acdp.agents.red_team_agent import ProbeTask, RedTeamAgent
from acdp.audit import AuditLog
from acdp.authorization import AuthorizationService
from acdp.models import AuditRecord, ScoredChunk, SourceCategory


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


class _TrackingRetriever:
    """Retriever stub that records every (query, category) pair that was retrieved.

    This fake captures the ``category`` argument passed to each ``retrieve``
    call so Property 22 can assert that plan_probe queries all three mandatory
    adversary-knowledge categories: OWASP_GENAI, MITRE_ATTACK, and MITRE_ATLAS.

    The retriever always returns an empty list — the RAG *content* is not under
    test here, only the *categories* that are queried.
    """

    def __init__(self) -> None:
        # Ordered list of (query, category) tuples recorded across all calls.
        self.calls: list[tuple[str, str | None]] = []

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        category: str | None = None,
    ) -> list[ScoredChunk]:
        self.calls.append((query, category))
        return []

    @property
    def queried_categories(self) -> set[str | None]:
        """Return the set of distinct category values that were queried."""
        return {cat for _, cat in self.calls}


# ---------------------------------------------------------------------------
# Hypothesis strategy: ProbeTask
# ---------------------------------------------------------------------------

# Non-empty printable identifier for task/event ids.
_identifier = st.text(
    alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x7E),
    min_size=1,
    max_size=24,
)

# Printable text for asset names and descriptions. Kept non-empty for the
# asset so it is a meaningful probe target.
_asset_text = st.text(
    alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x7E),
    min_size=1,
    max_size=40,
)

_description_text = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    min_size=0,
    max_size=200,
)


@st.composite
def probe_tasks(draw: st.DrawFn) -> ProbeTask:
    """Generate arbitrary :class:`ProbeTask` instances for Property 22.

    The content of the task is not significant for this property — what matters
    is that *any* probe task causes plan_probe to query the three mandatory
    adversary-knowledge categories from the RAG Core (Req 9.4).
    """
    return ProbeTask(
        task_id=draw(_identifier),
        event_id=draw(_identifier),
        target_asset=draw(_asset_text),
        description=draw(_description_text),
    )


# ---------------------------------------------------------------------------
# Property 22: Probe planning retrieves adversary knowledge
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(task=probe_tasks())
def test_property22_probe_planning_retrieves_adversary_knowledge(
    task: ProbeTask,
) -> None:
    """**Validates: Requirements 9.4**

    For any probe task, RedTeamAgent.plan_probe() SHALL retrieve context from
    the RAG Core using the OWASP GenAI and MITRE ATT&CK/ATLAS source categories
    (Req 9.4).

    Asserts:
    - retrieve() was called at least once with category=SourceCategory.OWASP_GENAI.value
    - retrieve() was called at least once with category=SourceCategory.MITRE_ATTACK.value
    - retrieve() was called at least once with category=SourceCategory.MITRE_ATLAS.value
    """
    audit_log = _InMemoryAuditLog()
    # No scopes needed — plan_probe does not check authorization.
    authz = AuthorizationService(scopes=[], audit=audit_log)
    tracking_retriever = _TrackingRetriever()

    agent = RedTeamAgent(
        audit_log=audit_log,
        retriever=tracking_retriever,
        authz=authz,
    )

    # Call plan_probe with the generated task.
    plan = agent.plan_probe(task)

    queried = tracking_retriever.queried_categories

    # OWASP GenAI must have been queried (Req 9.4).
    assert SourceCategory.OWASP_GENAI.value in queried, (
        f"plan_probe did not query OWASP_GENAI category. "
        f"Queried categories: {queried!r}"
    )

    # MITRE ATT&CK must have been queried (Req 9.4).
    assert SourceCategory.MITRE_ATTACK.value in queried, (
        f"plan_probe did not query MITRE_ATTACK category. "
        f"Queried categories: {queried!r}"
    )

    # MITRE ATLAS must have been queried (Req 9.4).
    assert SourceCategory.MITRE_ATLAS.value in queried, (
        f"plan_probe did not query MITRE_ATLAS category. "
        f"Queried categories: {queried!r}"
    )

    # The returned plan must be consistent with the task it was planned for.
    assert plan.task_id == task.task_id
    assert plan.event_id == task.event_id
    assert plan.target_asset == task.target_asset
