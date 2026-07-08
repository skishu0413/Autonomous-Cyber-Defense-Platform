"""Guardrail Agent — inline prompt/response firewall (Layer 4 — Agents).

The :class:`GuardrailAgent` sits inline between users and the backend LLM. This
module implements the **inbound** half of that firewall, :meth:`GuardrailAgent.
screen_inbound`, which decides whether a user prompt is safe to forward or must
be blocked as a prompt-injection / jailbreak attempt (Req 5.1-5.5):

    prompt -> similarity-match vs Blocklist + retrieved attack patterns
           -> block (>= threshold) with reason + matched pattern id
           -> else forward byte-for-byte unchanged

Screening evaluates the prompt against two sources of known-bad patterns:

* the **Blocklist** — a dynamic, vectorized collection of known malicious
  payloads, matched pattern-by-pattern (Req 5.1); and
* **retrieved attack patterns** — known-attack knowledge (OWASP GenAI, MITRE
  ATT&CK/ATLAS) pulled from the RAG Core by semantic similarity to the prompt
  (Req 5.1).

The prompt is **blocked if and only if** the maximum match similarity across
both sources is at or above the configured threshold (Req 5.2); otherwise it is
forwarded unchanged (Req 5.3). Every decision — block or forward — is recorded
in the audit log with the matched pattern identifier (Req 5.4) via the
*audit-before-act* helper :func:`~acdp.audit.guarded_action`, so the decision is
only returned once its audit record is durably committed (Req 12.4).

When the Blocklist is empty or unavailable, screening cannot be performed and
the agent falls back to the operator-configured default action — ``allow``
(forward) or ``block`` — and audits that too (Req 5.5).

The sibling outbound method ``scrub_outbound`` (PII masking / secret redaction,
Req 6.x) is implemented separately; :class:`GuardrailAgent` is structured so the
two halves coexist on the same agent.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Iterable, Literal, Protocol, Sequence, runtime_checkable

from enum import Enum

from pydantic import BaseModel, Field

from acdp.audit import AuditLog, guarded_action
from acdp.llm_gateway import LLMGateway
from acdp.models import (
    AuditAction,
    AuditRecord,
    PlatformConfig,
    SourceCategory,
)
from acdp.knowledge_base.retrieve import Retriever

__all__ = [
    "BlocklistPattern",
    "BlocklistMatch",
    "Blocklist",
    "EmbeddingBlocklist",
    "InboundDecision",
    "MaskingCategory",
    "MaskedItemSummary",
    "OutboundResult",
    "GuardrailAgent",
    "DEFAULT_ATTACK_PATTERN_CATEGORIES",
    "DEFAULT_MASKING_TOKENS",
]


# Knowledge categories treated as "known-attack patterns" when screening a
# prompt against the RAG Core (Req 5.1). These are the adversary-knowledge
# categories most relevant to prompt-injection / jailbreak detection.
DEFAULT_ATTACK_PATTERN_CATEGORIES: tuple[SourceCategory, ...] = (
    SourceCategory.OWASP_GENAI,
    SourceCategory.MITRE_ATLAS,
    SourceCategory.MITRE_ATTACK,
)


class BlocklistPattern(BaseModel):
    """A single known-bad pattern in the vectorized Blocklist.

    ``vector`` is the precomputed embedding of ``text`` (the Blocklist is a
    *vectorized* collection), so matching a prompt only requires embedding the
    prompt once and comparing against each pattern vector. ``category`` is an
    optional label such as ``"prompt_injection"`` or ``"jailbreak"``.
    """

    pattern_id: str
    text: str
    vector: list[float]
    category: str | None = None


class BlocklistMatch(BaseModel):
    """The similarity of a prompt to one Blocklist pattern (Req 5.1)."""

    pattern_id: str
    similarity: float


@runtime_checkable
class Blocklist(Protocol):
    """A vectorized collection of known-bad patterns the prompt is matched against.

    The protocol is deliberately narrow so the agent can be exercised with a
    stub that returns injected similarity scores, while the default
    :class:`EmbeddingBlocklist` computes real cosine similarities. ``match``
    scores the prompt against *every* pattern (Req 5.1); ``is_empty`` /
    ``is_available`` let the agent detect the empty-or-unavailable condition and
    fall back to the configured default action (Req 5.5).
    """

    def is_available(self) -> bool:
        """Return ``False`` when the Blocklist cannot be consulted (Req 5.5)."""
        ...

    def is_empty(self) -> bool:
        """Return ``True`` when the Blocklist holds no patterns (Req 5.5)."""
        ...

    def match(self, prompt: str) -> list[BlocklistMatch]:
        """Return the similarity of ``prompt`` to every Blocklist pattern (Req 5.1)."""
        ...


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors; higher means closer.

    Returns ``0.0`` when either vector has zero magnitude so ranking stays
    well-defined for degenerate inputs.
    """
    if len(a) != len(b):
        raise ValueError(
            f"vector dimension mismatch: prompt has {len(a)}, pattern has {len(b)}"
        )

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class EmbeddingBlocklist:
    """The default :class:`Blocklist`: embed the prompt, cosine-match each pattern.

    Patterns are supplied with precomputed vectors. On :meth:`match` the prompt
    is embedded once through the :class:`~acdp.llm.LLMGateway` (using the same
    embedding model as the patterns) and compared against every pattern vector
    (Req 5.1). An empty pattern set reports :meth:`is_empty` so the agent can
    apply the configured default action (Req 5.5).
    """

    def __init__(
        self,
        gateway: LLMGateway,
        patterns: Iterable[BlocklistPattern],
        *,
        embedding_model: str | None = None,
    ) -> None:
        self._gateway = gateway
        self._patterns: list[BlocklistPattern] = list(patterns)
        self._embedding_model = embedding_model

    def is_available(self) -> bool:
        """The in-process blocklist is always reachable."""
        return True

    def is_empty(self) -> bool:
        """Return ``True`` when no patterns are loaded (Req 5.5)."""
        return not self._patterns

    def match(self, prompt: str) -> list[BlocklistMatch]:
        """Cosine-match ``prompt`` against every loaded pattern (Req 5.1)."""
        if not self._patterns:
            return []
        vector = self._gateway.embed(prompt, model=self._embedding_model)
        return [
            BlocklistMatch(
                pattern_id=pattern.pattern_id,
                similarity=_cosine_similarity(vector, pattern.vector),
            )
            for pattern in self._patterns
        ]


class InboundDecision(BaseModel):
    """The outcome of screening one inbound prompt (Req 5.2, 5.3, 5.4).

    ``action`` is ``"block"`` when the prompt matched a known-bad pattern at or
    above the threshold (or the default action is ``block``), otherwise
    ``"forward"``. On a forward, ``prompt`` is the original prompt returned
    byte-for-byte unchanged (Req 5.3). ``matched_pattern_id`` /
    ``matched_similarity`` name the pattern responsible for a block (Req 5.4);
    ``reason`` carries a human-readable rejection reason on a block (Req 5.2).
    ``default_action_applied`` is ``True`` when the decision came from the
    empty/unavailable-blocklist fallback rather than similarity matching
    (Req 5.5).
    """

    action: Literal["block", "forward"]
    prompt: str
    reason: str | None = None
    matched_pattern_id: str | None = None
    matched_similarity: float | None = None
    default_action_applied: bool = False


class MaskingCategory(str, Enum):
    """The category of a sensitive value detected in an outbound response (Req 6.x).

    PII categories (Req 6.2) — ``EMAIL``, ``SSN``, ``CREDIT_CARD``,
    ``FINANCIAL_ACCOUNT`` — are *masked*; the secret category (Req 6.3) —
    ``API_KEY`` — is *redacted*. The category label is safe to record in the
    audit log because it names only the *kind* of value found, never the value
    itself (Req 6.4).
    """

    EMAIL = "email"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    FINANCIAL_ACCOUNT = "financial_account"
    API_KEY = "api_key"


# The token substituted for each detected category. PII (Req 6.2) is masked and
# secrets (Req 6.3) are redacted; both replace the raw value so it never reaches
# the user. Tokens are fixed strings that contain none of the original value.
DEFAULT_MASKING_TOKENS: dict[MaskingCategory, str] = {
    MaskingCategory.EMAIL: "[MASKED_EMAIL]",
    MaskingCategory.SSN: "[MASKED_SSN]",
    MaskingCategory.CREDIT_CARD: "[MASKED_CREDIT_CARD]",
    MaskingCategory.FINANCIAL_ACCOUNT: "[MASKED_FINANCIAL_ACCOUNT]",
    MaskingCategory.API_KEY: "[REDACTED_API_KEY]",
}


class MaskedItemSummary(BaseModel):
    """A count of masked/redacted items in one category — never the raw values.

    Only the :class:`MaskingCategory` and the number of occurrences are
    retained, so a summary is safe to embed in an audit record without leaking
    any sensitive value (Req 6.4).
    """

    category: MaskingCategory
    count: int


class OutboundResult(BaseModel):
    """The outcome of scrubbing one outbound response (Req 6.1-6.5).

    ``response`` is the text returned to the user: the original string when
    nothing sensitive was found (Req 6.5), otherwise the text with every
    detected PII value masked (Req 6.2) and every secret redacted (Req 6.3).
    ``masked`` is ``True`` when at least one value was replaced. ``summaries``
    lists the per-category counts of replaced items — counts and categories
    only, never raw values (Req 6.4).
    """

    response: str
    masked: bool = False
    summaries: list[MaskedItemSummary] = Field(default_factory=list)

    @property
    def total_masked(self) -> int:
        """Total number of values masked or redacted across all categories."""
        return sum(summary.count for summary in self.summaries)


# Ordered detection patterns. Order matters: more specific / higher-entropy
# patterns (API keys, credit cards) run before broader numeric patterns
# (financial accounts) so a value is attributed to its most specific category
# and not double-counted. Each pattern captures the full sensitive span so it
# can be replaced wholesale with the category's masking token.
_DETECTORS: tuple[tuple[MaskingCategory, re.Pattern[str]], ...] = (
    # Secret API keys (Req 6.3): common provider-style prefixes followed by a
    # long high-entropy token, plus generic "api_key = <token>" assignments.
    (
        MaskingCategory.API_KEY,
        re.compile(
            r"\b(?:sk|pk|rk|api|key|token)[-_](?:live|test|prod)?[-_]?[A-Za-z0-9]{16,}\b"
            r"|\bAKIA[0-9A-Z]{16}\b"
            r"|\bAIza[0-9A-Za-z_\-]{35}\b"
            r"|\bghp_[A-Za-z0-9]{36}\b"
            r"|\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
        ),
    ),
    # Email addresses (Req 6.2).
    (
        MaskingCategory.EMAIL,
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ),
    # US Social Security numbers (Req 6.2): NNN-NN-NNNN.
    (
        MaskingCategory.SSN,
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    # Payment card numbers (Req 6.2): 13-16 digits, optionally in groups of four
    # separated by spaces or hyphens.
    (
        MaskingCategory.CREDIT_CARD,
        re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    ),
    # Other financial account numbers (Req 6.2): a run of 8-17 digits not
    # already claimed by the SSN or card patterns above.
    (
        MaskingCategory.FINANCIAL_ACCOUNT,
        re.compile(r"\b\d{8,17}\b"),
    ),
)


class GuardrailAgent:
    """Inline prompt/response firewall; this half screens inbound prompts (Req 5.x).

    ``screen_inbound`` similarity-matches a prompt against the Blocklist and
    retrieved attack patterns and returns an :class:`InboundDecision`. Every
    decision is recorded in the injected :class:`~acdp.audit.AuditLog` as a
    ``GUARDRAIL_DECISION`` before it is returned (audit-before-act, Req 5.4,
    12.4).
    """

    def __init__(
        self,
        config: PlatformConfig,
        audit_log: AuditLog,
        blocklist: Blocklist,
        retriever: Retriever | None = None,
        *,
        actor_id: str = "guardrail",
        attack_pattern_categories: Sequence[SourceCategory]
        | None = DEFAULT_ATTACK_PATTERN_CATEGORIES,
    ) -> None:
        """Create a Guardrail Agent.

        Args:
            config: Platform configuration supplying the similarity threshold
                (Req 5.2), the default action (Req 5.5), and ``top_k`` for
                attack-pattern retrieval.
            audit_log: Append-only log every decision is recorded to (Req 5.4).
            blocklist: The vectorized Blocklist the prompt is matched against
                (Req 5.1); its empty/unavailable state triggers the default
                action (Req 5.5).
            retriever: Optional RAG retriever used to pull known-attack patterns
                for additional similarity matching (Req 5.1). When ``None``,
                only the Blocklist is consulted.
            actor_id: Actor identifier recorded on audit records.
            attack_pattern_categories: Source categories queried for known-attack
                patterns. ``None`` or empty disables category filtering / attack
                retrieval respectively.
        """
        self._config = config
        self._audit_log = audit_log
        self._blocklist = blocklist
        self._retriever = retriever
        self._actor_id = actor_id
        self._attack_categories: tuple[SourceCategory, ...] = (
            tuple(attack_pattern_categories)
            if attack_pattern_categories is not None
            else ()
        )

    def screen_inbound(self, prompt: str) -> InboundDecision:
        """Screen an inbound ``prompt``; block if it matches, else forward unchanged.

        Evaluates ``prompt`` against every Blocklist pattern and the retrieved
        known-attack patterns (Req 5.1) and blocks it — returning a rejection
        reason and the matched pattern identifier — if and only if the maximum
        match similarity is at or above the configured threshold (Req 5.2).
        Otherwise the prompt is forwarded byte-for-byte unchanged (Req 5.3).

        When the Blocklist is empty or unavailable, the operator-configured
        default action (``allow`` / ``block``) is applied instead (Req 5.5).

        Every decision is recorded in the audit log before it is returned
        (Req 5.4). Returns the :class:`InboundDecision`.
        """
        # Req 5.5: an empty or unavailable Blocklist means we cannot perform
        # similarity screening, so fall back to the configured default action.
        if not self._blocklist_usable():
            return self._apply_default_action(
                prompt, reason="blocklist empty or unavailable"
            )

        # Req 5.1: collect (pattern_id, similarity) matches from every Blocklist
        # pattern and every retrieved known-attack pattern.
        try:
            matches = self._collect_matches(prompt)
        except Exception:
            # A Blocklist that fails mid-match is treated as unavailable so we
            # still fail safe to the configured default action (Req 5.5).
            return self._apply_default_action(
                prompt, reason="blocklist unavailable during matching"
            )

        best = max(matches, key=lambda m: m[1], default=None)

        # Req 5.2: block iff the strongest match is at or above the threshold.
        if best is not None and best[1] >= self._config.similarity_threshold:
            matched_pattern_id, matched_similarity = best
            decision = InboundDecision(
                action="block",
                prompt=prompt,
                reason=(
                    "prompt matched known-attack pattern "
                    f"{matched_pattern_id!r} at similarity "
                    f"{matched_similarity:.4f} >= threshold "
                    f"{self._config.similarity_threshold}"
                ),
                matched_pattern_id=matched_pattern_id,
                matched_similarity=matched_similarity,
            )
            return self._finalize(decision)

        # Req 5.3: no match at/above threshold -> forward the prompt unchanged.
        decision = InboundDecision(action="forward", prompt=prompt)
        return self._finalize(decision)

    # -- internals -----------------------------------------------------------

    def _blocklist_usable(self) -> bool:
        """Return ``True`` when the Blocklist is available and non-empty (Req 5.5)."""
        try:
            if not self._blocklist.is_available():
                return False
            return not self._blocklist.is_empty()
        except Exception:
            # Any failure inspecting the Blocklist is treated as unavailable so
            # the agent fails safe to the default action.
            return False

    def _collect_matches(self, prompt: str) -> list[tuple[str, float]]:
        """Return (pattern_id, similarity) pairs from the Blocklist and RAG (Req 5.1)."""
        matches: list[tuple[str, float]] = [
            (m.pattern_id, m.similarity) for m in self._blocklist.match(prompt)
        ]

        # Retrieve known-attack patterns from the RAG Core; each scored chunk's
        # source identifier is the matched pattern id, its score the similarity.
        if self._retriever is not None:
            matches.extend(self._retrieve_attack_pattern_matches(prompt))

        return matches

    def _retrieve_attack_pattern_matches(self, prompt: str) -> list[tuple[str, float]]:
        """Retrieve known-attack patterns and return them as (source_id, score) pairs."""
        assert self._retriever is not None  # guarded by caller
        top_k = self._config.top_k

        results: list[tuple[str, float]] = []
        if self._attack_categories:
            for category in self._attack_categories:
                for scored in self._retriever.retrieve(
                    prompt, top_k=top_k, category=category
                ):
                    results.append((scored.chunk.source_id, scored.score))
        else:
            # No category filter configured: match against all known-attack
            # patterns regardless of category.
            for scored in self._retriever.retrieve(prompt, top_k=top_k):
                results.append((scored.chunk.source_id, scored.score))
        return results

    def _apply_default_action(self, prompt: str, *, reason: str) -> InboundDecision:
        """Apply the configured default action for an empty/unavailable Blocklist (Req 5.5)."""
        if self._config.guardrail_default_action == "block":
            decision = InboundDecision(
                action="block",
                prompt=prompt,
                reason=f"default action 'block' applied: {reason}",
                default_action_applied=True,
            )
        else:  # "allow" -> forward the prompt unchanged
            decision = InboundDecision(
                action="forward",
                prompt=prompt,
                reason=f"default action 'allow' applied: {reason}",
                default_action_applied=True,
            )
        return self._finalize(decision)

    def _finalize(self, decision: InboundDecision) -> InboundDecision:
        """Audit the decision (Req 5.4), then return it (audit-before-act, Req 12.4)."""
        record = AuditRecord(
            timestamp=datetime.now(timezone.utc),
            actor_id=self._actor_id,
            action=AuditAction.GUARDRAIL_DECISION,
            outcome=decision.action,
            target=decision.matched_pattern_id,
            detail={
                "decision": decision.action,
                "matched_pattern_id": decision.matched_pattern_id,
                "matched_similarity": decision.matched_similarity,
                "threshold": self._config.similarity_threshold,
                "default_action_applied": decision.default_action_applied,
                "reason": decision.reason,
            },
        )
        # Audit-before-act: the decision is only "committed" once its audit
        # record is durably appended; a failed append raises AuditWriteError.
        return guarded_action(self._audit_log, record, lambda: decision)

    # -- outbound scrubbing (Req 6.x) ---------------------------------------

    def scrub_outbound(self, response: str) -> OutboundResult:
        """Scrub an outbound ``response``; mask PII, redact secrets, else pass through.

        Scans the backend LLM's ``response`` for PII (emails, SSNs, financial
        account and payment-card numbers) and secret patterns such as API keys
        before it is returned to the user (Req 6.1). Every detected PII value is
        replaced with a masking token (Req 6.2) and every detected secret is
        redacted (Req 6.3).

        When at least one value is replaced, exactly one
        :class:`~acdp.models.AuditAction.MASKING_APPLIED` record is appended
        recording only the count and category of the masked items — never the
        raw values (Req 6.4) — before the scrubbed response is returned
        (audit-before-act, Req 12.4).

        When the response contains no PII or secrets, it is returned unchanged
        and no masking operation is performed and no audit record is written
        (Req 6.5). Returns the :class:`OutboundResult`.
        """
        scrubbed, summaries = self._mask_sensitive(response)

        # Req 6.5: nothing detected -> return the response byte-for-byte
        # unchanged and perform no masking or audit operation.
        if not summaries:
            return OutboundResult(response=response, masked=False, summaries=[])

        result = OutboundResult(response=scrubbed, masked=True, summaries=summaries)

        # Req 6.4: record only counts + categories of masked items, never any
        # raw masked/redacted value. The detail below is derived solely from the
        # category labels and their counts.
        record = AuditRecord(
            timestamp=datetime.now(timezone.utc),
            actor_id=self._actor_id,
            action=AuditAction.MASKING_APPLIED,
            outcome="masked",
            detail={
                "total_masked": result.total_masked,
                "categories": {
                    summary.category.value: summary.count for summary in summaries
                },
            },
        )
        # Audit-before-act: the scrubbed response is only returned once the
        # masking record is durably appended (Req 12.4).
        return guarded_action(self._audit_log, record, lambda: result)

    def _mask_sensitive(
        self, response: str
    ) -> tuple[str, list[MaskedItemSummary]]:
        """Replace every detected sensitive value; return scrubbed text + summaries.

        Applies each detector in order (most specific first) so a value is
        attributed to a single category and not double-counted (Req 6.2, 6.3).
        Returns the scrubbed text and one :class:`MaskedItemSummary` per category
        that matched — counts and categories only, never raw values (Req 6.4).
        """
        scrubbed = response
        counts: dict[MaskingCategory, int] = {}

        for category, pattern in _DETECTORS:
            token = DEFAULT_MASKING_TOKENS[category]
            # ``subn`` returns the replaced text and the number of substitutions,
            # giving us the per-category count without ever retaining the raw
            # matched value.
            scrubbed, replaced = pattern.subn(token, scrubbed)
            if replaced:
                counts[category] = counts.get(category, 0) + replaced

        summaries = [
            MaskedItemSummary(category=category, count=count)
            for category, count in counts.items()
        ]
        return scrubbed, summaries
