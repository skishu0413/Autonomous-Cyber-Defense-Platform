"""DevSecOps Agent — vulnerability remediation via pull requests (Layer 4 — Agents).

The :class:`DevSecOpsAgent` receives a vulnerability :class:`~acdp.models.Finding`
and, if the referenced repository is within the active :class:`~acdp.models.TargetScope`,
analyzes the code, retrieves secure-coding guidance from the RAG Core, generates a
proposed fix, and opens a review-required pull request referencing the guidance
(Req 10.1-10.4).

If the finding's asset references a repository that is outside the active scope,
the agent declines to act and records the decision in the Audit Log (Req 10.5).

The :class:`PullRequestClient` Protocol abstracts the Git hosting provider so
credentials (e.g. ``GIT_TOKEN``) come exclusively from environment variables,
never from ``config.yaml`` (security principle). The :class:`FakePullRequestClient`
implements the same Protocol in-memory for deterministic testing.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from acdp.audit import AuditLog, guarded_action
from acdp.models import (
    AuditAction,
    AuditRecord,
    Finding,
    LLMRequest,
    SourceCategory,
    TargetScope,
)
from acdp.knowledge_base.retrieve import Retriever

__all__ = [
    "PullRequest",
    "Declination",
    "PullRequestClient",
    "FakePullRequestClient",
    "DevSecOpsAgent",
]

# Sentinel agent identifier embedded in every audit record produced by this agent.
_AGENT_ID = "devsecops_agent"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class PullRequest:
    """A pull request opened by the DevSecOps Agent.

    Fields:
        pr_id: Unique identifier for the pull request.
        repo: The repository the PR targets.
        title: The pull request title.
        body: The pull request description, including vulnerability summary and
            references to secure-coding guidance retrieved from the RAG Core (Req 10.4).
        patch: The proposed fix as a diff/patch string (Req 10.1, 10.2).
        requires_review: Always ``True``; PRs are opened in a state that requires
            human review before merge (Req 10.3).
        created_at: UTC timestamp of when the PR was created.
    """

    pr_id: str
    repo: str
    title: str
    body: str
    patch: str
    requires_review: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Declination:
    """A record of the agent declining to act on an out-of-scope finding.

    Fields:
        finding_id: The identifier of the finding that was declined.
        reason: The human-readable reason for declining (always scope-related).
        declined_at: UTC timestamp of when the declination was recorded.
    """

    finding_id: str
    reason: str
    declined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# PullRequestClient Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PullRequestClient(Protocol):
    """Abstraction over a Git hosting provider's pull-request API.

    Implementations MUST source credentials exclusively from environment
    variables (e.g. ``os.environ.get("GIT_TOKEN")``), never from
    ``config.yaml`` or any configuration file.

    The :class:`FakePullRequestClient` implements this Protocol in-memory for
    deterministic testing without any live Git hosting service.
    """

    def create_pr(
        self,
        repo: str,
        title: str,
        body: str,
        branch: str,
        patch: str,
    ) -> PullRequest:
        """Open a review-required pull request on ``repo``.

        Args:
            repo: The repository identifier (e.g. ``"org/repo"``).
            title: A concise pull request title.
            body: A full description of the vulnerability and proposed fix,
                referencing secure-coding guidance source IDs (Req 10.4).
            branch: The branch name for the fix (e.g. ``"fix/vuln-<pr_id>``).
            patch: The proposed fix as a unified diff / patch string (Req 10.2).

        Returns:
            A :class:`PullRequest` with ``requires_review=True`` (Req 10.3).
        """
        ...


# ---------------------------------------------------------------------------
# FakePullRequestClient (for tests)
# ---------------------------------------------------------------------------


class FakePullRequestClient:
    """An in-memory :class:`PullRequestClient` for deterministic testing.

    Records every :meth:`create_pr` call in :attr:`created_prs` so test code
    can assert on the full set of opened pull requests without hitting any live
    Git hosting service.

    Credentials are still sourced from the environment variable
    ``GIT_TOKEN`` to validate the contract, but the value is not actually
    used for any network call.
    """

    def __init__(self) -> None:
        # Validate the credential source contract (ignored in tests — just
        # ensures callers don't pass credentials via config).
        self._token: str | None = os.environ.get("GIT_TOKEN")
        self.created_prs: list[PullRequest] = []

    def create_pr(
        self,
        repo: str,
        title: str,
        body: str,
        branch: str,
        patch: str,
    ) -> PullRequest:
        """Record and return a new in-memory pull request (``requires_review=True``)."""
        pr = PullRequest(
            pr_id=str(uuid.uuid4()),
            repo=repo,
            title=title,
            body=body,
            patch=patch,
            requires_review=True,
            created_at=datetime.now(timezone.utc),
        )
        self.created_prs.append(pr)
        return pr


# ---------------------------------------------------------------------------
# DevSecOpsAgent
# ---------------------------------------------------------------------------

# LLM Gateway Protocol (narrow interface used here to avoid circular imports)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from acdp.llm_gateway import LLMGateway


class DevSecOpsAgent:
    """DevSecOps Agent: code analysis, fix generation, and PR-based remediation.

    For an in-scope finding:

    1. Retrieve secure-coding guideline context from the RAG Core using the
       finding's detail as the query, filtered to the ``COMPLIANCE`` category
       (Req 10.4).
    2. Use the LLM Gateway to analyze the vulnerability and generate a proposed
       fix (Req 10.1).
    3. Open a review-required pull request containing the fix and a description
       that references the retrieved guidance source IDs (Req 10.2, 10.3, 10.4).
    4. Emit a ``PR_OPENED`` audit record with the ``pr_id`` and ``context_refs``
       (never raw code or secrets) (Req 12.1).

    For an out-of-scope finding:

    * Decline to act and emit a ``REMEDIATION_DECLINED`` audit record (Req 10.5).

    Constructor Args:
        retriever: RAG retriever for fetching secure-coding guidance.
        llm_gateway: LLM Gateway for code analysis and fix generation.
        audit_log: Append-only log for ``PR_OPENED`` and ``REMEDIATION_DECLINED``
            records.
        pr_client: Pull-request client abstracting the Git hosting provider.
            Credentials MUST come from environment variables, not config.
        actor_id: Actor identifier embedded in audit records.
    """

    def __init__(
        self,
        retriever: "Retriever",
        llm_gateway: "LLMGateway",
        audit_log: AuditLog,
        pr_client: PullRequestClient,
        *,
        actor_id: str = _AGENT_ID,
    ) -> None:
        self._retriever = retriever
        self._llm_gateway = llm_gateway
        self._audit_log = audit_log
        self._pr_client = pr_client
        self._actor_id = actor_id

    # ------------------------------------------------------------------
    # Remediation  (Req 10.1-10.5)
    # ------------------------------------------------------------------

    def remediate(
        self, finding: Finding, scope: TargetScope
    ) -> PullRequest | Declination:
        """Remediate a vulnerability finding by opening a review-required PR or declining.

        Checks whether ``finding.asset`` is within ``scope.assets`` and the scope
        is currently active. If so, analyzes the code, retrieves secure-coding
        guidance, generates a fix, and opens a review-required pull request
        referencing the guidance (Req 10.1-10.4). If not, declines and records
        the decision in the audit log (Req 10.5).

        Args:
            finding: The vulnerability finding to remediate. ``finding.asset``
                should be a repository reference (e.g. ``"org/repo"``).
            scope: The operator-defined target scope to authorize against.

        Returns:
            A :class:`PullRequest` when the finding is in scope (and a PR is
            opened), or a :class:`Declination` when the finding is out of scope.
        """
        now = datetime.now(timezone.utc)
        repo = finding.asset or ""

        # Req 10.5: decline if the asset is not in the scope or the scope is
        # not active. Fail closed — any ambiguity (missing asset, expired scope,
        # revoked scope) resolves to a declination.
        in_scope = bool(repo) and repo in scope.assets and scope.is_active(now)

        if not in_scope:
            return self._decline(finding, scope, now)

        # Req 10.1: analyze code and generate a proposed fix.
        # Req 10.4: retrieve secure-coding guideline context from RAG Core.
        return self._open_pr(finding, scope, now)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decline(
        self, finding: Finding, scope: TargetScope, now: datetime
    ) -> Declination:
        """Record a REMEDIATION_DECLINED audit entry and return the Declination."""
        repo = finding.asset or "<none>"
        reason = (
            f"Repository {repo!r} is not within the active target scope "
            f"(scope_id={scope.scope_id!r}). Remediation declined."
        )
        declination = Declination(
            finding_id=finding.finding_id,
            reason=reason,
            declined_at=now,
        )

        record = AuditRecord(
            timestamp=now,
            actor_id=self._actor_id,
            action=AuditAction.REMEDIATION_DECLINED,
            outcome="declined",
            target=repo,
            detail={
                "finding_id": finding.finding_id,
                "asset": repo,
                "scope_id": scope.scope_id,
                "reason": reason,
            },
        )
        # Audit-before-act: the declination is only returned once the audit
        # record is durably committed (Req 12.4).
        return guarded_action(self._audit_log, record, lambda: declination)

    def _open_pr(
        self, finding: Finding, scope: TargetScope, now: datetime
    ) -> PullRequest:
        """Retrieve guidance, generate a fix, open a PR, and audit PR_OPENED."""
        repo = finding.asset or ""

        # Req 10.4: retrieve secure-coding guideline context. We use the finding
        # detail as the semantic query and filter to COMPLIANCE. A non-empty
        # context is preferred; an empty result set is handled gracefully (the
        # PR is still opened, with an empty guidance reference list).
        guidance_chunks = self._retriever.retrieve(
            query=finding.detail,
            category=SourceCategory.COMPLIANCE.value,
        )
        # Fall back to an unfiltered search if the compliance category yields
        # nothing, to maximize the chance of grounding the PR in guidance.
        if not guidance_chunks:
            guidance_chunks = self._retriever.retrieve(query=finding.detail)

        guidance_source_ids: list[str] = [
            c.chunk.source_id for c in guidance_chunks
        ]

        # Req 10.1: use the LLM Gateway to analyze the vulnerability and
        # generate a proposed fix. The prompt grounds the LLM in the finding
        # detail and available secure-coding guidance.
        guidance_summary = (
            ", ".join(guidance_source_ids) if guidance_source_ids else "none retrieved"
        )
        fix_prompt = (
            f"You are a security engineer. Analyze the following vulnerability "
            f"finding and generate a minimal, secure code fix.\n\n"
            f"Repository: {repo}\n"
            f"Severity: {finding.severity.value}\n"
            f"Title: {finding.title}\n"
            f"Detail: {finding.detail}\n\n"
            f"Secure-coding guidance references: {guidance_summary}\n\n"
            f"Produce a unified diff patch that fixes the vulnerability. "
            f"Do not introduce new vulnerabilities. Keep changes minimal."
        )
        llm_response = self._llm_gateway.generate(
            LLMRequest(
                prompt=fix_prompt,
                system=(
                    "You are a secure code reviewer. Output only the proposed "
                    "unified diff patch with no additional commentary."
                ),
            )
        )
        patch = llm_response.content

        # Build the PR body referencing the guidance (Req 10.4).
        pr_body = _build_pr_body(finding, guidance_source_ids)

        # Derive a short branch name from the finding id.
        branch = f"fix/vuln-{finding.finding_id[:8]}"

        # Req 10.2: open the pull request containing the fix and description.
        # Req 10.3: requires_review is always True.
        pr = self._pr_client.create_pr(
            repo=repo,
            title=f"Security fix: {finding.title}",
            body=pr_body,
            branch=branch,
            patch=patch,
        )
        # Guarantee the requires_review flag even if the client forgets it.
        if not pr.requires_review:
            pr.requires_review = True

        # Emit PR_OPENED audit record; detail carries pr_id and context_refs
        # but NEVER raw code, secrets, or patch content (Req 12.1, 12.4).
        record = AuditRecord(
            timestamp=now,
            actor_id=self._actor_id,
            action=AuditAction.PR_OPENED,
            outcome="opened",
            target=repo,
            detail={
                "finding_id": finding.finding_id,
                "pr_id": pr.pr_id,
                "repo": repo,
                "branch": branch,
                "context_refs": guidance_source_ids,
                "requires_review": pr.requires_review,
                "scope_id": scope.scope_id,
            },
        )
        return guarded_action(self._audit_log, record, lambda: pr)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_pr_body(finding: Finding, guidance_source_ids: list[str]) -> str:
    """Build a pull request description referencing the vulnerability and guidance.

    The body summarises the finding (without embedding raw secrets or PII) and
    lists the secure-coding guidance source IDs retrieved from the RAG Core so
    the reviewer can trace the fix to authoritative references (Req 10.4).
    """
    guidance_section = (
        "\n".join(f"- {src_id}" for src_id in guidance_source_ids)
        if guidance_source_ids
        else "_No guidance references retrieved from RAG Core._"
    )

    return (
        f"## Security Fix: {finding.title}\n\n"
        f"**Severity:** {finding.severity.value}\n\n"
        f"**Vulnerability detail:**\n{finding.detail}\n\n"
        f"**Secure-coding guidance references (from RAG Core):**\n"
        f"{guidance_section}\n\n"
        f"---\n"
        f"*This pull request was automatically generated by the DevSecOps Agent. "
        f"Human review is required before merge.*\n\n"
        f"Finding ID: `{finding.finding_id}`"
    )
