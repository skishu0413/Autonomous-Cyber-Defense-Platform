"""Target scope registry and fail-closed authorization (Layer 3 — Policy).

The :class:`ScopeRegistry` is the operator's record of which assets agents may
act on and for how long. Every :class:`~acdp.models.TargetScope` carries a
required ``expires_at`` (enforced by the model itself), so a defined scope is
always bounded in time (Req 11.3). Defining a scope is an *auditable* action:
the registry appends a :class:`~acdp.models.AuditRecord` describing the scope
before the scope is recorded, using the *audit-before-act* helper
:func:`~acdp.audit.guarded_action`. If the audit append fails, the scope is not
recorded and the :class:`~acdp.exceptions.AuditWriteError` surfaces to the
caller.

The :class:`AuthorizationService` is the fail-closed gate every action-taking
agent routes through: it grants a request only when the target asset is a
member of a currently valid (non-expired, non-revoked) scope, denies everything
else, and records every decision — with the requesting agent and target asset —
in the audit log (Req 8.3, 9.1, 11.1, 11.2, 11.4, 11.5).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from acdp.audit import AuditLog, guarded_action
from acdp.models import (
    ActionRequest,
    AuditAction,
    AuditRecord,
    AuthorizationDecision,
    TargetScope,
)

__all__ = ["ScopeRegistry", "AuthorizationService"]


class ScopeRegistry:
    """An in-memory registry of operator-defined, expiring target scopes.

    Each defined scope is retained in definition order and its definition is
    recorded in the supplied :class:`~acdp.audit.AuditLog` (Req 11.3). Because
    :class:`~acdp.models.TargetScope.expires_at` is a required field, every
    stored scope necessarily has a populated expiration.
    """

    def __init__(self, audit_log: AuditLog) -> None:
        self._audit_log = audit_log
        self._scopes: list[TargetScope] = []

    def define_scope(self, scope: TargetScope) -> TargetScope:
        """Record ``scope`` (with its required expiration) and audit it.

        Appends a ``SCOPE_DEFINED`` audit record *before* the scope is stored
        (audit-before-act, Req 11.3, 12.4); a failed append raises
        :class:`~acdp.exceptions.AuditWriteError` and leaves the registry
        unchanged. Returns the recorded scope.
        """
        record = AuditRecord(
            timestamp=datetime.now(timezone.utc),
            actor_id="operator",
            action=AuditAction.SCOPE_DEFINED,
            outcome="defined",
            target=scope.scope_id,
            detail={
                "scope_id": scope.scope_id,
                "assets": ",".join(scope.assets),
                "expires_at": scope.expires_at.isoformat(),
            },
        )

        def _store() -> TargetScope:
            self._scopes.append(scope)
            return scope

        # Audit-before-act: the scope is only recorded once the audit append
        # is durably committed.
        return guarded_action(self._audit_log, record, _store)

    def get_active_scopes(self, at: datetime) -> list[TargetScope]:
        """Return the defined scopes that are active at ``at``, in definition order.

        A scope is active iff it is neither revoked nor expired at ``at`` (see
        :meth:`~acdp.models.TargetScope.is_active`).
        """
        return [scope for scope in self._scopes if scope.is_active(at)]


class AuthorizationService:
    """Fail-closed, scope-gated authorization (Req 11.1, 11.2, 11.4).

    Given a set of operator-defined :class:`~acdp.models.TargetScope` records,
    :meth:`authorize` grants a request *if and only if* the requested asset is
    a member of at least one scope that is neither expired nor revoked at the
    evaluation time. Every other case — the asset in no scope, the only
    matching scope expired, the only matching scope revoked, or the scope set
    otherwise unverifiable — fails closed to ``DENY`` with **no side effect**
    beyond recording the decision (Property 1).

    Every decision (grant or deny) is recorded in the supplied
    :class:`~acdp.audit.AuditLog` with the requesting agent and target asset
    (Req 11.5, Property 3). The audit append happens *before* the decision is
    returned (audit-before-act, Req 12.4); a failed append raises
    :class:`~acdp.exceptions.AuditWriteError` and no decision is returned.
    """

    def __init__(
        self,
        scopes: Iterable[TargetScope],
        audit: AuditLog,
    ) -> None:
        self._scopes: list[TargetScope] = list(scopes)
        self._audit_log = audit

    def authorize(
        self, request: ActionRequest, at: datetime
    ) -> AuthorizationDecision:
        """Return a fail-closed authorization decision for ``request`` at ``at``.

        Grants iff ``request.asset`` is a member of at least one scope that is
        active (neither expired nor revoked) at ``at``; denies otherwise. The
        decision, requesting agent, and target asset are recorded in the audit
        log before the decision is returned (Req 11.5).
        """
        # Positively match the asset to a currently valid scope. Anything that
        # is not a positive match — out of scope, expired, or revoked — falls
        # through to DENY (fail closed, Req 11.1, 11.2, 11.4).
        matching_scope = next(
            (
                scope
                for scope in self._scopes
                if scope.is_active(at) and request.asset in scope.assets
            ),
            None,
        )

        if matching_scope is not None:
            decision = AuthorizationDecision(
                grant=True,
                reason="asset within active scope",
                scope_id=matching_scope.scope_id,
            )
        else:
            decision = AuthorizationDecision(
                grant=False,
                reason="no active scope authorizes this asset",
                scope_id=None,
            )

        # Record the decision (grant/deny), requesting agent, and target asset
        # (Req 11.5, Property 3). Audit-before-return: a failed append raises
        # AuditWriteError and the caller receives no decision.
        record = AuditRecord(
            timestamp=datetime.now(timezone.utc),
            actor_id=request.agent_id,
            action=(
                AuditAction.AUTHZ_GRANT
                if decision.grant
                else AuditAction.AUTHZ_DENY
            ),
            outcome="grant" if decision.grant else "deny",
            target=request.asset,
            detail={
                "agent_id": request.agent_id,
                "asset": request.asset,
                "action": request.action,
                "decision": "grant" if decision.grant else "deny",
                "reason": decision.reason,
                **(
                    {"scope_id": decision.scope_id}
                    if decision.scope_id is not None
                    else {}
                ),
            },
        )

        return guarded_action(self._audit_log, record, lambda: decision)
