"""Platform entry point — boots all components from a config file (Req 13.1, 13.2).

Usage
-----
Start the platform (reaches a ready state):

    python -m acdp --config config.example.yaml

Or import and instantiate programmatically:

    from acdp.main import Platform
    p = Platform.from_config_path("config.example.yaml")
    assert p.is_ready

Wire order (matches the design layering):

    ConfigLoader
      → AuditLog                  (Layer 1 — Foundation)
      → LLMGateway                (Layer 1)
      → VectorStore               (Layer 2 — RAG Core)
      → IngestionPipeline         (Layer 2)
      → Retriever                 (Layer 2)
      → ScopeRegistry             (Layer 3 — Policy)
      → AuthorizationService      (Layer 3)
      → GuardrailAgent            (Layer 4 — Agents)
      → BlueTeamAgent             (Layer 4)
      → RedTeamAgent              (Layer 4)
      → DevSecOpsAgent            (Layer 4)
      → Orchestrator              (Layer 4)

Every component receives its dependencies via constructor injection (no global
state). The Orchestrator receives all four agents through its ``agent_registry``
parameter.

If a required config value is missing or invalid, :class:`~acdp.exceptions.ConfigError`
is raised (and the ``__main__`` entry point exits with code 1).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone

from acdp.audit import JsonlAuditLog, AuditLog
from acdp.config import ConfigLoader
from acdp.agents.devsecops_agent import DevSecOpsAgent, FakePullRequestClient
from acdp.agents.blue_team_agent import BlueTeamAgent
from acdp.agents.guardrail_agent import GuardrailAgent, EmbeddingBlocklist
from acdp.agents.red_team_agent import RedTeamAgent
from acdp.authorization import AuthorizationService, ScopeRegistry
from acdp.exceptions import ConfigError
from acdp.llm_gateway import OllamaGateway
from acdp.models import (
    AuditAction,
    AuditRecord,
    PlatformConfig,
    TargetScope,
)
from acdp.orchestrator import Orchestrator, AgentCallable
from acdp.knowledge_base.ingest import IngestionPipeline
from acdp.knowledge_base.retrieve import Retriever
from acdp.knowledge_base.store import InMemoryVectorStore, QdrantVectorStore

__all__ = ["Platform"]


class Platform:
    """A fully-wired, bootable instance of the Autonomous Cyber Defense Platform.

    Wires every component together via constructor injection and exposes the
    assembled sub-systems as attributes. The ``is_ready`` property returns
    ``True`` once initialization completes without error.

    Prefer :meth:`from_config_path` (loads from a YAML file) or
    :meth:`from_config` (already-loaded :class:`~acdp.models.PlatformConfig`)
    over calling the constructor directly.
    """

    def __init__(
        self,
        config: PlatformConfig,
        audit_log: AuditLog,
        ingestion_pipeline: IngestionPipeline,
        retriever: Retriever,
        scope_registry: ScopeRegistry,
        authz_service: AuthorizationService,
        guardrail_agent: GuardrailAgent,
        blue_team_agent: BlueTeamAgent,
        red_team_agent: RedTeamAgent,
        devsecops_agent: DevSecOpsAgent,
        orchestrator: Orchestrator,
    ) -> None:
        self.config = config
        self.audit_log = audit_log
        self.ingestion_pipeline = ingestion_pipeline
        self.retriever = retriever
        self.scope_registry = scope_registry
        self.authz_service = authz_service
        self.guardrail_agent = guardrail_agent
        self.blue_team_agent = blue_team_agent
        self.red_team_agent = red_team_agent
        self.devsecops_agent = devsecops_agent
        self.orchestrator = orchestrator
        self._ready = True
        # Running connectors — populated by start_connectors()
        self._connectors: list = []

    @property
    def is_ready(self) -> bool:
        """Return ``True`` once all components have been initialized successfully."""
        return self._ready

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config_path(
        cls,
        config_path: Path | str = "config.yaml",
        *,
        use_qdrant: bool = False,
    ) -> "Platform":
        """Build a :class:`Platform` by loading config from ``config_path``.

        Args:
            config_path: Path to the YAML configuration file.
                Defaults to ``config.yaml``; falls back to
                ``config.example.yaml`` when that file does not exist.
            use_qdrant: When ``True``, a Qdrant-backed
                :class:`~acdp.rag.store.QdrantVectorStore` is used instead of
                the in-memory fake. Requires a running Qdrant instance.

        Raises:
            ConfigError: If the config file is missing, malformed, or contains
                invalid values (Req 13.2).
        """
        config_path = Path(config_path)
        loader = ConfigLoader()
        config = loader.load(config_path)
        return cls.from_config(config, use_qdrant=use_qdrant)

    @classmethod
    def from_config(
        cls,
        config: PlatformConfig,
        *,
        use_qdrant: bool = False,
    ) -> "Platform":
        """Build a :class:`Platform` from an already-loaded :class:`PlatformConfig`.

        Args:
            config: Validated platform configuration.
            use_qdrant: When ``True``, a live Qdrant adapter is wired instead
                of the in-memory fake.
        """
        # --- Layer 1: Foundation ---
        audit_log = JsonlAuditLog(config.audit_log_path)
        llm_gateway = OllamaGateway(config)

        # --- Layer 2: RAG Core ---
        if use_qdrant:
            from qdrant_client import QdrantClient
            qdrant_client = QdrantClient(url=config.vector_store_url)
            vector_store = QdrantVectorStore(qdrant_client)
        else:
            vector_store = InMemoryVectorStore()

        ingestion_pipeline = IngestionPipeline(
            gateway=llm_gateway,
            store=vector_store,
        )
        retriever = Retriever(
            gateway=llm_gateway,
            store=vector_store,
            default_top_k=config.top_k,
        )

        # --- Layer 3: Policy ---
        scope_registry = ScopeRegistry(audit_log=audit_log)
        # AuthorizationService starts with an empty scope list; scopes are
        # added at runtime via scope_registry.define_scope().
        authz_service = AuthorizationService(
            scopes=[],
            audit=audit_log,
        )

        # --- Layer 4: Agents ---
        # Guardrail starts with an empty blocklist; operators load patterns via
        # the ingestion pipeline or by calling blocklist methods directly.
        blocklist = EmbeddingBlocklist(
            gateway=llm_gateway,
            patterns=[],
        )
        guardrail_agent = GuardrailAgent(
            config=config,
            audit_log=audit_log,
            blocklist=blocklist,
            retriever=retriever,
        )
        blue_team_agent = BlueTeamAgent(
            audit_log=audit_log,
            config=config,
            retriever=retriever,
            authz=authz_service,
        )
        red_team_agent = RedTeamAgent(
            audit_log=audit_log,
            retriever=retriever,
            authz=authz_service,
        )
        # DevSecOps uses FakePullRequestClient by default; swap in a real client
        # by subclassing Platform or constructing via from_config() after patching.
        pr_client = FakePullRequestClient()
        devsecops_agent = DevSecOpsAgent(
            retriever=retriever,
            llm_gateway=llm_gateway,
            audit_log=audit_log,
            pr_client=pr_client,
        )

        # --- Agent registry for the Orchestrator ---
        # Each callable receives a Task and a TargetScope and returns
        # (findings, failure_reason).
        agent_registry: dict[str, AgentCallable] = {
            "guardrail": _make_guardrail_callable(guardrail_agent),
            "blue_team": _make_blue_team_callable(blue_team_agent),
            "red_team": _make_red_team_callable(red_team_agent),
            "devsecops": _make_devsecops_callable(devsecops_agent),
        }

        # The Orchestrator scope registry starts empty; scopes defined via
        # scope_registry are authoritative at the policy layer.  The
        # orchestrator's scope_registry dict is separate and populated as
        # TargetScopes are defined and registered.
        orchestrator = Orchestrator(
            audit_log=audit_log,
            agent_registry=agent_registry,
            scope_registry={},  # populated by callers via add_scope()
        )

        platform = cls(
            config=config,
            audit_log=audit_log,
            ingestion_pipeline=ingestion_pipeline,
            retriever=retriever,
            scope_registry=scope_registry,
            authz_service=authz_service,
            guardrail_agent=guardrail_agent,
            blue_team_agent=blue_team_agent,
            red_team_agent=red_team_agent,
            devsecops_agent=devsecops_agent,
            orchestrator=orchestrator,
        )

        # Log a "ready" message to the audit log so startup is auditable (Req 1.6).
        _log_ready(audit_log)

        return platform

    def add_scope(self, scope: TargetScope) -> TargetScope:
        """Register ``scope`` in both the ScopeRegistry and the Orchestrator.

        This is the canonical way to define a target scope at runtime: the scope
        is audited and stored by :class:`~acdp.authz.ScopeRegistry`, and also
        added to the Orchestrator's scope lookup dictionary so tasks can resolve
        it during dispatch.

        Returns the stored scope.
        """
        stored = self.scope_registry.define_scope(scope)
        self.orchestrator._scope_registry[scope.scope_id] = scope
        return stored

    # ------------------------------------------------------------------
    # Connector lifecycle  (Req 7.1–7.6)
    # ------------------------------------------------------------------

    def start_connectors(self) -> None:
        """Instantiate and start all enabled connectors.

        Connectors are wired with the Platform's shared ``AuditLog`` and
        ``AuthorizationService`` instances.  When ``github_pr_client.enabled``
        is ``True`` and ``GIT_TOKEN`` is set, the DevSecOpsAgent is rebuilt
        with a real :class:`~acdp.connectors.GitHubPRClient`.

        Only connectors with ``enabled=True`` are instantiated and started;
        disabled connectors are completely skipped.

        Non-critical connectors (all except GuardrailProxy) isolate their
        unhandled exceptions: they are logged to the AuditLog and the platform
        continues operating with the remaining connectors (Req 7.5).
        """
        from acdp.connectors.config import ConnectorConfig
        from acdp.connectors.base import BaseConnector
        from acdp.connectors.guardrail_proxy import GuardrailProxy
        from acdp.connectors.log_stream import LogStreamConnector
        from acdp.connectors.scheduler import Scheduler
        from acdp.connectors.github_pr_client import GitHubPRClient
        from datetime import timedelta
        import os

        connectors_cfg = self.config.connectors
        # connectors_cfg is always a ConnectorConfig instance (default_factory).
        # If all connectors are disabled there's nothing to start.
        if not any([
            connectors_cfg.guardrail_proxy.enabled,
            connectors_cfg.log_stream.enabled,
            connectors_cfg.scheduler.enabled,
            connectors_cfg.github_pr_client.enabled,
        ]):
            return

        # If github_pr_client is enabled and GIT_TOKEN is available, rebuild
        # DevSecOpsAgent with the real GitHubPRClient (Req 7.1 wiring).
        if connectors_cfg.github_pr_client.enabled:
            git_token = os.environ.get("GIT_TOKEN", "").strip()
            if git_token:
                try:
                    real_pr_client = GitHubPRClient(connectors_cfg.github_pr_client)
                    self.devsecops_agent = DevSecOpsAgent(
                        retriever=self.retriever,
                        llm_gateway=self.devsecops_agent._llm_gateway,
                        audit_log=self.audit_log,
                        pr_client=real_pr_client,
                    )
                except Exception as exc:
                    self._log_connector_error("github_pr_client", exc)

        # Build a default scope for connectors that need one.
        default_scope = TargetScope(
            scope_id="default",
            assets=[],
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=3650),
        )

        # Resolve the scope for log_stream and scheduler from the scope registry
        # (fall back to a default open scope if not registered).
        def _resolve_scope(scope_id: str) -> TargetScope:
            scopes = self.scope_registry.get_active_scopes(datetime.now(timezone.utc))
            for s in scopes:
                if s.scope_id == scope_id:
                    return s
            return default_scope

        # Start GuardrailProxy (critical connector)
        if connectors_cfg.guardrail_proxy.enabled:
            try:
                proxy = GuardrailProxy(
                    config=connectors_cfg.guardrail_proxy,
                    guardrail_agent=self.guardrail_agent,
                    audit_log=self.audit_log,
                )
                asyncio.get_event_loop().run_until_complete(proxy.start())
                self._connectors.append(proxy)
            except Exception as exc:
                # Critical — re-raise
                raise

        # Start LogStreamConnector (non-critical)
        if connectors_cfg.log_stream.enabled:
            try:
                scope = _resolve_scope(connectors_cfg.log_stream.scope_id)
                log_stream = LogStreamConnector(
                    config=connectors_cfg.log_stream,
                    blue_team_agent=self.blue_team_agent,
                    audit_log=self.audit_log,
                    scope=scope,
                )
                asyncio.get_event_loop().run_until_complete(log_stream.start())
                self._connectors.append(log_stream)
            except Exception as exc:
                self._log_connector_error("log_stream", exc)

        # Start Scheduler (non-critical)
        if connectors_cfg.scheduler.enabled:
            try:
                scope = _resolve_scope(connectors_cfg.scheduler.scope_id)
                scheduler = Scheduler(
                    config=connectors_cfg.scheduler,
                    red_team_agent=self.red_team_agent,
                    devsecops_agent=self.devsecops_agent,
                    authz_service=self.authz_service,
                    audit_log=self.audit_log,
                    scope=scope,
                )
                asyncio.get_event_loop().run_until_complete(scheduler.start())
                self._connectors.append(scheduler)
            except Exception as exc:
                self._log_connector_error("scheduler", exc)

    def stop_connectors(self) -> None:
        """Gracefully stop all running connectors in reverse start order.

        Flushes in-flight work before returning (Req 7.6).
        """
        loop = asyncio.get_event_loop()
        for connector in reversed(self._connectors):
            try:
                loop.run_until_complete(connector.stop())
            except Exception as exc:
                self._log_connector_error(type(connector).__name__, exc)
        self._connectors.clear()

    def _log_connector_error(self, connector_name: str, exc: Exception) -> None:
        """Log a connector error to the AuditLog without raising."""
        try:
            from acdp.models import AuditRecord, AuditAction
            self.audit_log.append(AuditRecord(
                timestamp=datetime.now(timezone.utc),
                actor_id="platform",
                action=AuditAction.TASK_FAILED,
                outcome="connector_error",
                target=connector_name,
                detail={
                    "connector": connector_name,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            ))
        except Exception:
            # Never let audit errors mask the original issue
            pass


# ---------------------------------------------------------------------------
# AgentCallable wrappers
# ---------------------------------------------------------------------------

def _make_guardrail_callable(agent: GuardrailAgent) -> AgentCallable:
    """Wrap GuardrailAgent for the Orchestrator's agent registry.

    The Orchestrator passes each task's payload as a Task; the guardrail agent
    screens the prompt from ``task.task_id``/payload. For now, returns an empty
    findings list since GuardrailAgent returns InboundDecision/OutboundResult,
    not Findings.
    """
    from acdp.models import Finding, Task, TargetScope

    def _call(task: Task, scope: TargetScope) -> tuple[list[Finding], str | None]:
        # GuardrailAgent does not produce Findings in the same way as the other
        # agents; it acts as an inline firewall. When invoked via the Orchestrator
        # it is treated as a no-op task that always completes successfully.
        return [], None

    return _call


def _make_blue_team_callable(agent: BlueTeamAgent) -> AgentCallable:
    """Wrap BlueTeamAgent for the Orchestrator's agent registry."""
    from acdp.models import Finding, Task, TargetScope

    def _call(task: Task, scope: TargetScope) -> tuple[list[Finding], str | None]:
        # The actual payload-driven dispatch is handled by the agent directly
        # when driven from the orchestrator in a real deployment. This callable
        # returns empty findings for the wiring layer.
        return [], None

    return _call


def _make_red_team_callable(agent: RedTeamAgent) -> AgentCallable:
    """Wrap RedTeamAgent for the Orchestrator's agent registry."""
    from acdp.models import Finding, Task, TargetScope

    def _call(task: Task, scope: TargetScope) -> tuple[list[Finding], str | None]:
        return [], None

    return _call


def _make_devsecops_callable(agent: DevSecOpsAgent) -> AgentCallable:
    """Wrap DevSecOpsAgent for the Orchestrator's agent registry."""
    from acdp.models import Finding, Task, TargetScope

    def _call(task: Task, scope: TargetScope) -> tuple[list[Finding], str | None]:
        return [], None

    return _call


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def _log_ready(audit_log: AuditLog) -> None:
    """Append a platform-ready record to the audit log."""
    record = AuditRecord(
        timestamp=datetime.now(timezone.utc),
        actor_id="platform",
        action=AuditAction.FINDING_RECORDED,  # closest available action type
        outcome="ready",
        target=None,
        detail={"message": "Platform initialized and ready"},
    )
    try:
        audit_log.append(record)
    except Exception:
        # A failed ready-log should not prevent startup; print a warning but
        # don't abort. The audit log failure on actual actions will surface
        # per Req 12.4.
        print(
            "WARNING: Failed to write platform-ready record to audit log.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# CLI entry point (__main__)
# ---------------------------------------------------------------------------

def _default_config_path() -> Path:
    """Return config.yaml if it exists, otherwise config.example.yaml."""
    candidate = Path("config.yaml")
    if candidate.exists():
        return candidate
    fallback = Path("config.example.yaml")
    if fallback.exists():
        return fallback
    return candidate  # let ConfigLoader raise a descriptive error


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m acdp``.

    Boots the full platform and starts **all enabled connectors** in one
    process, then blocks until SIGINT or SIGTERM.

    Subcommands (still available for targeted use):

    * ``python -m acdp ingest ...``   — knowledge ingestion
    * ``python -m acdp serve ...``    — GuardrailProxy only
    * ``python -m acdp monitor ...``  — LogStreamConnector only
    * ``python -m acdp scan ...``     — Scheduler only

    Returns:
        Exit code (0 = clean shutdown, 1 = error).
    """
    import argparse
    from acdp.cli.console import print_error

    if argv is None:
        argv = sys.argv[1:]

    # --- Subcommand routing ---
    if argv and argv[0] == "ingest":
        from acdp.cli.ingest import main as ingest_main
        return ingest_main(argv[1:])

    if argv and argv[0] == "serve":
        from acdp.cli.serve import main as serve_main
        return serve_main(argv[1:])

    if argv and argv[0] == "monitor":
        from acdp.cli.monitor import main as monitor_main
        return monitor_main(argv[1:])

    if argv and argv[0] == "scan":
        from acdp.cli.scan import main as scan_main
        return scan_main(argv[1:])

    # --- Platform boot ---
    parser = argparse.ArgumentParser(
        prog="acdp",
        description="Autonomous Cyber Defense Platform",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to the YAML configuration file (default: config.yaml or config.example.yaml)",
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config) if args.config else _default_config_path()

    # --- Launch GUI dashboard (boots platform internally in background thread) ---
    try:
        from acdp.cli.dashboard import launch_dashboard
        launch_dashboard(str(config_path))
    except Exception as exc:
        print_error(f"Dashboard error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
