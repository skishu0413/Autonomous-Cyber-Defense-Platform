"""ConnectorConfig — Pydantic v2 configuration models for all production connectors.

All sub-configs default to ``enabled=False`` so the platform boots safely without
any live infrastructure. Operators opt-in by setting ``enabled: true`` in the
``connectors:`` section of ``config.yaml``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "GuardrailProxyConfig",
    "LogStreamConfig",
    "SchedulerConfig",
    "GitHubPRClientConfig",
    "ConnectorConfig",
]


class GuardrailProxyConfig(BaseModel):
    """Configuration for the HTTP Proxy / Guardrail API connector."""

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    upstream_url: str = "http://localhost:11434/api/generate"
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None


class LogStreamConfig(BaseModel):
    """Configuration for the Log Stream Connector."""

    enabled: bool = False
    mode: str = "tail"          # "tail" | "syslog_udp" | "poll_dir"
    log_path: str | None = None  # for tail / poll_dir
    udp_host: str = "0.0.0.0"
    udp_port: int = 514
    poll_interval_seconds: float = 5.0
    retry_interval_seconds: float = 2.0
    max_retries: int = 5
    scope_id: str = "default"


class SchedulerConfig(BaseModel):
    """Configuration for the Scheduler / Red Team connector."""

    enabled: bool = False
    mode: str = "interval"          # "interval" | "cron"
    interval_seconds: float = 3600.0
    cron_expression: str | None = None
    targets: list[str] = Field(default_factory=list)
    auto_remediate: bool = False
    scope_id: str = "default"


class GitHubPRClientConfig(BaseModel):
    """Configuration for the GitHub PR Client connector."""

    enabled: bool = False
    github_api_base_url: str = "https://api.github.com"
    max_retries: int = 3


class ConnectorConfig(BaseModel):
    """Top-level connector configuration containing sub-configs for every connector."""

    guardrail_proxy: GuardrailProxyConfig = Field(
        default_factory=GuardrailProxyConfig
    )
    log_stream: LogStreamConfig = Field(
        default_factory=LogStreamConfig
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig
    )
    github_pr_client: GitHubPRClientConfig = Field(
        default_factory=GitHubPRClientConfig
    )
