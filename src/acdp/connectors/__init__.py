"""Production connectors package.

Exports all connector classes and configuration models so callers can import
from ``acdp.connectors`` without knowing the internal module layout.
"""

from acdp.connectors.base import BaseConnector
from acdp.connectors.config import (
    ConnectorConfig,
    GitHubPRClientConfig,
    GuardrailProxyConfig,
    LogStreamConfig,
    SchedulerConfig,
)
from acdp.connectors.github_pr_client import GitHubPRClient
from acdp.connectors.guardrail_proxy import GuardrailProxy
from acdp.connectors.log_stream import LogStreamConnector
from acdp.connectors.scheduler import Scheduler

__all__ = [
    "BaseConnector",
    "ConnectorConfig",
    "GitHubPRClientConfig",
    "GuardrailProxyConfig",
    "LogStreamConfig",
    "SchedulerConfig",
    "GitHubPRClient",
    "GuardrailProxy",
    "LogStreamConnector",
    "Scheduler",
]
