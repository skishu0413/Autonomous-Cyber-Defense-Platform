"""Smoke test for platform boot (Task 20.3).

Asserts that the platform boots with a valid config and reaches a ready state
(Req 13.1) without requiring real Ollama or Qdrant services.

External dependencies are stubbed via:
- A fake httpx transport injected into OllamaGateway so no live Ollama is needed.
- InMemoryVectorStore (the default in Platform.from_config when use_qdrant=False).
- A temporary file for the audit log (cleaned up by pytest's tmp_path fixture).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import httpx
import pytest

from acdp.main import Platform
from acdp.models import PlatformConfig


# ---------------------------------------------------------------------------
# Fake HTTP transport — simulates an Ollama server for embed/generate calls
# ---------------------------------------------------------------------------

class _FakeOllamaTransport(httpx.BaseTransport):
    """Minimal fake Ollama HTTP transport.

    Returns valid JSON for /api/generate and /api/embeddings so that
    OllamaGateway works without a live Ollama instance.
    """

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path

        if path == "/api/generate":
            body = '{"model": "llama3", "response": "OK", "done": true}'
            return httpx.Response(200, content=body.encode(), headers={"content-type": "application/json"})

        if path == "/api/embeddings":
            # Return a small 3-dimensional embedding vector.
            body = '{"embedding": [0.1, 0.2, 0.3]}'
            return httpx.Response(200, content=body.encode(), headers={"content-type": "application/json"})

        # Any other path — return a generic 200 so health-checks also pass.
        return httpx.Response(200, content=b'{}', headers={"content-type": "application/json"})


def _fake_ollama_gateway_factory(config: PlatformConfig):
    """Build an OllamaGateway whose HTTP client uses the fake transport."""
    from acdp.llm_gateway import OllamaGateway
    fake_client = httpx.Client(transport=_FakeOllamaTransport(), base_url=config.ollama_url)
    return OllamaGateway(config, client=fake_client)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_config(audit_log_path: str) -> PlatformConfig:
    """Return a valid PlatformConfig with a writable audit log path."""
    return PlatformConfig(
        reasoning_model="llama3",
        embedding_model="nomic-embed-text",
        llm_timeout_seconds=5.0,
        similarity_threshold=0.85,
        top_k=5,
        audit_log_path=audit_log_path,
    )


# ---------------------------------------------------------------------------
# Smoke test — platform boot
# ---------------------------------------------------------------------------


def test_platform_boots_and_reaches_ready_state(tmp_path: pytest.TempPathFactory) -> None:
    """Platform boots with a valid config and reaches a ready state (Req 13.1).

    Stubs out Ollama via a fake httpx transport injected into OllamaGateway.
    Uses InMemoryVectorStore (the default when use_qdrant=False) so no Qdrant
    instance is required.
    """
    audit_log_file = str(tmp_path / "audit.jsonl")
    config = _minimal_config(audit_log_file)

    # Patch acdp.main.OllamaGateway so the Platform factory uses our fake.
    with patch("acdp.main.OllamaGateway", side_effect=_fake_ollama_gateway_factory):
        platform = Platform.from_config(config, use_qdrant=False)

    # Core assertion: the platform reports it is ready.
    assert platform.is_ready is True, (
        "Platform.is_ready must be True after successful initialization (Req 13.1)"
    )


def test_platform_boots_from_config_path(tmp_path: pytest.TempPathFactory) -> None:
    """Platform boots from a config file path using from_config_path (Req 13.1).

    Writes a minimal valid YAML config to a temp file and passes its path to
    Platform.from_config_path. Stubs Ollama with a fake transport.
    """
    # Write a minimal YAML config file.
    config_file = tmp_path / "config.yaml"
    audit_log_file = tmp_path / "audit.jsonl"
    config_file.write_text(
        f"reasoning_model: llama3\n"
        f"embedding_model: nomic-embed-text\n"
        f"audit_log_path: {audit_log_file}\n",
        encoding="utf-8",
    )

    with patch("acdp.main.OllamaGateway", side_effect=_fake_ollama_gateway_factory):
        platform = Platform.from_config_path(config_file, use_qdrant=False)

    assert platform.is_ready is True, (
        "Platform.is_ready must be True after loading from a config file path (Req 13.1)"
    )


def test_platform_boot_writes_ready_audit_record(tmp_path: pytest.TempPathFactory) -> None:
    """A platform-ready audit record is written to the audit log on boot.

    The platform appends a startup record to the audit log so that startup is
    auditable (Req 1.6, 13.1).
    """
    audit_log_file = str(tmp_path / "audit.jsonl")
    config = _minimal_config(audit_log_file)

    with patch("acdp.main.OllamaGateway", side_effect=_fake_ollama_gateway_factory):
        platform = Platform.from_config(config, use_qdrant=False)

    records = platform.audit_log.read_all()
    assert len(records) >= 1, (
        "At least one audit record must be written during platform boot."
    )

    # The ready record's outcome should indicate readiness.
    outcomes = {r.outcome for r in records}
    assert "ready" in outcomes, (
        f"Expected a 'ready' audit record after boot. Found outcomes: {outcomes}"
    )


def test_platform_exposes_all_components_after_boot(tmp_path: pytest.TempPathFactory) -> None:
    """All platform components are accessible after boot.

    Verifies that the key sub-systems (audit_log, orchestrator, guardrail_agent,
    blue_team_agent, red_team_agent, devsecops_agent, retriever, authz_service)
    are wired and non-None after initialization.
    """
    audit_log_file = str(tmp_path / "audit.jsonl")
    config = _minimal_config(audit_log_file)

    with patch("acdp.main.OllamaGateway", side_effect=_fake_ollama_gateway_factory):
        platform = Platform.from_config(config, use_qdrant=False)

    assert platform.audit_log is not None, "audit_log must be wired"
    assert platform.orchestrator is not None, "orchestrator must be wired"
    assert platform.guardrail_agent is not None, "guardrail_agent must be wired"
    assert platform.blue_team_agent is not None, "blue_team_agent must be wired"
    assert platform.red_team_agent is not None, "red_team_agent must be wired"
    assert platform.devsecops_agent is not None, "devsecops_agent must be wired"
    assert platform.retriever is not None, "retriever must be wired"
    assert platform.authz_service is not None, "authz_service must be wired"
    assert platform.scope_registry is not None, "scope_registry must be wired"


def test_platform_boots_with_example_config_defaults(tmp_path: pytest.TempPathFactory) -> None:
    """Platform boots correctly when optional config values are omitted (Req 13.5).

    A config with only required / minimally set values should use documented
    defaults and still reach a ready state.
    """
    audit_log_file = str(tmp_path / "audit.jsonl")
    # Construct a PlatformConfig with only the audit_log_path set explicitly;
    # every other field takes its default.
    config = PlatformConfig(audit_log_path=audit_log_file)

    with patch("acdp.main.OllamaGateway", side_effect=_fake_ollama_gateway_factory):
        platform = Platform.from_config(config, use_qdrant=False)

    assert platform.is_ready is True
    # Verify a couple of documented defaults are in effect (Req 13.5).
    assert platform.config.reasoning_model == "llama3"
    assert platform.config.embedding_model == "nomic-embed-text"
    assert platform.config.top_k == 5
    assert platform.config.similarity_threshold == 0.85
