"""Property-based tests for the LLM Gateway routing contract (Req 4.3, 4.4)."""

from __future__ import annotations

import json

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from acdp.exceptions import LLMTimeoutError, ModelUnavailableError
from acdp.llm_gateway import OllamaGateway
from acdp.models import LLMRequest, PlatformConfig
from tests.strategies import model_names


def _recording_gateway(
    routed_models: list[str],
    *,
    available: bool,
    config: PlatformConfig,
) -> OllamaGateway:
    """Build an OllamaGateway backed by a fake transport that records routing.

    The transport appends the ``model`` field of every outgoing request body to
    ``routed_models`` so tests can assert exactly which model name was routed.
    When ``available`` is False it emulates Ollama's model-not-found response
    (HTTP 404 with an ``error`` body), which the gateway maps to
    :class:`ModelUnavailableError`.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        routed_models.append(body["model"])
        if not available:
            return httpx.Response(
                404,
                json={"error": f"model '{body['model']}' not found, try pulling it first"},
            )
        return httpx.Response(200, json={"response": "ok", "embedding": [0.0]})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url=config.ollama_url)
    return OllamaGateway(config, client=client)


# Feature: autonomous-cyber-defense-platform, Property 10: LLM Gateway never
# silently substitutes a model — For any requested model name, the gateway SHALL
# route the outgoing request to exactly that model name; if the model is
# unavailable it SHALL raise an error naming that model and SHALL NOT route to
# any other model.
# Validates: Requirements 4.3, 4.4
@settings(max_examples=200)
@given(
    requested_model=model_names(),
    prompt=st.text(max_size=64),
    available=st.booleans(),
)
def test_gateway_never_silently_substitutes_model(
    requested_model: str,
    prompt: str,
    available: bool,
) -> None:
    # Configure the platform with a *different* default reasoning model so that
    # any substitution (e.g. silently falling back to the configured default)
    # would be observable as a routed model name != the requested name.
    config = PlatformConfig(
        reasoning_model=f"__default__{requested_model}",
        embedding_model=f"__default_embed__{requested_model}",
    )
    routed_models: list[str] = []
    gateway = _recording_gateway(routed_models, available=available, config=config)

    request = LLMRequest(prompt=prompt, model=requested_model)

    if available:
        response = gateway.generate(request)
        # The outgoing request routed to exactly the requested model name...
        assert routed_models == [requested_model]
        # ...and the response reports that same model (no substitution).
        assert response.model == requested_model
    else:
        # An unavailable model raises a named error rather than substituting.
        with pytest.raises(ModelUnavailableError) as exc_info:
            gateway.generate(request)
        # The error names exactly the requested model.
        assert exc_info.value.model_name == requested_model
        # The gateway attempted to route to the requested model (Req 4.4) and
        # never routed to any *other* model as a fallback (Req 4.3).
        assert routed_models == [requested_model]


# Unit test: a reasoning request that exceeds the configured timeout must
# surface as an LLMTimeoutError rather than a raw transport exception (Req 4.5).
# The fake transport raises httpx.TimeoutException immediately, so the test is
# deterministic and fast (no real sleeping / no live Ollama server).
# Validates: Requirements 4.5
def test_generate_raises_llm_timeout_error_on_request_timeout() -> None:
    config = PlatformConfig(reasoning_model="llama3", llm_timeout_seconds=0.5)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated request timeout", request=request)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url=config.ollama_url)
    gateway = OllamaGateway(config, client=client)

    request = LLMRequest(prompt="analyze this alert", model="llama3")

    with pytest.raises(LLMTimeoutError) as exc_info:
        gateway.generate(request)

    # The error message names the offending model and the configured timeout so
    # the failure is clearly attributable to the operator.
    message = str(exc_info.value)
    assert "llama3" in message
    assert "0.5" in message
