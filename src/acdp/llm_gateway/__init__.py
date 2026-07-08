"""Provider-agnostic LLM Gateway over local models (Layer 1 — Foundation).

Agents reason and embed text through a narrow, provider-agnostic
:class:`LLMGateway` interface so no component is coupled to a specific model
(Req 4.1, 4.2). The default :class:`OllamaGateway` serves both reasoning and
embeddings from a locally hosted Ollama instance (``http://localhost:11434`` by
default) over HTTP using ``httpx``.

The Gateway treats model selection as explicit and honest:

* A request may name a model. The Gateway routes to that exact name, attempting
  it even when it is unavailable, and never silently falls back to another model
  (Req 4.3, 4.4). When the model is unavailable it raises
  :class:`~acdp.exceptions.ModelUnavailableError` carrying the offending name.
* A per-request timeout aborts the call and raises
  :class:`~acdp.exceptions.LLMTimeoutError` (Req 4.5).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx

from acdp.exceptions import LLMTimeoutError, ModelUnavailableError
from acdp.models import LLMRequest, LLMResponse, PlatformConfig

__all__ = ["LLMGateway", "OllamaGateway"]


@runtime_checkable
class LLMGateway(Protocol):
    """Provider-agnostic gateway to local reasoning and embedding models."""

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Route a reasoning request to the configured/named local model.

        Routes to ``request.model`` when set, otherwise to the configured
        reasoning model, and returns the model response (Req 4.1). Raises
        :class:`~acdp.exceptions.ModelUnavailableError` naming the model when it
        is unavailable, never substituting a different model (Req 4.3, 4.4).
        Raises :class:`~acdp.exceptions.LLMTimeoutError` when the request exceeds
        the configured timeout (Req 4.5).
        """
        ...

    def embed(self, text: str, model: str | None = None) -> list[float]:
        """Return an embedding vector from the configured embedding model.

        Routes to ``model`` when set, otherwise to the configured embedding
        model (Req 4.2). Raises
        :class:`~acdp.exceptions.ModelUnavailableError` naming the model when it
        is unavailable, and :class:`~acdp.exceptions.LLMTimeoutError` on timeout.
        """
        ...


class OllamaGateway:
    """An :class:`LLMGateway` backed by a local Ollama server.

    The gateway calls Ollama's HTTP API (``/api/generate`` for reasoning,
    ``/api/embeddings`` for embeddings). The base URL, default model names, and
    per-request timeout are taken from :class:`~acdp.models.PlatformConfig` so
    the same configuration that boots the platform drives model routing.
    """

    def __init__(
        self,
        config: PlatformConfig,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        """Create a gateway from platform configuration.

        Args:
            config: Platform configuration supplying ``ollama_url``,
                ``reasoning_model``, ``embedding_model``, and
                ``llm_timeout_seconds``.
            client: Optional pre-configured ``httpx.Client``. Injecting a client
                (e.g. with a custom transport) enables deterministic testing
                without a live Ollama server. When omitted, a client bound to
                ``config.ollama_url`` is created.
        """
        self._config = config
        self._base_url = config.ollama_url.rstrip("/")
        self._timeout = config.llm_timeout_seconds
        self._client = client or httpx.Client(base_url=self._base_url)

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Route a reasoning request to the resolved model (Req 4.1, 4.3-4.5)."""
        # Resolve the target model: an explicit request model wins and is used
        # verbatim; otherwise fall back to the configured reasoning model. We
        # never substitute one named model for another (Req 4.3, 4.4).
        model = request.model or self._config.reasoning_model

        payload: dict[str, object] = {
            "model": model,
            "prompt": request.prompt,
            "stream": False,
        }
        if request.system is not None:
            payload["system"] = request.system

        data = self._post("/api/generate", payload, model)
        return LLMResponse(model=model, content=str(data.get("response", "")))

    def embed(self, text: str, model: str | None = None) -> list[float]:
        """Return an embedding vector for ``text`` (Req 4.2)."""
        # Named model wins; otherwise use the configured embedding model. As
        # with generate, no silent substitution across model names.
        target = model or self._config.embedding_model

        data = self._post(
            "/api/embeddings",
            {"model": target, "prompt": text},
            target,
        )
        embedding = data.get("embedding", [])
        return [float(value) for value in embedding]

    def _post(
        self,
        path: str,
        payload: dict[str, object],
        model: str,
    ) -> dict[str, object]:
        """POST ``payload`` to Ollama, mapping transport failures to platform errors.

        Raises:
            LLMTimeoutError: If the request exceeds the configured timeout (Req 4.5).
            ModelUnavailableError: If Ollama reports the model is unavailable,
                naming the offending model without substituting (Req 4.3, 4.4).
        """
        try:
            response = self._client.post(path, json=payload, timeout=self._timeout)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"LLM request to model {model!r} exceeded "
                f"{self._timeout}s timeout"
            ) from exc

        # Ollama returns 404 when the model is not present locally. Treat any
        # model-not-found signal as an unavailable model rather than a fallback.
        if response.status_code == 404 or self._is_model_missing(response):
            raise ModelUnavailableError(model)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ModelUnavailableError(
                model,
                f"Model {model!r} request failed with status "
                f"{response.status_code}",
            ) from exc

        return response.json()

    @staticmethod
    def _is_model_missing(response: httpx.Response) -> bool:
        """Return True if the response body indicates the model is not available."""
        try:
            body = response.json()
        except ValueError:
            return False
        if not isinstance(body, dict):
            return False
        error = str(body.get("error", "")).lower()
        return "not found" in error or "no such model" in error or "try pulling" in error
