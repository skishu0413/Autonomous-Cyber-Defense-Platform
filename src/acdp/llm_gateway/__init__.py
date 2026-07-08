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

__all__ = ["LLMGateway", "OllamaGateway", "OpenAIGateway", "AnthropicGateway", "create_gateway"]


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


# ---------------------------------------------------------------------------
# OpenAI Gateway (ChatGPT — gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.)
# ---------------------------------------------------------------------------

class OpenAIGateway:
    """An :class:`LLMGateway` backed by the OpenAI API (ChatGPT).

    Credentials are read exclusively from the ``OPENAI_API_KEY`` environment
    variable — never from config files. The reasoning model (e.g. ``gpt-4o``)
    and embedding model (e.g. ``text-embedding-3-small``) are taken from
    :class:`~acdp.models.PlatformConfig`.

    Install the optional dependency before use::

        pip install openai>=1.0

    Supported reasoning models:  gpt-4o, gpt-4-turbo, gpt-4, gpt-3.5-turbo
    Supported embedding models:  text-embedding-3-small, text-embedding-3-large,
                                  text-embedding-ada-002
    """

    def __init__(self, config: PlatformConfig) -> None:
        self._config = config
        self._client = self._build_client()

    def _build_client(self):
        import os
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai>=1.0"
            )
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before starting ACDP: export OPENAI_API_KEY=sk-..."
            )
        return OpenAI(api_key=api_key)

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Send a chat completion request to OpenAI."""
        model = request.model or self._config.reasoning_model
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        try:
            from openai import APITimeoutError, APIStatusError
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=self._config.llm_timeout_seconds,
            )
            content = response.choices[0].message.content or ""
            return LLMResponse(model=model, content=content)
        except APITimeoutError as exc:
            raise LLMTimeoutError(
                f"OpenAI request to model {model!r} exceeded "
                f"{self._config.llm_timeout_seconds}s timeout"
            ) from exc
        except APIStatusError as exc:
            if exc.status_code == 404:
                raise ModelUnavailableError(model) from exc
            raise ModelUnavailableError(
                model, f"OpenAI API error {exc.status_code}: {exc.message}"
            ) from exc

    def embed(self, text: str, model: str | None = None) -> list[float]:
        """Return an embedding vector using OpenAI embeddings API."""
        target = model or self._config.embedding_model
        try:
            response = self._client.embeddings.create(
                model=target,
                input=text,
                timeout=self._config.llm_timeout_seconds,
            )
            return [float(v) for v in response.data[0].embedding]
        except Exception as exc:
            raise ModelUnavailableError(
                target, f"OpenAI embedding error: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Anthropic Gateway (Claude — claude-3-5-sonnet, claude-3-opus, etc.)
# ---------------------------------------------------------------------------

class AnthropicGateway:
    """An :class:`LLMGateway` backed by the Anthropic API (Claude).

    Credentials are read exclusively from the ``ANTHROPIC_API_KEY`` environment
    variable — never from config files. The reasoning model (e.g.
    ``claude-3-5-sonnet-20241022``) is taken from
    :class:`~acdp.models.PlatformConfig`.

    Note: Anthropic does not provide a native embedding API. When used as the
    platform provider, embeddings fall back to a simple hash-based stub that
    returns a deterministic zero-padded vector. For production use with Claude,
    pair it with a separate embedding provider or use the Ollama gateway for
    embeddings only.

    Install the optional dependency before use::

        pip install anthropic>=0.30

    Supported reasoning models:  claude-3-5-sonnet-20241022, claude-3-opus-20240229,
                                   claude-3-haiku-20240307
    """

    # Anthropic does not expose an embeddings endpoint. We use a fixed
    # dimension matching nomic-embed-text so the vector store stays compatible.
    _EMBED_DIM = 768

    def __init__(self, config: PlatformConfig) -> None:
        self._config = config
        self._client = self._build_client()

    def _build_client(self):
        import os
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic>=0.30"
            )
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Export it before starting ACDP: export ANTHROPIC_API_KEY=sk-ant-..."
            )
        return anthropic.Anthropic(api_key=api_key)

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Send a messages request to Anthropic Claude."""
        import anthropic
        model = request.model or self._config.reasoning_model
        kwargs: dict = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            kwargs["system"] = request.system
        try:
            response = self._client.messages.create(**kwargs)
            content = response.content[0].text if response.content else ""
            return LLMResponse(model=model, content=content)
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError(
                f"Anthropic request to model {model!r} exceeded timeout"
            ) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code == 404:
                raise ModelUnavailableError(model) from exc
            raise ModelUnavailableError(
                model, f"Anthropic API error {exc.status_code}"
            ) from exc

    def embed(self, text: str, model: str | None = None) -> list[float]:
        """Anthropic has no embeddings API — returns a deterministic stub vector.

        For production embeddings with Claude, set ``llm_provider: anthropic``
        for reasoning and keep a separate Ollama instance for embeddings, or
        switch to ``openai`` which provides both.
        """
        import hashlib
        # Deterministic stub: hash text into a reproducible unit vector.
        digest = hashlib.sha256(text.encode()).digest()
        raw = [(b / 255.0) - 0.5 for b in digest]
        # Pad or truncate to target dimension
        while len(raw) < self._EMBED_DIM:
            raw.extend(raw)
        raw = raw[:self._EMBED_DIM]
        # Normalize to unit vector
        norm = sum(v * v for v in raw) ** 0.5 or 1.0
        return [v / norm for v in raw]


# ---------------------------------------------------------------------------
# Factory — pick the right gateway from config
# ---------------------------------------------------------------------------

def create_gateway(config: PlatformConfig) -> LLMGateway:
    """Instantiate the correct :class:`LLMGateway` based on ``config.llm_provider``.

    Provider routing:

    +--------------+--------------------+----------------------------------+
    | Provider     | Class              | Required env var                 |
    +==============+====================+==================================+
    | ``ollama``   | OllamaGateway      | none (local server)              |
    +--------------+--------------------+----------------------------------+
    | ``openai``   | OpenAIGateway      | OPENAI_API_KEY                   |
    +--------------+--------------------+----------------------------------+
    | ``anthropic``| AnthropicGateway   | ANTHROPIC_API_KEY                |
    +--------------+--------------------+----------------------------------+

    Raises:
        ValueError: If ``config.llm_provider`` is not a recognised value.
        EnvironmentError: If a required API key env var is missing.
        ImportError: If the provider's optional dependency is not installed.
    """
    provider = config.llm_provider
    if provider == "ollama":
        return OllamaGateway(config)
    if provider == "openai":
        return OpenAIGateway(config)
    if provider == "anthropic":
        return AnthropicGateway(config)
    raise ValueError(
        f"Unknown llm_provider {provider!r}. "
        "Supported values: 'ollama', 'openai', 'anthropic'"
    )
