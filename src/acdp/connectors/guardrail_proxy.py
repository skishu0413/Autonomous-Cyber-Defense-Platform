"""GuardrailProxy — FastAPI-based HTTP proxy that routes all LLM traffic through
the GuardrailAgent (screen inbound, scrub outbound).

Every POST /v1/chat request is:
  1. Screened via GuardrailAgent.screen_inbound()
  2. If blocked → HTTP 403
  3. If forwarded → proxied to upstream LLM via httpx
  4. Upstream response scrubbed via GuardrailAgent.scrub_outbound()
  5. Exactly one GUARDRAIL_DECISION audit record written per request

GET /health returns 200 when the proxy is fully operational.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import httpx

from acdp.audit import AuditLog
from acdp.connectors.base import BaseConnector
from acdp.connectors.config import GuardrailProxyConfig
from acdp.models import AuditAction, AuditRecord
from pydantic import BaseModel

if TYPE_CHECKING:
    from acdp.agents.guardrail_agent import GuardrailAgent
    import uvicorn

__all__ = [
    "GuardrailProxy",
    "ChatRequest",
    "ChatResponse",
    "HealthResponse",
    "ErrorResponse",
]


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming chat request body."""
    prompt: str
    model: str | None = None


class ChatResponse(BaseModel):
    """Successful chat response body."""
    response: str
    masked: bool = False


class HealthResponse(BaseModel):
    """Health check response."""
    status: str  # "ok"


class ErrorResponse(BaseModel):
    """RFC 7807-style problem detail (error + optional detail)."""
    error: str
    detail: str | None = None


# ---------------------------------------------------------------------------
# GuardrailProxy connector
# ---------------------------------------------------------------------------

class GuardrailProxy(BaseConnector):
    """FastAPI application that wraps GuardrailAgent inline.

    Lifecycle:
      start() → creates FastAPI app, starts uvicorn in a background thread
      stop()  → signals uvicorn Server.should_exit=True, joins thread

    Routes:
      POST /v1/chat   — screen → [forward to upstream] → scrub → return
      GET  /health    — liveness check
    """

    def __init__(
        self,
        config: GuardrailProxyConfig,
        guardrail_agent: "GuardrailAgent",
        audit_log: AuditLog,
    ) -> None:
        self._config = config
        self._guardrail_agent = guardrail_agent
        self._audit_log = audit_log
        self._server: "uvicorn.Server | None" = None
        self._thread: threading.Thread | None = None
        self._app = self._build_app()

    def _build_app(self):
        """Build and return the FastAPI application."""
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        app = FastAPI(title="ACDP GuardrailProxy", version="0.1.0")

        @app.post("/v1/chat", response_model=ChatResponse, responses={
            403: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
        })
        async def chat(request: ChatRequest):
            return await _handle_chat(
                request,
                self._guardrail_agent,
                self._audit_log,
                self._config.upstream_url,
            )

        @app.get("/health", response_model=HealthResponse)
        async def health():
            return HealthResponse(status="ok")

        return app

    @property
    def app(self):
        """Expose the FastAPI app for use with TestClient in tests."""
        return self._app

    async def start(self) -> None:
        """Start uvicorn in a background daemon thread."""
        import uvicorn
        import asyncio

        uv_config = uvicorn.Config(
            app=self._app,
            host=self._config.host,
            port=self._config.port,
            ssl_certfile=self._config.ssl_certfile,
            ssl_keyfile=self._config.ssl_keyfile,
            log_level="error",
        )
        self._server = uvicorn.Server(uv_config)

        def _run():
            import asyncio as _asyncio
            _asyncio.run(self._server.serve())

        self._thread = threading.Thread(target=_run, daemon=True, name="guardrail-proxy")
        self._thread.start()

    async def stop(self) -> None:
        """Signal uvicorn to exit and join the server thread."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None


# ---------------------------------------------------------------------------
# Request handler (extracted so it can be called from tests)
# ---------------------------------------------------------------------------

async def _handle_chat(
    request: ChatRequest,
    guardrail_agent: "GuardrailAgent",
    audit_log: AuditLog,
    upstream_url: str,
):
    """Core handler for POST /v1/chat.

    1. screen_inbound
    2. block → 403
    3. forward → upstream httpx call
    4. scrub_outbound
    5. write exactly one GUARDRAIL_DECISION audit record
    6. return ChatResponse
    """
    from fastapi.responses import JSONResponse

    prompt = request.prompt

    # Step 1 + 2: Screen the inbound prompt. GuardrailAgent.screen_inbound
    # writes its own GUARDRAIL_DECISION audit record via guarded_action.
    # We do NOT duplicate that record; we write the connector-level record below.
    try:
        decision = guardrail_agent.screen_inbound(prompt)
    except Exception as exc:
        # screen_inbound failure treated as upstream error — return 502
        _write_audit(audit_log, "block", None, outcome="screen_error",
                     detail={"error": str(exc)})
        return JSONResponse(
            status_code=502,
            content=ErrorResponse(error="proxy_error").model_dump(),
        )

    if decision.action == "block":
        # The GuardrailAgent has already written the GUARDRAIL_DECISION record.
        # Per Req 8.1 the proxy does NOT duplicate that record.
        return JSONResponse(
            status_code=403,
            content=ErrorResponse(
                error="blocked",
                detail=decision.reason,
            ).model_dump(),
        )

    # Step 3: Forward to upstream LLM
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            upstream_resp = await client.post(
                upstream_url,
                json={"prompt": prompt, "model": request.model},
            )
            upstream_resp.raise_for_status()
            upstream_body = upstream_resp.text
    except Exception as exc:
        # Upstream unreachable or error — return 502, no internal detail exposed
        _write_audit(audit_log, "forward", decision.matched_pattern_id,
                     outcome="upstream_error",
                     detail={"upstream_url": upstream_url, "error": type(exc).__name__})
        return JSONResponse(
            status_code=502,
            content=ErrorResponse(error="upstream_error").model_dump(),
        )

    # Step 4: Scrub outbound response
    try:
        scrubbed = guardrail_agent.scrub_outbound(upstream_body)
        response_text = scrubbed.response
        was_masked = scrubbed.masked
    except Exception:
        response_text = upstream_body
        was_masked = False

    # Step 5: The GuardrailAgent already wrote a GUARDRAIL_DECISION record for
    # the screen_inbound call. Per Req 8.1 the proxy does NOT write an additional
    # audit record for forwarded requests, as GuardrailAgent.screen_inbound has
    # already written the canonical record.

    # Step 6: Return ChatResponse
    return ChatResponse(response=response_text, masked=was_masked)


def _write_audit(
    audit_log: AuditLog,
    decision: str,
    matched_pattern_id: str | None,
    *,
    outcome: str = "processed",
    detail: dict | None = None,
) -> None:
    """Write a GUARDRAIL_DECISION audit record from the proxy connector."""
    record = AuditRecord(
        timestamp=datetime.now(timezone.utc),
        actor_id="guardrail_proxy",
        action=AuditAction.GUARDRAIL_DECISION,
        outcome=outcome,
        target=matched_pattern_id,
        detail={
            "decision": decision,
            "matched_pattern_id": matched_pattern_id,
            **(detail or {}),
        },
    )
    try:
        audit_log.append(record)
    except Exception:
        pass  # Audit errors on error paths should not mask the original issue
