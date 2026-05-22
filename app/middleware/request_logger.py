"""Structured request/response logging middleware.

Logs one structured line per completed request using *structlog*. The log
level is chosen based on the HTTP status code:

* ``INFO``    — 1xx / 2xx / 3xx
* ``WARNING`` — 4xx
* ``ERROR``   — 5xx

A unique ``request_id`` (``req_<12 hex chars>``) is assigned here if the
:class:`~app.middleware.auth.AuthMiddleware` has not already set one (e.g. for
public paths that skip authentication). The ``X-Request-ID`` header is added to
every response so clients can correlate log entries with their requests.
"""

from __future__ import annotations

import secrets
import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that emits a structured log line for every request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.monotonic()

        # Generate a request_id if AuthMiddleware has not set one (public paths).
        if not hasattr(request.state, "request_id"):
            request.state.request_id = "req_" + secrets.token_hex(6)

        response = await call_next(request)

        duration_ms = round((time.monotonic() - start) * 1000, 2)
        status = response.status_code

        api_key_prefix: str | None = None
        api_key = getattr(request.state, "api_key", None)
        if api_key is not None:
            api_key_prefix = api_key.key_prefix

        log_data: dict = {
            "event": "request_completed",
            "request_id": request.state.request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status,
            "duration_ms": duration_ms,
            "api_key_prefix": api_key_prefix,
        }

        if status >= 500:
            logger.error(**log_data)
        elif status >= 400:
            logger.warning(**log_data)
        else:
            logger.info(**log_data)

        response.headers["X-Request-ID"] = request.state.request_id
        return response
