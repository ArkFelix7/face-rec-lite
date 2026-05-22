"""Fixed-window rate-limiting middleware.

Must be registered **after** :class:`~app.middleware.auth.AuthMiddleware` so
that ``request.state.api_key`` is already populated when this middleware runs.

On every non-public request it calls the Redis-backed
:class:`~app.services.rate_limiter.RateLimiter` and either:

* passes the request through, adding ``X-RateLimit-Limit`` and
  ``X-RateLimit-Remaining`` headers to the response, or
* rejects it with ``429 Too Many Requests`` and a ``Retry-After`` header
  indicating how many seconds remain in the current one-minute window.
"""

from __future__ import annotations

import os
import time

from starlette.middleware.base import BaseHTTPMiddleware

_AUTH_DISABLED = os.environ.get("DISABLE_AUTH", "").lower() in ("true", "1", "yes")
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

PUBLIC_PATHS: frozenset[str] = frozenset({"/v1/health", "/v1/ready", "/v1/metrics"})


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that enforces per-API-key request rate limits."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if _AUTH_DISABLED:
            return await call_next(request)

        # Public paths bypass rate limiting.
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # If auth middleware did not attach an API key (e.g. auth itself failed
        # and returned early), let the request pass through so that the auth
        # middleware's 401 response is returned unchanged.
        api_key = getattr(request.state, "api_key", None)
        if api_key is None:
            return await call_next(request)

        rate_limiter = request.app.state.rate_limiter
        allowed, count, limit = await rate_limiter.check_rate_limit(
            str(api_key.id), api_key.rate_limit
        )

        if not allowed:
            # Seconds remaining in the current one-minute window.
            seconds_into_window = int(time.time()) % 60
            retry_after = 60 - seconds_into_window

            request_id = getattr(request.state, "request_id", "")
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={
                    "success": False,
                    "data": None,
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": (
                            f"Rate limit of {limit} requests per minute exceeded. "
                            f"Retry after {retry_after} seconds."
                        ),
                    },
                    "request_id": request_id,
                },
            )

        response = await call_next(request)

        remaining = max(0, limit - count)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response
