"""Bearer-token authentication middleware.

Extracts the API key from the ``Authorization: Bearer <key>`` header,
looks it up in the database by its 8-character prefix, verifies the
bcrypt hash, and attaches the resolved ``ApiKey`` object to
``request.state.api_key``.

Public paths that bypass authentication:
    * /v1/health
    * /v1/ready
    * /v1/metrics
"""

from __future__ import annotations

import secrets

from passlib.hash import bcrypt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.dependencies import get_async_session_maker
from app.services.database import DatabaseService

PUBLIC_PATHS: frozenset[str] = frozenset({"/v1/health", "/v1/ready", "/v1/metrics"})


def _error_response(status_code: int, code: str, message: str, request_id: str) -> JSONResponse:
    """Return a JSON error response using the standard API envelope format."""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "data": None,
            "error": {"code": code, "message": message},
            "request_id": request_id,
        },
    )


class AuthMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that authenticates every non-public request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip authentication for public endpoints.
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        request_id = "req_" + secrets.token_hex(6)
        request.state.request_id = request_id

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return _error_response(
                401,
                "UNAUTHORIZED",
                "Missing or malformed Authorization header",
                request_id,
            )

        raw_key = auth_header[7:]
        # Keys are "sk_live_<48 hex>"; use chars [8:16] as the lookup prefix
        # (the random portion), not [:8] which is always "sk_live_" for all keys.
        prefix = raw_key[8:16]

        session_factory = get_async_session_maker(request.app)
        async with session_factory() as session:
            db = DatabaseService(session)

            api_key = await db.get_api_key_by_prefix(prefix)
            if api_key is None or not api_key.is_active:
                return _error_response(
                    401,
                    "UNAUTHORIZED",
                    "Invalid or inactive API key",
                    request_id,
                )

            # Verify the full key against the stored bcrypt hash.
            try:
                valid = bcrypt.verify(raw_key, api_key.key_hash)
            except Exception:
                valid = False

            if not valid:
                return _error_response(
                    401,
                    "UNAUTHORIZED",
                    "Invalid API key",
                    request_id,
                )

            # Persist the last-used timestamp and commit before releasing the
            # session so the update is not lost.
            await db.update_api_key_last_used(api_key.id)
            await session.commit()

        # Make the resolved key available to downstream middleware and routes.
        request.state.api_key = api_key

        return await call_next(request)
