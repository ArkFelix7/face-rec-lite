"""FastAPI dependency functions and app-state helpers.

Routes use these via ``Depends()``. Middleware that cannot use ``Depends``
accesses ``get_async_session_maker`` directly, passing ``request.app``.
"""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


def get_async_session_maker(app) -> async_sessionmaker:
    """Return the async sessionmaker stored in *app.state*.

    Called by middleware (which cannot use ``Depends``) to obtain a session
    factory without going through the FastAPI dependency injection system.
    """
    return app.state.session_factory


async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yield a DB session from the app's session factory.

    Rolls back automatically on any unhandled exception so the connection is
    returned to the pool in a clean state.
    """
    session_factory: async_sessionmaker = request.app.state.session_factory
    async with session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def get_face_ml(request: Request):
    """FastAPI dependency: return the FaceMLService from app state."""
    return request.app.state.face_ml


def get_rate_limiter(request: Request):
    """FastAPI dependency: return the RateLimiter from app state."""
    return request.app.state.rate_limiter


def get_settings(request: Request):
    """FastAPI dependency: return the Settings instance from app state."""
    return request.app.state.settings
