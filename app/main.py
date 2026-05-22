"""FastAPI application factory and lifespan manager for face-rec-lite."""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings, settings
from app.middleware.auth import AuthMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.request_logger import RequestLoggerMiddleware
from app.routers import faces, health, users, verify
from app.services.face_ml import FaceMLService
from app.services.rate_limiter import RateLimiter

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize shared resources on startup, clean up on shutdown."""
    logger.info("Starting up face-rec-lite API")

    # --- Database -----------------------------------------------------------
    engine = create_async_engine(
        settings.database_url,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        echo=(settings.app_env == "development"),
    )
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    app.state.engine = engine
    app.state.session_factory = session_factory

    # --- Redis --------------------------------------------------------------
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    app.state.redis = redis
    app.state.rate_limiter = RateLimiter(redis)

    # --- ML Model -----------------------------------------------------------
    # Skip loading if already attached (e.g. injected in tests via create_app).
    if not hasattr(app.state, "face_ml") or app.state.face_ml is None:
        logger.info(
            "Loading face recognition model",
            model=settings.insightface_model_name,
        )
        face_ml = FaceMLService(settings)
        app.state.face_ml = face_ml
        logger.info("Face recognition model loaded")
    else:
        logger.info("Face recognition model already loaded (injected), skipping init")

    # Expose settings on app state for dependency injection
    app.state.settings = settings

    yield  # Application runs here

    # --- Cleanup ------------------------------------------------------------
    logger.info("Shutting down face-rec-lite API")
    await redis.aclose()
    await engine.dispose()


def create_app(
    cfg: Settings = settings,
    face_ml: FaceMLService | None = None,
) -> FastAPI:
    """Application factory.

    Args:
        cfg: Settings instance (defaults to the module-level singleton).
        face_ml: Optional pre-constructed FaceMLService. When provided the
                 lifespan will skip model loading, which is useful in tests.
    """
    app = FastAPI(
        title="Face Recognition API",
        version="1.0.0",
        openapi_url="/v1/openapi.json",
        docs_url="/v1/docs",
        redoc_url="/v1/redoc",
        lifespan=lifespan,
    )

    # Inject test overrides before lifespan starts if provided
    if face_ml is not None:
        app.state.face_ml = face_ml

    # -----------------------------------------------------------------------
    # Middleware — Starlette applies in LIFO order (last added = outermost).
    # CORS must be outermost so preflight OPTIONS requests are answered before
    # AuthMiddleware rejects them with 401.
    # -----------------------------------------------------------------------
    app.add_middleware(RequestLoggerMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Routers
    # -----------------------------------------------------------------------
    app.include_router(health.router)
    app.include_router(users.router)
    app.include_router(faces.router)
    app.include_router(verify.router)

    # -----------------------------------------------------------------------
    # Global exception handlers
    # -----------------------------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = getattr(
            request.state, "request_id", "req_" + secrets.token_hex(6)
        )
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "data": None,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": {"errors": exc.errors()},
                },
                "request_id": request_id,
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        request_id = getattr(
            request.state, "request_id", "req_" + secrets.token_hex(6)
        )
        logger.error(
            "Unhandled exception",
            exc_info=exc,
            path=request.url.path,
        )
        # Hide internal details in production
        error_details = (
            None
            if cfg.app_env == "production"
            else {"error": str(exc)}
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "data": None,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal error occurred",
                    "details": error_details,
                },
                "request_id": request_id,
            },
        )

    return app


# Module-level application instance — used by uvicorn / gunicorn.
app = create_app()
