"""
Shared fixtures for the face-rec-lite test suite.

Scope hierarchy:
  session  – engine, settings (created once per test run)
  function – db_session (rolls back after each test), client, mock_face_ml
"""

from __future__ import annotations

import asyncio
import base64
import io
import secrets
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from passlib.hash import bcrypt
from PIL import Image
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings
from app.models.api_key import ApiKey
from app.models.base import Base
from app.models.user import User
from app.services.database import DatabaseService
from app.services.face_ml import (
    DuplicateCheckResult,
    FaceDetectionResult,
    FaceMLService,
    MLProcessResult,
    QualityMetrics,
)
from app.utils.image import InvalidImageError


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    return Settings(
        database_url="postgresql+asyncpg://faceapi:faceapi@localhost:5432/facedb_test",
        redis_url="redis://localhost:6379/1",
        min_quality_score=0.3,
        min_face_size_px=50,
        ml_device="cpu",
        bcrypt_rounds=4,  # Fast for tests
    )


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
async def db_engine(test_settings: Settings):
    engine = create_async_engine(test_settings.database_url, echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine):
    """Provide a DB session that rolls back after each test.

    Note: This rolls back the entire transaction so each test starts clean.
    Data added here is NOT visible to other database connections until committed.
    Use this fixture only for unit-style DB tests that don't go through the HTTP layer.
    Integration tests that use `client` should use `api_key_record` which commits.
    """
    async with AsyncSession(db_engine, expire_on_commit=False) as session:
        async with session.begin():
            yield session
            await session.rollback()


# ---------------------------------------------------------------------------
# ML helpers
# ---------------------------------------------------------------------------


def make_good_quality() -> QualityMetrics:
    return QualityMetrics(
        overall_score=0.85,
        blur_score=0.90,
        brightness=0.65,
        face_confidence=0.98,
        face_width_px=200,
        face_height_px=240,
        pitch_deg=2.0,
        yaw_deg=-3.0,
        roll_deg=1.0,
    )


def make_bad_quality() -> QualityMetrics:
    return QualityMetrics(
        overall_score=0.20,
        blur_score=0.10,
        brightness=0.65,
        face_confidence=0.95,
        face_width_px=30,  # Too small
        face_height_px=35,
        pitch_deg=2.0,
        yaw_deg=-3.0,
        roll_deg=1.0,
    )


def make_embedding(seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    emb = rng.randn(512).astype(np.float32)
    return emb / np.linalg.norm(emb)


def make_ml_result(
    embedding_seed: int = 42, quality: QualityMetrics | None = None
) -> tuple[list, MLProcessResult]:
    q = quality or make_good_quality()
    emb = make_embedding(embedding_seed)
    face_result = FaceDetectionResult(
        embedding=emb,
        quality=q,
        bbox=[100.0, 80.0, 300.0, 320.0],
        landmarks=[],
    )
    result = MLProcessResult(face=face_result)
    fake_faces = [object()]  # 1 fake face object
    return fake_faces, result


# ---------------------------------------------------------------------------
# Mock ML service
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_face_ml() -> MagicMock:
    """Mocked FaceMLService that returns a good face by default."""
    mock = MagicMock(spec=FaceMLService)
    mock.process_image.return_value = make_ml_result()
    mock.check_quality_gate.return_value = []  # No failing checks
    mock.check_duplicate.return_value = DuplicateCheckResult(is_duplicate=False)
    return mock


# ---------------------------------------------------------------------------
# Test images (synthetic)
# ---------------------------------------------------------------------------


def make_jpeg_base64(
    width: int = 200, height: int = 200, color: tuple = (120, 80, 60)
) -> str:
    """Create a valid JPEG image as base64 string."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def valid_image_b64() -> str:
    return make_jpeg_base64()


@pytest.fixture
def valid_image_b64_with_prefix() -> str:
    raw = base64.b64decode(make_jpeg_base64())
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode()


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------


@pytest.fixture
async def api_key_record(db_engine) -> tuple[str, ApiKey]:
    """Create and COMMIT a test API key in the database.

    Commits so the key is visible to the auth middleware (which opens its own
    DB session). Cleans up the record after the test.

    Returns (raw_key, ApiKey record).
    """
    # Key format must match production: "sk_live_<48 hex chars>".
    # Auth middleware extracts raw_key[8:16] for the prefix lookup.
    raw_key = "sk_live_" + secrets.token_hex(24)
    key_prefix = raw_key[8:16]  # First 8 chars of the random portion — unique per key
    key_hash = bcrypt.hash(raw_key, rounds=4)
    record = ApiKey(
        key_hash=key_hash,
        key_prefix=key_prefix,
        name="test-key",
        is_active=True,
        rate_limit=10000,  # High limit to avoid rate-limiting in tests
    )
    async with AsyncSession(db_engine, expire_on_commit=False) as session:
        async with session.begin():
            session.add(record)
    # record.id is populated after the commit

    yield raw_key, record

    # Cleanup: remove the API key after the test
    async with AsyncSession(db_engine, expire_on_commit=False) as session:
        async with session.begin():
            from sqlalchemy import delete as _sa_delete
            await session.execute(
                _sa_delete(ApiKey).where(ApiKey.id == record.id)
            )


@pytest.fixture
async def auth_headers(api_key_record) -> dict:
    raw_key, _ = api_key_record
    return {"Authorization": f"Bearer {raw_key}"}


# ---------------------------------------------------------------------------
# App + AsyncClient
# ---------------------------------------------------------------------------


@pytest.fixture
async def test_app(test_settings, db_engine, mock_face_ml):
    """
    Create a FastAPI app with a test lifespan that injects real DB but mocked ML.
    Does NOT load the real ML model or connect to production infrastructure.
    """
    from redis.asyncio import Redis

    from app.middleware.auth import AuthMiddleware
    from app.middleware.rate_limit import RateLimitMiddleware
    from app.middleware.request_logger import RequestLoggerMiddleware
    from app.routers import faces, health, users, verify
    from app.schemas.common import ApiResponse
    from app.services.rate_limiter import RateLimiter

    @asynccontextmanager
    async def test_lifespan(app):
        factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
        redis = Redis.from_url(test_settings.redis_url, decode_responses=True)

        app.state.engine = db_engine
        app.state.session_factory = factory
        app.state.redis = redis
        app.state.rate_limiter = RateLimiter(redis)
        app.state.face_ml = mock_face_ml
        app.state.settings = test_settings

        yield

        await redis.aclose()

    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    app = FastAPI(lifespan=test_lifespan)

    # Middleware (added in reverse execution order)
    app.add_middleware(RequestLoggerMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthMiddleware)

    # Routers
    app.include_router(health.router)
    app.include_router(users.router)
    app.include_router(faces.router)
    app.include_router(verify.router)

    # Exception handlers (mirroring main.py)
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "req_" + secrets.token_hex(6))
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
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "req_" + secrets.token_hex(6))
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "data": None,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal error occurred",
                    "details": {"error": str(exc)},
                },
                "request_id": request_id,
            },
        )

    yield app


@pytest.fixture
async def client(test_app) -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# User helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def created_user(db_session) -> User:
    """Pre-create a user for tests that need one."""
    db = DatabaseService(db_session)
    user = await db.create_user(
        external_id=f"testuser_{uuid.uuid4().hex[:8]}",
        display_name="Test User",
        metadata={},
    )
    await db_session.flush()
    return user
