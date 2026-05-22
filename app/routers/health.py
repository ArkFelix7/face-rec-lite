"""Health, readiness, and Prometheus metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text
from starlette.responses import Response

from app.schemas.common import ApiResponse, HealthResponse, ReadyResponse, generate_request_id

router = APIRouter(tags=["health"])


@router.get("/v1/health")
async def health() -> dict:
    """Lightweight liveness check — always returns 200 if the process is up."""
    request_id = generate_request_id()
    response = ApiResponse.ok(
        HealthResponse(status="ok", version="1.0.0"),
        request_id=request_id,
    )
    return response.model_dump()


@router.get("/v1/ready")
async def ready(request: Request) -> Response:
    """Readiness check — verifies DB, Redis, and ML model are available."""
    request_id = generate_request_id()
    checks: dict[str, str] = {}
    all_ok = True

    # --- Database -----------------------------------------------------------
    try:
        session_factory = request.app.state.session_factory
        async with session_factory() as session:
            await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["database"] = f"error: {exc}"
        all_ok = False

    # --- Redis --------------------------------------------------------------
    try:
        rate_limiter = request.app.state.rate_limiter
        redis_ok = await rate_limiter.ping()
        checks["redis"] = "ok" if redis_ok else "error: ping failed"
        if not redis_ok:
            all_ok = False
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = f"error: {exc}"
        all_ok = False

    # --- ML model -----------------------------------------------------------
    try:
        face_ml = request.app.state.face_ml
        checks["ml_model"] = "ok" if face_ml is not None else "error: not loaded"
        if face_ml is None:
            all_ok = False
    except AttributeError:
        checks["ml_model"] = "error: not loaded"
        all_ok = False

    if all_ok:
        response = ApiResponse.ok(
            ReadyResponse(status="ready", checks=checks),
            request_id=request_id,
        )
        return JSONResponse(status_code=200, content=response.model_dump())

    response = ApiResponse.fail(
        code="SERVICE_UNAVAILABLE",
        message="One or more dependency checks failed",
        details={"checks": checks},
        request_id=request_id,
    )
    return JSONResponse(status_code=503, content=response.model_dump())


@router.get("/v1/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics in text format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
