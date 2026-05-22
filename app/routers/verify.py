"""Face verification endpoint."""

from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.dependencies import get_db_session, get_face_ml, get_settings
from app.schemas.common import ApiResponse
from app.schemas.face import (
    FaceScoreItem,
    QueryFaceQuality,
    VerifyFaceRequest,
    VerifyFaceResponse,
)
from app.services.database import DatabaseService
from app.services.face_ml import FaceMLService
from app.utils.image import InvalidImageError, decode_image
from app.utils.metrics import VERIFICATION_COUNTER

router = APIRouter(tags=["verify"])


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or ""


@router.post("/v1/users/{user_id}/verify")
async def verify_face(
    user_id: str,
    body: VerifyFaceRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    face_ml: FaceMLService = Depends(get_face_ml),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    """Verify a query face against all enrolled faces for a user.

    Returns the best-matching face, a similarity score, and whether the score
    meets the configured (or caller-supplied) threshold.

    No quality gate is applied — quality metrics are returned for caller
    information only.
    """
    start_time = time.monotonic()
    request_id = _get_request_id(request)
    db = DatabaseService(session)

    # 1. Verify user exists
    user = await db.get_user_by_external_id(user_id)
    if user is None:
        VERIFICATION_COUNTER.labels(result="error").inc()
        response = ApiResponse.fail(
            code="USER_NOT_FOUND",
            message=f"No user found with external_id '{user_id}'.",
            request_id=request_id,
        )
        return JSONResponse(status_code=404, content=response.model_dump())

    # 2. Load stored face embeddings
    stored_faces = await db.get_faces_with_embeddings_for_user(user.id)
    if len(stored_faces) == 0:
        VERIFICATION_COUNTER.labels(result="error").inc()
        response = ApiResponse.fail(
            code="USER_HAS_NO_FACES",
            message=f"User '{user_id}' has no enrolled faces to verify against.",
            request_id=request_id,
        )
        return JSONResponse(status_code=409, content=response.model_dump())

    # 3. Decode query image
    try:
        image_bgr = decode_image(body.image)
    except InvalidImageError as exc:
        VERIFICATION_COUNTER.labels(result="error").inc()
        response = ApiResponse.fail(
            code="INVALID_IMAGE",
            message=str(exc),
            request_id=request_id,
        )
        return JSONResponse(status_code=400, content=response.model_dump())

    # 4. Run ML detection on query image
    raw_faces, result = face_ml.process_image(image_bgr)

    if len(raw_faces) == 0:
        VERIFICATION_COUNTER.labels(result="error").inc()
        response = ApiResponse.fail(
            code="NO_FACE_DETECTED",
            message="No face was detected in the provided image.",
            request_id=request_id,
        )
        return JSONResponse(status_code=400, content=response.model_dump())

    if len(raw_faces) > 1:
        VERIFICATION_COUNTER.labels(result="error").inc()
        response = ApiResponse.fail(
            code="MULTIPLE_FACES",
            message="Multiple faces detected. Please provide an image with exactly one face.",
            details={"face_count": len(raw_faces)},
            request_id=request_id,
        )
        return JSONResponse(status_code=400, content=response.model_dump())

    if result is None:
        VERIFICATION_COUNTER.labels(result="error").inc()
        response = ApiResponse.fail(
            code="NO_FACE_DETECTED",
            message="Face detection did not produce a usable result.",
            request_id=request_id,
        )
        return JSONResponse(status_code=400, content=response.model_dump())

    # 5. Determine threshold
    threshold = body.threshold if body.threshold is not None else settings.default_verification_threshold

    # 6. Compute similarities against all stored embeddings
    query_embedding = result.face.embedding  # already L2-normalized float32

    all_scores: list[tuple[str, float]] = [
        (str(f.id), FaceMLService.cosine_similarity(query_embedding, np.array(f.embedding, dtype=np.float32)))
        for f in stored_faces
    ]
    all_scores.sort(key=lambda x: x[1], reverse=True)

    best_face_id, best_score = all_scores[0]
    match = best_score >= threshold

    processing_time_ms = (time.monotonic() - start_time) * 1000

    VERIFICATION_COUNTER.labels(result="match" if match else "no_match").inc()

    quality = result.face.quality
    response = ApiResponse.ok(
        VerifyFaceResponse(
            match=match,
            confidence=best_score,
            threshold_used=threshold,
            best_matching_face_id=best_face_id,
            user_id=user_id,
            query_face_quality=QueryFaceQuality(
                overall_score=quality.overall_score,
                blur_score=quality.blur_score,
                brightness=quality.brightness,
                face_confidence=quality.face_confidence,
            ),
            all_scores=[FaceScoreItem(face_id=fid, similarity=sim) for fid, sim in all_scores],
            processing_time_ms=processing_time_ms,
        ),
        request_id=request_id,
    )
    return JSONResponse(status_code=200, content=response.model_dump())
