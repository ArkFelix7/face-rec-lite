"""Face enrollment, listing, and deletion endpoints."""

from __future__ import annotations

import base64
import re
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.dependencies import get_db_session, get_face_ml, get_settings
from app.schemas.common import ApiResponse
from app.schemas.face import (
    BoundingBoxResponse,
    DeleteFaceResponse,
    EnrollFaceRequest,
    EnrollFaceResponse,
    FaceListItem,
    FaceSizeResponse,
    FailingCheck,
    HeadPoseResponse,
    ListFacesResponse,
    LowQualityDetails,
    QualityMetricsResponse,
)
from app.services.database import DatabaseService
from app.services.face_ml import FaceMLService
from app.utils.image import InvalidImageError, compute_image_hash, decode_image
from app.utils.metrics import ENROLLMENT_COUNTER

router = APIRouter(tags=["faces"])


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or ""


def _strip_data_uri(image: str) -> str:
    """Remove data URI prefix from a base64 image string if present."""
    match = re.match(r"data:image/[^;]+;base64,", image)
    if match:
        return image[match.end():]
    return image


@router.post("/v1/users/{user_id}/faces", status_code=201)
async def enroll_face(
    user_id: str,
    body: EnrollFaceRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    face_ml: FaceMLService = Depends(get_face_ml),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    """Enroll a new face for a user.

    Full pipeline:
    1. Verify user exists.
    2. Check face count limit.
    3. Decode image.
    4. Hash-based deduplication.
    5. ML detection (exactly one face required).
    6. Quality gate.
    7. Embedding-based deduplication.
    8. Persist and return.
    """
    request_id = _get_request_id(request)
    db = DatabaseService(session)

    # 1. Verify user exists
    user = await db.get_user_by_external_id(user_id)
    if user is None:
        ENROLLMENT_COUNTER.labels(result="error").inc()
        response = ApiResponse.fail(
            code="USER_NOT_FOUND",
            message=f"No user found with external_id '{user_id}'.",
            request_id=request_id,
        )
        return JSONResponse(status_code=404, content=response.model_dump())

    # 2. Check face count limit
    current_face_count = await db.get_face_count_for_user(user.id)
    if current_face_count >= settings.max_faces_per_user:
        ENROLLMENT_COUNTER.labels(result="error").inc()
        response = ApiResponse.fail(
            code="MAX_FACES_REACHED",
            message=(
                f"User already has {current_face_count} enrolled faces, "
                f"which is the maximum allowed ({settings.max_faces_per_user})."
            ),
            details={"current": current_face_count, "max": settings.max_faces_per_user},
            request_id=request_id,
        )
        return JSONResponse(status_code=409, content=response.model_dump())

    # 3. Decode image
    try:
        image_bgr = decode_image(body.image)
    except InvalidImageError as exc:
        ENROLLMENT_COUNTER.labels(result="error").inc()
        response = ApiResponse.fail(
            code="INVALID_IMAGE",
            message=str(exc),
            request_id=request_id,
        )
        return JSONResponse(status_code=400, content=response.model_dump())

    # 4. Compute image hash for deduplication (use raw bytes from base64 input)
    stripped_b64 = _strip_data_uri(body.image).strip()
    try:
        raw_bytes = base64.b64decode(stripped_b64)
    except Exception:
        raw_bytes = body.image.encode("utf-8")

    image_hash = compute_image_hash(raw_bytes)

    # 5a. Hash-based deduplication (same exact image bytes)
    existing_by_hash = await db.get_face_by_hash_for_user(user.id, image_hash)
    if existing_by_hash is not None:
        ENROLLMENT_COUNTER.labels(result="duplicate_skipped").inc()
        quality_metrics = QualityMetricsResponse(
            overall_score=existing_by_hash.quality_score,
            blur_score=existing_by_hash.blur_score,
            brightness=existing_by_hash.brightness,
            face_confidence=existing_by_hash.face_confidence,
            face_size=FaceSizeResponse(
                width_px=existing_by_hash.face_width_px,
                height_px=existing_by_hash.face_height_px,
            ),
            head_pose=HeadPoseResponse(
                pitch_deg=existing_by_hash.pitch_deg,
                yaw_deg=existing_by_hash.yaw_deg,
                roll_deg=existing_by_hash.roll_deg,
            ),
        )
        response = ApiResponse.ok(
            EnrollFaceResponse(
                face_id=str(existing_by_hash.id),
                user_id=user_id,
                label=existing_by_hash.label,
                quality_metrics=quality_metrics,
                bounding_box=BoundingBoxResponse(x=0.0, y=0.0, width=0.0, height=0.0),
                enrolled_at=existing_by_hash.enrolled_at.isoformat(),
                total_faces_for_user=current_face_count,
                duplicate=True,
                duplicate_of_face_id=str(existing_by_hash.id),
                similarity=1.0,
                message="Identical image already enrolled.",
            ),
            request_id=request_id,
        )
        return JSONResponse(status_code=200, content=response.model_dump())

    # 5b. ML face detection
    raw_faces, result = face_ml.process_image(image_bgr)

    if len(raw_faces) == 0:
        ENROLLMENT_COUNTER.labels(result="error").inc()
        response = ApiResponse.fail(
            code="NO_FACE_DETECTED",
            message="No face was detected in the provided image.",
            request_id=request_id,
        )
        return JSONResponse(status_code=400, content=response.model_dump())

    if len(raw_faces) > 1:
        ENROLLMENT_COUNTER.labels(result="error").inc()
        response = ApiResponse.fail(
            code="MULTIPLE_FACES",
            message="Multiple faces detected. Please provide an image with exactly one face.",
            details={"face_count": len(raw_faces)},
            request_id=request_id,
        )
        return JSONResponse(status_code=400, content=response.model_dump())

    if result is None:
        ENROLLMENT_COUNTER.labels(result="error").inc()
        response = ApiResponse.fail(
            code="NO_FACE_DETECTED",
            message="Face detection did not produce a usable result.",
            request_id=request_id,
        )
        return JSONResponse(status_code=400, content=response.model_dump())

    # 6. Quality gate
    quality = result.face.quality
    quality_threshold = body.quality_threshold if body.quality_threshold is not None else settings.min_quality_score
    failing_checks = face_ml.check_quality_gate(quality, quality_threshold)

    if failing_checks:
        ENROLLMENT_COUNTER.labels(result="quality_rejected").inc()
        low_quality_details = LowQualityDetails(
            overall_score=quality.overall_score,
            threshold=quality_threshold,
            failing_checks=[
                FailingCheck(
                    check=fc["check"],
                    value=fc["value"],
                    minimum=fc["minimum"],
                    reason=fc["reason"],
                )
                for fc in failing_checks
            ],
        )
        response = ApiResponse.fail(
            code="LOW_QUALITY",
            message="The image did not meet the minimum quality requirements.",
            details=low_quality_details.model_dump(),
            request_id=request_id,
        )
        return JSONResponse(status_code=400, content=response.model_dump())

    # 7. Embedding-based deduplication
    embedding = result.face.embedding  # already L2-normalized float32

    existing_faces = await db.get_faces_for_user(user.id)
    stored_embeddings: list[tuple[str, np.ndarray]] = [
        (str(f.id), np.array(f.embedding, dtype=np.float32))
        for f in existing_faces
    ]

    dup_result = face_ml.check_duplicate(embedding, stored_embeddings, settings.dedup_threshold)
    if dup_result.is_duplicate:
        ENROLLMENT_COUNTER.labels(result="duplicate_skipped").inc()
        quality_metrics = QualityMetricsResponse(
            overall_score=quality.overall_score,
            blur_score=quality.blur_score,
            brightness=quality.brightness,
            face_confidence=quality.face_confidence,
            face_size=FaceSizeResponse(
                width_px=quality.face_width_px,
                height_px=quality.face_height_px,
            ),
            head_pose=HeadPoseResponse(
                pitch_deg=quality.pitch_deg,
                yaw_deg=quality.yaw_deg,
                roll_deg=quality.roll_deg,
            ),
        )
        bbox = result.face.bbox
        bounding_box = BoundingBoxResponse(
            x=bbox[0],
            y=bbox[1],
            width=bbox[2] - bbox[0],
            height=bbox[3] - bbox[1],
        )
        response = ApiResponse.ok(
            EnrollFaceResponse(
                face_id=dup_result.existing_face_id or "",
                user_id=user_id,
                label=body.label,
                quality_metrics=quality_metrics,
                bounding_box=bounding_box,
                enrolled_at=datetime.now(timezone.utc).isoformat(),
                total_faces_for_user=current_face_count,
                duplicate=True,
                duplicate_of_face_id=dup_result.existing_face_id,
                similarity=dup_result.similarity,
                message="A visually identical face is already enrolled.",
            ),
            request_id=request_id,
        )
        return JSONResponse(status_code=200, content=response.model_dump())

    # 8. Persist the new face
    face = await db.create_face(
        user_id=user.id,
        embedding=embedding.tolist(),
        image_hash=image_hash,
        quality_score=quality.overall_score,
        blur_score=quality.blur_score,
        brightness=quality.brightness,
        face_confidence=quality.face_confidence,
        face_width_px=quality.face_width_px,
        face_height_px=quality.face_height_px,
        pitch_deg=quality.pitch_deg,
        yaw_deg=quality.yaw_deg,
        roll_deg=quality.roll_deg,
        label=body.label,
    )
    await session.commit()

    face_count = await db.get_face_count_for_user(user.id)

    ENROLLMENT_COUNTER.labels(result="success").inc()

    quality_metrics = QualityMetricsResponse(
        overall_score=quality.overall_score,
        blur_score=quality.blur_score,
        brightness=quality.brightness,
        face_confidence=quality.face_confidence,
        face_size=FaceSizeResponse(
            width_px=quality.face_width_px,
            height_px=quality.face_height_px,
        ),
        head_pose=HeadPoseResponse(
            pitch_deg=quality.pitch_deg,
            yaw_deg=quality.yaw_deg,
            roll_deg=quality.roll_deg,
        ),
    )
    bbox = result.face.bbox
    bounding_box = BoundingBoxResponse(
        x=bbox[0],
        y=bbox[1],
        width=bbox[2] - bbox[0],
        height=bbox[3] - bbox[1],
    )

    response = ApiResponse.ok(
        EnrollFaceResponse(
            face_id=str(face.id),
            user_id=user_id,
            label=face.label,
            quality_metrics=quality_metrics,
            bounding_box=bounding_box,
            enrolled_at=face.enrolled_at.isoformat(),
            total_faces_for_user=face_count,
            duplicate=False,
            duplicate_of_face_id=None,
            similarity=None,
            message=None,
        ),
        request_id=request_id,
    )
    return JSONResponse(status_code=201, content=response.model_dump())


@router.get("/v1/users/{user_id}/faces")
async def list_faces(
    user_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    """List all enrolled faces for a user."""
    request_id = _get_request_id(request)
    db = DatabaseService(session)

    user = await db.get_user_by_external_id(user_id)
    if user is None:
        response = ApiResponse.fail(
            code="USER_NOT_FOUND",
            message=f"No user found with external_id '{user_id}'.",
            request_id=request_id,
        )
        return JSONResponse(status_code=404, content=response.model_dump())

    faces = await db.get_faces_for_user(user.id)
    face_items = [
        FaceListItem(
            face_id=str(f.id),
            label=f.label,
            quality_score=f.quality_score,
            face_size=FaceSizeResponse(
                width_px=f.face_width_px,
                height_px=f.face_height_px,
            ),
            enrolled_at=f.enrolled_at.isoformat(),
        )
        for f in faces
    ]

    response = ApiResponse.ok(
        ListFacesResponse(
            user_id=user_id,
            faces=face_items,
            total=len(face_items),
            max_allowed=settings.max_faces_per_user,
        ),
        request_id=request_id,
    )
    return JSONResponse(status_code=200, content=response.model_dump())


@router.delete("/v1/users/{user_id}/faces/{face_id}")
async def delete_face(
    user_id: str,
    face_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """Delete a specific enrolled face."""
    import uuid as _uuid

    request_id = _get_request_id(request)
    db = DatabaseService(session)

    user = await db.get_user_by_external_id(user_id)
    if user is None:
        response = ApiResponse.fail(
            code="USER_NOT_FOUND",
            message=f"No user found with external_id '{user_id}'.",
            request_id=request_id,
        )
        return JSONResponse(status_code=404, content=response.model_dump())

    # Validate face_id is a parseable UUID before querying
    try:
        face_uuid = _uuid.UUID(face_id)
    except ValueError:
        response = ApiResponse.fail(
            code="FACE_NOT_FOUND",
            message=f"No face found with id '{face_id}' for user '{user_id}'.",
            request_id=request_id,
        )
        return JSONResponse(status_code=404, content=response.model_dump())

    face = await db.get_face_by_id_for_user(face_uuid, user.id)
    if face is None:
        response = ApiResponse.fail(
            code="FACE_NOT_FOUND",
            message=f"No face found with id '{face_id}' for user '{user_id}'.",
            request_id=request_id,
        )
        return JSONResponse(status_code=404, content=response.model_dump())

    await db.delete_face(face)
    await session.commit()

    remaining_faces = await db.get_face_count_for_user(user.id)

    response = ApiResponse.ok(
        DeleteFaceResponse(
            deleted_face_id=face_id,
            user_id=user_id,
            remaining_faces=remaining_faces,
            deleted_at=datetime.now(timezone.utc).isoformat(),
        ),
        request_id=request_id,
    )
    return JSONResponse(status_code=200, content=response.model_dump())
