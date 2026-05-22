from __future__ import annotations

from pydantic import BaseModel, Field


class EnrollFaceRequest(BaseModel):
    model_config = {"frozen": False}

    image: str = Field(..., description="Base64-encoded image with optional data URI prefix")
    label: str | None = Field(None, min_length=1, max_length=64)
    quality_threshold: float | None = Field(None, ge=0.0, le=1.0)


class FaceSizeResponse(BaseModel):
    model_config = {"frozen": False}

    width_px: int
    height_px: int


class HeadPoseResponse(BaseModel):
    model_config = {"frozen": False}

    pitch_deg: float
    yaw_deg: float
    roll_deg: float


class BoundingBoxResponse(BaseModel):
    model_config = {"frozen": False}

    x: float
    y: float
    width: float
    height: float


class QualityMetricsResponse(BaseModel):
    model_config = {"frozen": False}

    overall_score: float
    blur_score: float
    brightness: float
    face_confidence: float
    face_size: FaceSizeResponse
    head_pose: HeadPoseResponse


class EnrollFaceResponse(BaseModel):
    model_config = {"frozen": False}

    face_id: str
    user_id: str
    label: str | None
    quality_metrics: QualityMetricsResponse
    bounding_box: BoundingBoxResponse
    enrolled_at: str
    total_faces_for_user: int
    duplicate: bool = False
    duplicate_of_face_id: str | None = None
    similarity: float | None = None
    message: str | None = None


class FaceListItem(BaseModel):
    model_config = {"frozen": False}

    face_id: str
    label: str | None
    quality_score: float
    face_size: FaceSizeResponse
    enrolled_at: str


class ListFacesResponse(BaseModel):
    model_config = {"frozen": False}

    user_id: str
    faces: list[FaceListItem]
    total: int
    max_allowed: int


class DeleteFaceResponse(BaseModel):
    model_config = {"frozen": False}

    deleted_face_id: str
    user_id: str
    remaining_faces: int
    deleted_at: str


class VerifyFaceRequest(BaseModel):
    model_config = {"frozen": False}

    image: str = Field(..., description="Base64-encoded image")
    threshold: float | None = Field(None, ge=0.0, le=1.0)


class FaceScoreItem(BaseModel):
    model_config = {"frozen": False}

    face_id: str
    similarity: float


class QueryFaceQuality(BaseModel):
    model_config = {"frozen": False}

    overall_score: float
    blur_score: float
    brightness: float
    face_confidence: float


class VerifyFaceResponse(BaseModel):
    model_config = {"frozen": False}

    match: bool
    confidence: float
    threshold_used: float
    best_matching_face_id: str
    user_id: str
    query_face_quality: QueryFaceQuality
    all_scores: list[FaceScoreItem]
    processing_time_ms: float


class FailingCheck(BaseModel):
    model_config = {"frozen": False}

    check: str
    value: float
    minimum: float
    reason: str


class LowQualityDetails(BaseModel):
    model_config = {"frozen": False}

    overall_score: float
    threshold: float
    failing_checks: list[FailingCheck]
