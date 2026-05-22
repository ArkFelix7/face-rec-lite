from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateUserRequest(BaseModel):
    model_config = {"frozen": False}

    external_id: str = Field(..., min_length=1, max_length=255)
    display_name: str | None = Field(None, min_length=1, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserResponse(BaseModel):
    model_config = {"frozen": False}

    id: str  # UUID as string
    external_id: str
    display_name: str | None
    metadata: dict[str, Any]
    face_count: int
    created_at: str  # ISO datetime string
    updated_at: str | None = None


class DeleteUserResponse(BaseModel):
    model_config = {"frozen": False}

    deleted_user_id: str
    faces_deleted: int
    deleted_at: str  # ISO datetime string
