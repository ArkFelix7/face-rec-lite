"""User management endpoints: create, get, delete."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db_session
from app.schemas.common import ApiResponse
from app.schemas.user import CreateUserRequest, DeleteUserResponse, UserResponse
from app.services.database import DatabaseService

router = APIRouter(tags=["users"])


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or ""


@router.post("/v1/users", status_code=201)
async def create_user(
    body: CreateUserRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """Register a new user.

    Returns 409 if a user with the same external_id already exists.
    """
    request_id = _get_request_id(request)
    db = DatabaseService(session)

    existing = await db.get_user_by_external_id(body.external_id)
    if existing is not None:
        response = ApiResponse.fail(
            code="USER_ALREADY_EXISTS",
            message=f"A user with external_id '{body.external_id}' already exists.",
            request_id=request_id,
        )
        return JSONResponse(status_code=409, content=response.model_dump())

    user = await db.create_user(body.external_id, body.display_name, body.metadata)
    await session.commit()

    updated_at = user.updated_at.isoformat() if user.updated_at else None
    response = ApiResponse.ok(
        UserResponse(
            id=str(user.id),
            external_id=user.external_id,
            display_name=user.display_name,
            metadata=user.metadata_,
            face_count=0,
            created_at=user.created_at.isoformat(),
            updated_at=updated_at,
        ),
        request_id=request_id,
    )
    return JSONResponse(status_code=201, content=response.model_dump())


@router.get("/v1/users/{user_id}")
async def get_user(
    user_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """Retrieve a user by external_id."""
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

    face_count = await db.get_face_count_for_user(user.id)
    updated_at = user.updated_at.isoformat() if user.updated_at else None
    response = ApiResponse.ok(
        UserResponse(
            id=str(user.id),
            external_id=user.external_id,
            display_name=user.display_name,
            metadata=user.metadata_,
            face_count=face_count,
            created_at=user.created_at.isoformat(),
            updated_at=updated_at,
        ),
        request_id=request_id,
    )
    return JSONResponse(status_code=200, content=response.model_dump())


@router.delete("/v1/users/{user_id}")
async def delete_user(
    user_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """Delete a user and all their enrolled faces."""
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

    faces_deleted = await db.delete_user(user)
    await session.commit()

    response = ApiResponse.ok(
        DeleteUserResponse(
            deleted_user_id=user_id,
            faces_deleted=faces_deleted,
            deleted_at=datetime.now(timezone.utc).isoformat(),
        ),
        request_id=request_id,
    )
    return JSONResponse(status_code=200, content=response.model_dump())
