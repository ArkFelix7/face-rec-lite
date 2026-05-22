from __future__ import annotations

import secrets
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


def generate_request_id() -> str:
    return "req_" + secrets.token_hex(6)


class ErrorDetail(BaseModel):
    model_config = {"frozen": False}

    code: str
    message: str
    details: dict[str, Any] | None = None


class ApiResponse(BaseModel, Generic[T]):
    """Standard response envelope for all endpoints."""

    model_config = {"frozen": False}

    success: bool
    data: T | None = None
    error: ErrorDetail | None = None
    request_id: str = Field(default_factory=generate_request_id)

    @classmethod
    def ok(cls, data: T, request_id: str | None = None) -> "ApiResponse[T]":
        return cls(
            success=True,
            data=data,
            error=None,
            request_id=request_id or generate_request_id(),
        )

    @classmethod
    def fail(
        cls,
        code: str,
        message: str,
        details: dict | None = None,
        request_id: str | None = None,
    ) -> "ApiResponse[None]":
        return cls(
            success=False,
            data=None,
            error=ErrorDetail(code=code, message=message, details=details),
            request_id=request_id or generate_request_id(),
        )


class HealthResponse(BaseModel):
    model_config = {"frozen": False}

    status: str
    version: str = "1.0.0"


class ReadyResponse(BaseModel):
    model_config = {"frozen": False}

    status: str
    checks: dict[str, str]
