"""Async database service providing all CRUD operations for the face recognition API."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.api_key import ApiKey
from app.models.face import Face
from app.models.user import User


def _utcnow() -> datetime:
    """Return the current UTC time as a naive datetime (matches TIMESTAMP WITHOUT TIME ZONE columns)."""
    return datetime.utcnow()


class DatabaseService:
    """Thin async repository layer wrapping an SQLAlchemy AsyncSession."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # API Keys
    # ------------------------------------------------------------------

    async def get_api_key_by_prefix(self, prefix: str) -> ApiKey | None:
        """Fetch the first active ApiKey whose key_prefix matches *prefix*."""
        stmt = select(ApiKey).where(
            ApiKey.key_prefix == prefix,
            ApiKey.is_active.is_(True),
        )
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def update_api_key_last_used(self, api_key_id: uuid.UUID) -> None:
        """Stamp *last_used_at* on the ApiKey identified by *api_key_id*."""
        stmt = select(ApiKey).where(ApiKey.id == api_key_id)
        result = await self._session.execute(stmt)
        api_key = result.scalars().first()
        if api_key is not None:
            api_key.last_used_at = _utcnow()
            await self._session.flush()

    async def create_api_key(
        self,
        key_hash: str,
        key_prefix: str,
        name: str,
        rate_limit: int = 100,
    ) -> ApiKey:
        """Persist a new ApiKey and return it."""
        api_key = ApiKey(
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=name,
            rate_limit=rate_limit,
            is_active=True,
            created_at=_utcnow(),
        )
        self._session.add(api_key)
        await self._session.flush()
        await self._session.refresh(api_key)
        return api_key

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    async def create_user(
        self,
        external_id: str,
        display_name: str | None,
        metadata: dict,
    ) -> User:
        """Persist a new User and return it."""
        now = _utcnow()
        user = User(
            external_id=external_id,
            display_name=display_name,
            metadata_=metadata,
            created_at=now,
            updated_at=now,
        )
        self._session.add(user)
        await self._session.flush()
        await self._session.refresh(user)
        return user

    async def get_user_by_external_id(self, external_id: str) -> User | None:
        """Return the User with the given *external_id*, or None."""
        stmt = select(User).where(User.external_id == external_id)
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def delete_user(self, user: User) -> int:
        """Delete *user* and all its faces via cascade.

        Returns the count of Face rows deleted before the user row itself is
        removed.
        """
        # Count faces first so the caller can log / return the figure.
        face_count_stmt = select(func.count()).where(Face.user_id == user.id)
        count_result = await self._session.execute(face_count_stmt)
        face_count: int = count_result.scalar_one()

        # Explicitly delete faces (cascade would also handle this, but being
        # explicit avoids relying on DB-level cascade being set up correctly in
        # all test environments).
        delete_faces_stmt = delete(Face).where(Face.user_id == user.id)
        await self._session.execute(delete_faces_stmt)

        await self._session.delete(user)
        await self._session.flush()
        return face_count

    # ------------------------------------------------------------------
    # Faces
    # ------------------------------------------------------------------

    async def get_face_count_for_user(self, user_id: uuid.UUID) -> int:
        """Return the number of Face rows belonging to *user_id*."""
        stmt = select(func.count()).where(Face.user_id == user_id)
        result = await self._session.execute(stmt)
        return result.scalar_one()

    async def get_face_by_hash_for_user(
        self, user_id: uuid.UUID, image_hash: str
    ) -> Face | None:
        """Return the Face with the given *image_hash* for *user_id*, or None."""
        stmt = select(Face).where(
            Face.user_id == user_id,
            Face.image_hash == image_hash,
        )
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def create_face(
        self,
        user_id: uuid.UUID,
        embedding: list[float],
        image_hash: str,
        quality_score: float,
        blur_score: float,
        brightness: float,
        face_confidence: float,
        face_width_px: int,
        face_height_px: int,
        pitch_deg: float,
        yaw_deg: float,
        roll_deg: float,
        label: str | None,
    ) -> Face:
        """Persist a new Face and return it."""
        face = Face(
            user_id=user_id,
            embedding=embedding,
            image_hash=image_hash,
            quality_score=quality_score,
            blur_score=blur_score,
            brightness=brightness,
            face_confidence=face_confidence,
            face_width_px=face_width_px,
            face_height_px=face_height_px,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            roll_deg=roll_deg,
            label=label,
            enrolled_at=_utcnow(),
        )
        self._session.add(face)
        await self._session.flush()
        await self._session.refresh(face)
        return face

    async def get_faces_for_user(self, user_id: uuid.UUID) -> list[Face]:
        """Return all Face rows for *user_id* (embeddings included)."""
        stmt = (
            select(Face)
            .where(Face.user_id == user_id)
            .order_by(Face.enrolled_at)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_faces_with_embeddings_for_user(self, user_id: uuid.UUID) -> list[Face]:
        """Return all Face rows for *user_id* with the embedding field populated.

        The embedding column is always loaded by SQLAlchemy; this method is an
        explicit alias that makes the intent clear at the call-site.
        """
        stmt = (
            select(Face)
            .where(Face.user_id == user_id)
            .options(selectinload(Face.user))
            .order_by(Face.enrolled_at)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_face_by_id_for_user(
        self, face_id: uuid.UUID, user_id: uuid.UUID
    ) -> Face | None:
        """Return the Face with *face_id* belonging to *user_id*, or None."""
        stmt = select(Face).where(
            Face.id == face_id,
            Face.user_id == user_id,
        )
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def delete_face(self, face: Face) -> None:
        """Delete a single Face row."""
        await self._session.delete(face)
        await self._session.flush()

    async def get_user_face_count(self, user_id: uuid.UUID) -> int:
        """Alias for :meth:`get_face_count_for_user`."""
        return await self.get_face_count_for_user(user_id)
