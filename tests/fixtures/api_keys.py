"""
Fixture helpers for API key creation in tests.

These are factory functions that create in-memory ApiKey ORM objects
without persisting them to the database. The caller is responsible for
adding them to a session and flushing.
"""

from __future__ import annotations

import secrets

from passlib.hash import bcrypt

from app.models.api_key import ApiKey


def make_test_api_key(
    name: str = "test",
    rate_limit: int = 1000,
    rounds: int = 4,
    is_active: bool = True,
) -> tuple[str, ApiKey]:
    """Create an ApiKey ORM object (not persisted). Returns (raw_key, record).

    Args:
        name: Human-readable name for the API key.
        rate_limit: Requests per minute allowed for this key.
        rounds: bcrypt rounds (use low value like 4 for tests to keep them fast).
        is_active: Whether the key should be active.

    Returns:
        A tuple of (raw_key, ApiKey) where raw_key is the plain-text key
        (e.g. "sk_live_...") and ApiKey is the unsaved ORM object with
        the bcrypt hash stored.

    Example::

        raw_key, record = make_test_api_key()
        session.add(record)
        await session.flush()
        headers = {"Authorization": f"Bearer {raw_key}"}
    """
    # Key format must match the production format: "sk_live_<48 hex chars>".
    # Auth middleware extracts raw_key[8:16] as the lookup prefix.
    raw_key = "sk_live_" + secrets.token_hex(24)
    key_prefix = raw_key[8:16]  # First 8 chars of the random portion — unique per key
    return raw_key, ApiKey(
        key_hash=bcrypt.hash(raw_key, rounds=rounds),
        key_prefix=key_prefix,
        name=name,
        is_active=is_active,
        rate_limit=rate_limit,
    )


def make_inactive_api_key(
    name: str = "inactive-test",
    rounds: int = 4,
) -> tuple[str, ApiKey]:
    """Create an inactive ApiKey for testing auth rejection."""
    raw_key, record = make_test_api_key(name=name, rounds=rounds, is_active=False)
    return raw_key, record
