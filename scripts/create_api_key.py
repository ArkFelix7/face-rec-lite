#!/usr/bin/env python3
"""Create a new API key and store it in the database.

Usage::

    python scripts/create_api_key.py --name "my-service" [--rate-limit 200]

The generated raw key is printed exactly once and is never stored. Save it
immediately — it cannot be recovered afterwards because only its bcrypt hash
is persisted in the database.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import sys

# Ensure the project root is on the Python path when the script is executed
# directly (i.e. ``python scripts/create_api_key.py …``).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bcrypt as _bcrypt  # noqa: E402  (after sys.path insert)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402

from app.config import settings  # noqa: E402
from app.models.api_key import ApiKey  # noqa: E402
from app.services.database import DatabaseService  # noqa: E402


async def create_key(name: str, rate_limit: int = 100) -> str:
    """Generate, hash, and persist a new API key.

    Args:
        name: Human-readable label for the key (stored in the database).
        rate_limit: Maximum requests per minute for this key (default 100).

    Returns:
        The raw API key string (``sk_live_<48 hex chars>``).
    """
    # Key format: "sk_live_<48 hex chars>"
    # key_prefix = raw_key[8:16]: first 8 chars of the random portion, unique per key.
    # Using raw_key[:8] would give "sk_live_" for every key, defeating prefix-based lookup.
    raw_key = "sk_live_" + secrets.token_hex(24)
    key_prefix = raw_key[8:16]  # First 8 chars of the random portion — unique per key

    # Hash with the configured bcrypt work factor.
    key_hash = _bcrypt.hashpw(
        raw_key.encode("utf-8"),
        _bcrypt.gensalt(rounds=settings.bcrypt_rounds),
    ).decode("utf-8")

    engine = create_async_engine(settings.database_url)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with factory() as session:
            db = DatabaseService(session)
            api_key = await db.create_api_key(
                key_hash=key_hash,
                key_prefix=key_prefix,
                name=name,
                rate_limit=rate_limit,
            )
            await session.commit()

            print(f"\nCreated API key:")
            print(f"  Name       : {api_key.name}")
            print(f"  ID         : {api_key.id}")
            print(f"  Key prefix : {api_key.key_prefix}")
            print(f"  Rate limit : {api_key.rate_limit} req/min")
            print(f"\nYour API key (save this — shown once):\n  {raw_key}\n")
    finally:
        await engine.dispose()

    return raw_key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a new API key for the face recognition API."
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Human-readable label for this API key (e.g. 'my-service')",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=100,
        metavar="RPM",
        help="Maximum requests per minute (default: 100)",
    )
    args = parser.parse_args()

    asyncio.run(create_key(args.name, args.rate_limit))


if __name__ == "__main__":
    main()
