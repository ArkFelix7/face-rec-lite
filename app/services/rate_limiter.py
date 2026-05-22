"""Redis-backed fixed-window rate limiter (per minute per API key)."""

from __future__ import annotations

import time

from redis.asyncio import Redis


class RateLimiter:
    """Redis-backed fixed-window rate limiter (per minute per API key)."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def check_rate_limit(
        self, api_key_id: str, limit: int
    ) -> tuple[bool, int, int]:
        """Check if a request is within the rate limit.

        Algorithm (fixed window per minute):
        1. key = ``ratelimit:{api_key_id}:{current_minute}``
           where ``current_minute = int(time.time()) // 60``
        2. Atomically increment the counter for the current window.
        3. On first increment, set a 60-second TTL so keys self-expire.
        4. Return ``(allowed, count, limit)`` where ``allowed = count <= limit``.

        Args:
            api_key_id: String representation of the API key's UUID.
            limit: Maximum requests allowed per minute.

        Returns:
            A 3-tuple ``(allowed, current_count, limit)``.
        """
        current_minute = int(time.time()) // 60
        key = f"ratelimit:{api_key_id}:{current_minute}"

        count = await self._redis.incr(key)
        if count == 1:
            await self._redis.expire(key, 60)

        allowed = count <= limit
        return allowed, int(count), limit

    async def ping(self) -> bool:
        """Return True if Redis is reachable, False otherwise."""
        try:
            await self._redis.ping()
            return True
        except Exception:
            return False
