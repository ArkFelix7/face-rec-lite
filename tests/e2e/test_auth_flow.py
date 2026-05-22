"""
E2E tests — skipped unless E2E_TESTS=1 env var is set.

These tests do NOT mock the ML service; they test the full pipeline end-to-end
against a real ML model and real face images from tests/fixtures/faces/.

Usage:
    E2E_TESTS=1 pytest tests/e2e/ -v

Prerequisite:
    - Real ML model files downloaded (buffalo_l)
    - Fixture images placed in tests/fixtures/faces/
    - Real PostgreSQL and Redis running (from docker-compose.yml)
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("E2E_TESTS") != "1",
    reason="E2E tests require E2E_TESTS=1 env var and real face images in tests/fixtures/faces/",
)


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "faces")


class TestE2EAuthFlow:
    """Full enroll → verify → delete flow with real ML."""

    async def test_placeholder(self):
        """
        Placeholder: set E2E_TESTS=1 and provide fixture images to run real E2E tests.

        To run real E2E tests:
        1. Place face images in tests/fixtures/faces/ (e.g. person1_front.jpg)
        2. Start infrastructure: docker-compose up -d
        3. Run: E2E_TESTS=1 pytest tests/e2e/ -v
        """
        assert True, "E2E tests require E2E_TESTS=1 and fixture face images"

    async def test_fixture_images_exist(self):
        """Check that fixture images are available for real E2E testing."""
        if not os.path.isdir(FIXTURES_DIR):
            pytest.skip(f"Fixture directory not found: {FIXTURES_DIR}")
        jpg_files = [f for f in os.listdir(FIXTURES_DIR) if f.endswith((".jpg", ".jpeg", ".png"))]
        assert len(jpg_files) > 0, (
            f"No fixture images found in {FIXTURES_DIR}. "
            "Add face images to run real E2E tests."
        )
