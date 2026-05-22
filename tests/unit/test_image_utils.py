"""Unit tests for app/utils/image.py — no DB, no ML, no network."""

from __future__ import annotations

import base64
import hashlib
import io

import numpy as np
import pytest
from PIL import Image

from app.utils.image import InvalidImageError, compute_image_hash, crop_face, decode_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_jpeg_bytes(width: int = 200, height: int = 200) -> bytes:
    img = Image.new("RGB", (width, height), color=(120, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def to_b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


# ---------------------------------------------------------------------------
# TestDecodeImage
# ---------------------------------------------------------------------------


class TestDecodeImage:
    def test_decode_base64_without_prefix(self):
        raw = make_jpeg_bytes()
        result = decode_image(to_b64(raw))
        assert result is not None
        assert result.ndim == 3
        assert result.shape[2] == 3  # BGR

    def test_decode_base64_with_data_uri_prefix(self):
        raw = make_jpeg_bytes()
        b64 = "data:image/jpeg;base64," + to_b64(raw)
        result = decode_image(b64)
        assert result is not None
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_decode_raw_bytes(self):
        raw = make_jpeg_bytes()
        result = decode_image(raw)
        assert result is not None
        assert result.ndim == 3

    def test_decode_corrupt_bytes_raises(self):
        with pytest.raises(InvalidImageError):
            decode_image(b"not-an-image-at-all")

    def test_decode_corrupt_base64_raises(self):
        # "this is not an image" — valid base64 but not a valid image
        with pytest.raises(InvalidImageError):
            decode_image("dGhpcyBpcyBub3QgYW4gaW1hZ2U=")

    def test_decode_too_large_raises(self):
        # Create a fake base64 string whose decoded payload > 10 MB
        big = b"x" * (10 * 1024 * 1024 + 1)
        b64 = base64.b64encode(big).decode()
        with pytest.raises(InvalidImageError, match="too large|10 MB"):
            decode_image(b64)

    def test_decode_png(self):
        img = Image.new("RGB", (100, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = decode_image(base64.b64encode(buf.getvalue()).decode())
        assert result is not None
        assert result.ndim == 3

    def test_decode_data_uri_png_prefix(self):
        img = Image.new("RGB", (50, 50), color=(10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        result = decode_image(b64)
        assert result is not None

    def test_decode_returns_bgr_not_rgb(self):
        # A red-only image: R=255, G=0, B=0
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = decode_image(base64.b64encode(buf.getvalue()).decode())
        # In BGR the red channel is index 2
        # The blue channel (index 0) should be low, red channel (index 2) should be high
        assert result[5, 5, 2] > result[5, 5, 0]

    def test_decode_empty_bytes_raises(self):
        with pytest.raises(InvalidImageError):
            decode_image(b"")

    def test_decode_invalid_type_raises(self):
        with pytest.raises((InvalidImageError, TypeError)):
            decode_image(12345)  # type: ignore[arg-type]

    def test_decode_whitespace_stripped_from_b64(self):
        """Base64 strings with leading/trailing whitespace should still decode."""
        raw = make_jpeg_bytes()
        b64_with_spaces = "  " + to_b64(raw) + "  "
        result = decode_image(b64_with_spaces)
        assert result is not None


# ---------------------------------------------------------------------------
# TestComputeImageHash
# ---------------------------------------------------------------------------


class TestComputeImageHash:
    def test_hash_deterministic(self):
        raw = make_jpeg_bytes()
        assert compute_image_hash(raw) == compute_image_hash(raw)

    def test_hash_different_images(self):
        raw1 = make_jpeg_bytes(100, 100)
        raw2 = make_jpeg_bytes(200, 200)
        assert compute_image_hash(raw1) != compute_image_hash(raw2)

    def test_hash_is_sha256(self):
        raw = make_jpeg_bytes()
        expected = hashlib.sha256(raw).hexdigest()
        assert compute_image_hash(raw) == expected

    def test_hash_is_hex_string(self):
        raw = make_jpeg_bytes()
        h = compute_image_hash(raw)
        assert isinstance(h, str)
        assert len(h) == 64
        int(h, 16)  # Should not raise

    def test_hash_empty_bytes(self):
        # sha256 of empty bytes is well-defined
        expected = hashlib.sha256(b"").hexdigest()
        assert compute_image_hash(b"") == expected

    def test_hash_single_byte_change(self):
        raw = bytearray(make_jpeg_bytes())
        raw2 = bytearray(raw)
        raw2[0] ^= 0xFF  # Flip one byte
        assert compute_image_hash(bytes(raw)) != compute_image_hash(bytes(raw2))


# ---------------------------------------------------------------------------
# TestCropFace
# ---------------------------------------------------------------------------


class TestCropFace:
    def test_basic_crop(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = np.array([100.0, 80.0, 300.0, 320.0])
        result = crop_face(img, bbox, margin=0.0)
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    def test_crop_clamps_to_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = np.array([-50.0, -50.0, 200.0, 200.0])
        result = crop_face(img, bbox, margin=0.0)
        assert result.shape[0] <= 100
        assert result.shape[1] <= 100

    def test_crop_with_margin_is_larger(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = np.array([100.0, 80.0, 300.0, 280.0])
        no_margin = crop_face(img, bbox, margin=0.0)
        with_margin = crop_face(img, bbox, margin=0.2)
        assert with_margin.shape[0] >= no_margin.shape[0]
        assert with_margin.shape[1] >= no_margin.shape[1]

    def test_crop_preserves_channels(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = np.array([50.0, 50.0, 200.0, 200.0])
        result = crop_face(img, bbox, margin=0.0)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_crop_correct_dimensions_no_margin(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = np.array([100.0, 80.0, 300.0, 280.0])
        result = crop_face(img, bbox, margin=0.0)
        # With no margin: width should be roughly 200, height roughly 200
        assert result.shape[1] == pytest.approx(200, abs=2)
        assert result.shape[0] == pytest.approx(200, abs=2)

    def test_crop_at_image_boundary(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        bbox = np.array([90.0, 90.0, 110.0, 110.0])  # Extends outside
        result = crop_face(img, bbox, margin=0.0)
        # Should clamp without error
        assert result is not None
        assert result.shape[0] >= 1
        assert result.shape[1] >= 1

    def test_crop_full_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = np.array([0.0, 0.0, 100.0, 100.0])
        result = crop_face(img, bbox, margin=0.0)
        assert result.shape[:2] == (100, 100)

    def test_crop_preserves_pixel_values(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[20:40, 30:60] = 255  # White region
        bbox = np.array([30.0, 20.0, 60.0, 40.0])
        result = crop_face(img, bbox, margin=0.0)
        # All pixels in the crop should be 255
        assert result.min() == 255
