import base64
import hashlib
import re
import cv2
import numpy as np
from PIL import Image, ExifTags
import io


class InvalidImageError(Exception):
    """Raised when image cannot be decoded or is invalid."""
    pass


# EXIF Orientation tag number (tag 274 per TIFF/EXIF spec)
_ORIENTATION_TAG = 274

# Map EXIF orientation values to (cv2.ROTATE_* constant or None, flip_code or None)
# EXIF orientation values:
#   1 = Normal
#   2 = Flip horizontal
#   3 = Rotate 180
#   4 = Flip vertical
#   5 = Transpose (rotate 90 CCW + flip horizontal)
#   6 = Rotate 90 CW
#   7 = Transverse (rotate 90 CW + flip horizontal)
#   8 = Rotate 90 CCW
_EXIF_ORIENTATION_TRANSFORMS = {
    1: None,                                        # Normal — no-op
    2: ("flip", 1),                                 # Flip horizontal
    3: ("rotate", cv2.ROTATE_180),                  # Rotate 180
    4: ("flip", 0),                                 # Flip vertical
    5: ("transpose", None),                         # Rotate 90 CCW then flip horizontal
    6: ("rotate", cv2.ROTATE_90_CLOCKWISE),         # Rotate 90 CW
    7: ("transverse", None),                        # Rotate 90 CW then flip horizontal
    8: ("rotate", cv2.ROTATE_90_COUNTERCLOCKWISE),  # Rotate 90 CCW
}


def decode_image(image_input: str | bytes) -> np.ndarray:
    """
    Accept base64 string (with or without data URI prefix) or raw bytes.
    Return BGR numpy array (OpenCV convention).
    Raise InvalidImageError on failure.

    Steps:
    1. If string: strip "data:image/...;base64," prefix if present, then base64-decode.
    2. If already bytes: treat as raw image bytes.
    3. Validate decoded size <= 10 MB (10 * 1024 * 1024 bytes).
    4. np.frombuffer() -> np.uint8 array.
    5. cv2.imdecode(buf, cv2.IMREAD_COLOR) -> BGR ndarray.
    6. If result is None: raise InvalidImageError.
    7. EXIF orientation correction using PIL.
    8. Return BGR ndarray.
    """
    MAX_BYTES = 10 * 1024 * 1024  # 10 MB

    if isinstance(image_input, str):
        # Strip data URI prefix if present: "data:image/jpeg;base64,<data>"
        match = re.match(r"data:image/[^;]+;base64,", image_input)
        if match:
            image_input = image_input[match.end():]

        # Remove any whitespace that might have crept in (line breaks etc.)
        image_input = image_input.strip()

        try:
            raw_bytes = base64.b64decode(image_input)
        except Exception as exc:
            raise InvalidImageError(f"Failed to base64-decode image string: {exc}") from exc

    elif isinstance(image_input, (bytes, bytearray)):
        raw_bytes = bytes(image_input)
    else:
        raise InvalidImageError(
            f"image_input must be str (base64) or bytes, got {type(image_input).__name__}"
        )

    if len(raw_bytes) > MAX_BYTES:
        raise InvalidImageError(
            f"Image exceeds maximum allowed size of 10 MB "
            f"(received {len(raw_bytes)} bytes)"
        )

    if len(raw_bytes) == 0:
        raise InvalidImageError("Image data is empty.")

    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr_image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if bgr_image is None:
        raise InvalidImageError(
            "cv2.imdecode returned None — image data is corrupt or format is unsupported."
        )

    # Apply EXIF orientation correction
    bgr_image = _correct_exif_orientation(raw_bytes, bgr_image)

    return bgr_image


def _correct_exif_orientation(image_bytes: bytes, bgr_image: np.ndarray) -> np.ndarray:
    """
    Read EXIF orientation from image bytes and rotate BGR image accordingly.
    Return corrected BGR numpy array.
    If no EXIF or error, return original image unchanged.
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        exif_data = pil_image._getexif()  # type: ignore[attr-defined]

        if exif_data is None:
            return bgr_image

        orientation = exif_data.get(_ORIENTATION_TAG)
        if orientation is None or orientation not in _EXIF_ORIENTATION_TRANSFORMS:
            return bgr_image

        transform = _EXIF_ORIENTATION_TRANSFORMS[orientation]
        if transform is None:
            # Orientation 1 — no correction needed
            return bgr_image

        kind, param = transform

        if kind == "rotate":
            return cv2.rotate(bgr_image, param)

        elif kind == "flip":
            # param: 0 = vertical flip, 1 = horizontal flip
            return cv2.flip(bgr_image, param)

        elif kind == "transpose":
            # Rotate 90 CCW then flip horizontally (EXIF 5)
            rotated = cv2.rotate(bgr_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return cv2.flip(rotated, 1)

        elif kind == "transverse":
            # Rotate 90 CW then flip horizontally (EXIF 7)
            rotated = cv2.rotate(bgr_image, cv2.ROTATE_90_CLOCKWISE)
            return cv2.flip(rotated, 1)

        return bgr_image

    except Exception:
        # Any EXIF parsing failure is non-fatal; return the image as-is
        return bgr_image


def compute_image_hash(image_bytes: bytes) -> str:
    """Return SHA256 hex digest of raw image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()


def crop_face(image_bgr: np.ndarray, bbox: np.ndarray, margin: float = 0.1) -> np.ndarray:
    """
    Crop face region from image with optional margin.
    bbox is [x1, y1, x2, y2] float array from InsightFace.
    Clamp to image bounds. Return cropped BGR region.
    """
    h, w = image_bgr.shape[:2]

    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    face_w = x2 - x1
    face_h = y2 - y1

    # Add margin proportional to face dimensions
    dx = face_w * margin
    dy = face_h * margin

    x1 = max(0.0, x1 - dx)
    y1 = max(0.0, y1 - dy)
    x2 = min(float(w), x2 + dx)
    y2 = min(float(h), y2 + dy)

    # Convert to integer pixel indices
    ix1 = int(round(x1))
    iy1 = int(round(y1))
    ix2 = int(round(x2))
    iy2 = int(round(y2))

    # Ensure at least a 1x1 crop even after clamping
    ix2 = max(ix2, ix1 + 1)
    iy2 = max(iy2, iy1 + 1)

    return image_bgr[iy1:iy2, ix1:ix2]
