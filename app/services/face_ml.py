import dataclasses
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from app.config import Settings
from app.utils.image import crop_face, InvalidImageError


@dataclasses.dataclass
class QualityMetrics:
    overall_score: float
    blur_score: float
    brightness: float
    face_confidence: float
    face_width_px: int
    face_height_px: int
    pitch_deg: float
    yaw_deg: float
    roll_deg: float


@dataclasses.dataclass
class FaceDetectionResult:
    """Result of detect_and_embed for a single detected face."""
    embedding: np.ndarray          # L2-normalized 512-dim float32
    quality: QualityMetrics
    bbox: list[float]              # [x1, y1, x2, y2]
    landmarks: list[list[float]]   # 5-point kps


@dataclasses.dataclass
class MLProcessResult:
    """Full result of process_image — always 1 face or raises."""
    face: FaceDetectionResult


@dataclasses.dataclass
class DuplicateCheckResult:
    is_duplicate: bool
    existing_face_id: str | None = None
    similarity: float | None = None


# Quality component weights — must sum to 1.0
_QUALITY_WEIGHTS = {
    "blur": 0.30,
    "brightness": 0.20,
    "face_confidence": 0.25,
    "face_size": 0.15,
    "head_pose": 0.10,
}


class FaceMLService:
    """
    Wraps InsightFace FaceAnalysis. Initialized once at startup.
    The buffalo_l model pack provides:
    - RetinaFace detection (det_10g.onnx)
    - ArcFace embedding (w600k_r50.onnx) -> 512-dim float32
    - Pose estimation
    """

    def __init__(self, settings: Settings):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if settings.ml_device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._app = FaceAnalysis(
            name=settings.insightface_model_name,
            root=settings.insightface_model_dir,
            providers=providers,
        )
        ctx_id = 0 if settings.ml_device == "cuda" else -1
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_image(self, image_bgr: np.ndarray) -> tuple[list, MLProcessResult | None]:
        """
        Run full face detection + embedding pipeline.
        Returns (raw_faces, result_or_none).
        raw_faces is the list returned by insightface (length 0, 1, or 2+).
        result_or_none is None if 0 or 2+ faces (caller handles errors).
        If exactly 1 face: return (faces, MLProcessResult).
        """
        faces = self._app.get(image_bgr)

        if len(faces) != 1:
            return faces, None

        face = faces[0]

        quality = self.compute_quality(image_bgr, face)
        embedding = self.get_embedding(face)

        bbox: list[float] = face.bbox.tolist()
        landmarks: list[list[float]] = face.kps.tolist() if face.kps is not None else []

        detection_result = FaceDetectionResult(
            embedding=embedding,
            quality=quality,
            bbox=bbox,
            landmarks=landmarks,
        )

        return faces, MLProcessResult(face=detection_result)

    def compute_quality(self, image_bgr: np.ndarray, face) -> QualityMetrics:
        """
        Compute quality score from image + InsightFace face object.

        QUALITY_WEIGHTS:
          blur: 0.30
          brightness: 0.20
          face_confidence: 0.25
          face_size: 0.15
          head_pose: 0.10
        """
        # --- Face crop -------------------------------------------------------
        face_crop = crop_face(image_bgr, face.bbox, margin=0.1)

        face_h, face_w = face_crop.shape[:2]

        # --- Blur (Laplacian variance on grayscale crop) ---------------------
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = float(min(lap_var / 500.0, 1.0))

        # --- Brightness ------------------------------------------------------
        # Work in float [0, 1] range
        mean_pixel = float(face_crop.mean()) / 255.0
        if 0.3 <= mean_pixel <= 0.8:
            brightness_score = 1.0
        elif mean_pixel < 0.3:
            brightness_score = mean_pixel / 0.3
        else:
            # mean_pixel > 0.8
            brightness_score = (1.0 - mean_pixel) / 0.2
        brightness_score = max(0.0, float(brightness_score))

        # --- Face confidence -------------------------------------------------
        face_confidence = float(face.det_score)

        # --- Face size -------------------------------------------------------
        face_size_score = float(min(min(face_w, face_h) / 400.0, 1.0))

        # --- Head pose -------------------------------------------------------
        # face.pose is [pitch, yaw, roll] as a numpy array (degrees)
        if face.pose is not None:
            pose = face.pose
            pitch_deg = float(pose[0])
            yaw_deg = float(pose[1])
            roll_deg = float(pose[2])
        else:
            # If pose estimation is unavailable, assume frontal
            pitch_deg = 0.0
            yaw_deg = 0.0
            roll_deg = 0.0

        pitch_ok = 1.0 - min(abs(pitch_deg) / self._settings.max_pitch_deg, 1.0)
        yaw_ok = 1.0 - min(abs(yaw_deg) / self._settings.max_yaw_deg, 1.0)
        roll_ok = 1.0 - min(abs(roll_deg) / self._settings.max_roll_deg, 1.0)
        pose_score = (pitch_ok + yaw_ok + roll_ok) / 3.0

        # --- Overall weighted score ------------------------------------------
        overall_score = (
            _QUALITY_WEIGHTS["blur"] * blur_score
            + _QUALITY_WEIGHTS["brightness"] * brightness_score
            + _QUALITY_WEIGHTS["face_confidence"] * face_confidence
            + _QUALITY_WEIGHTS["face_size"] * face_size_score
            + _QUALITY_WEIGHTS["head_pose"] * pose_score
        )

        return QualityMetrics(
            overall_score=float(overall_score),
            blur_score=blur_score,
            brightness=brightness_score,
            face_confidence=face_confidence,
            face_width_px=face_w,
            face_height_px=face_h,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            roll_deg=roll_deg,
        )

    def check_quality_gate(self, quality: QualityMetrics, threshold: float) -> list[dict]:
        """
        Return list of failing check dicts. Empty list = all pass.

        Hard rules (checked first, regardless of overall score):
          - face_width_px  < settings.min_face_size_px
          - face_height_px < settings.min_face_size_px
          - abs(yaw_deg)   > settings.max_yaw_deg
          - abs(pitch_deg) > settings.max_pitch_deg
          - abs(roll_deg)  > settings.max_roll_deg

        Soft rule: overall_score < threshold

        Each failing check dict: {"check": str, "value": float, "minimum": float, "reason": str}
        """
        failures: list[dict] = []
        s = self._settings

        # Hard rule: face width
        if quality.face_width_px < s.min_face_size_px:
            failures.append({
                "check": "face_width_px",
                "value": float(quality.face_width_px),
                "minimum": float(s.min_face_size_px),
                "reason": (
                    f"Face width {quality.face_width_px}px is below the minimum "
                    f"required {s.min_face_size_px}px."
                ),
            })

        # Hard rule: face height
        if quality.face_height_px < s.min_face_size_px:
            failures.append({
                "check": "face_height_px",
                "value": float(quality.face_height_px),
                "minimum": float(s.min_face_size_px),
                "reason": (
                    f"Face height {quality.face_height_px}px is below the minimum "
                    f"required {s.min_face_size_px}px."
                ),
            })

        # Hard rule: yaw
        if abs(quality.yaw_deg) > s.max_yaw_deg:
            failures.append({
                "check": "yaw_deg",
                "value": float(abs(quality.yaw_deg)),
                "minimum": float(s.max_yaw_deg),
                "reason": (
                    f"Head yaw {quality.yaw_deg:.1f}° exceeds the maximum allowed "
                    f"±{s.max_yaw_deg}°."
                ),
            })

        # Hard rule: pitch
        if abs(quality.pitch_deg) > s.max_pitch_deg:
            failures.append({
                "check": "pitch_deg",
                "value": float(abs(quality.pitch_deg)),
                "minimum": float(s.max_pitch_deg),
                "reason": (
                    f"Head pitch {quality.pitch_deg:.1f}° exceeds the maximum allowed "
                    f"±{s.max_pitch_deg}°."
                ),
            })

        # Hard rule: roll
        if abs(quality.roll_deg) > s.max_roll_deg:
            failures.append({
                "check": "roll_deg",
                "value": float(abs(quality.roll_deg)),
                "minimum": float(s.max_roll_deg),
                "reason": (
                    f"Head roll {quality.roll_deg:.1f}° exceeds the maximum allowed "
                    f"±{s.max_roll_deg}°."
                ),
            })

        # Soft rule: overall quality score
        if quality.overall_score < threshold:
            failures.append({
                "check": "overall_score",
                "value": float(quality.overall_score),
                "minimum": float(threshold),
                "reason": (
                    f"Overall quality score {quality.overall_score:.3f} is below the "
                    f"required threshold {threshold:.3f}."
                ),
            })

        return failures

    def get_embedding(self, face) -> np.ndarray:
        """
        Return L2-normalized 512-dim float32 embedding from InsightFace face object.
        face.embedding is already the embedding vector.
        Normalize: emb / ||emb|| (L2 norm). If norm == 0, return as-is.
        """
        emb = face.embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity for L2-normalized vectors.
        For unit vectors: cosine_similarity = dot(a, b).
        Returns float in [-1.0, 1.0].
        """
        return float(np.dot(a, b))

    def check_duplicate(
        self,
        new_embedding: np.ndarray,
        stored_embeddings: list[tuple[str, np.ndarray]],
        threshold: float,
    ) -> DuplicateCheckResult:
        """
        Check if new_embedding is too similar to any stored embedding.
        stored_embeddings: list of (face_id_str, embedding_array) tuples.
        threshold: cosine similarity above which = duplicate.
        Returns DuplicateCheckResult.
        """
        if not stored_embeddings:
            return DuplicateCheckResult(is_duplicate=False)

        similarities = [
            (fid, self.cosine_similarity(new_embedding, emb))
            for fid, emb in stored_embeddings
        ]
        best_fid, best_sim = max(similarities, key=lambda x: x[1])

        if best_sim >= threshold:
            return DuplicateCheckResult(
                is_duplicate=True,
                existing_face_id=best_fid,
                similarity=best_sim,
            )
        return DuplicateCheckResult(is_duplicate=False)
