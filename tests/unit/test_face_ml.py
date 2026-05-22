"""Unit tests for FaceMLService — no DB, no image I/O, no real ML model loaded."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.config import Settings
from app.services.face_ml import DuplicateCheckResult, FaceMLService, QualityMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_settings(**kwargs) -> Settings:
    defaults = dict(
        database_url="postgresql+asyncpg://x:x@localhost/x",
        min_quality_score=0.5,
        min_face_size_px=80,
        max_pitch_deg=30.0,
        max_yaw_deg=35.0,
        max_roll_deg=25.0,
        default_verification_threshold=0.60,
        max_faces_per_user=5,
        dedup_threshold=0.95,
        ml_device="cpu",
    )
    defaults.update(kwargs)
    return Settings(**defaults)


def make_quality(**kwargs) -> QualityMetrics:
    defaults = dict(
        overall_score=0.85,
        blur_score=0.90,
        brightness=0.65,
        face_confidence=0.98,
        face_width_px=200,
        face_height_px=240,
        pitch_deg=2.0,
        yaw_deg=-3.0,
        roll_deg=1.0,
    )
    defaults.update(kwargs)
    return QualityMetrics(**defaults)


def unit_vec(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def service() -> FaceMLService:
    """FaceMLService with FaceAnalysis constructor patched out (no model download)."""
    with patch("app.services.face_ml.FaceAnalysis") as mock_fa:
        mock_fa.return_value.prepare = MagicMock()
        svc = FaceMLService(make_settings())
    return svc


# ---------------------------------------------------------------------------
# TestCosineSimilarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert FaceMLService.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert FaceMLService.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert FaceMLService.cosine_similarity(v, -v) == pytest.approx(-1.0)

    def test_512_dim_normalized_in_range(self):
        rng = np.random.RandomState(1)
        a = rng.randn(512).astype(np.float32)
        b = rng.randn(512).astype(np.float32)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        sim = FaceMLService.cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0

    def test_returns_float(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        result = FaceMLService.cosine_similarity(v, v)
        assert isinstance(result, float)

    def test_symmetry(self):
        a = unit_vec(10)
        b = unit_vec(20)
        assert FaceMLService.cosine_similarity(a, b) == pytest.approx(
            FaceMLService.cosine_similarity(b, a), abs=1e-6
        )

    def test_self_similarity_is_one(self):
        v = unit_vec(99)
        assert FaceMLService.cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# TestCheckQualityGate
# ---------------------------------------------------------------------------


class TestCheckQualityGate:
    def test_good_quality_passes(self, service):
        q = make_quality()
        failures = service.check_quality_gate(q, threshold=0.5)
        assert failures == []

    def test_overall_score_too_low(self, service):
        q = make_quality(overall_score=0.20)
        failures = service.check_quality_gate(q, threshold=0.5)
        codes = [f["check"] for f in failures]
        assert "overall_score" in codes

    def test_overall_score_exactly_at_threshold_passes(self, service):
        q = make_quality(overall_score=0.5)
        failures = service.check_quality_gate(q, threshold=0.5)
        assert all(f["check"] != "overall_score" for f in failures)

    def test_face_width_too_small(self, service):
        q = make_quality(face_width_px=40)  # < min_face_size_px=80
        failures = service.check_quality_gate(q, threshold=0.5)
        codes = [f["check"] for f in failures]
        assert "face_width_px" in codes

    def test_face_height_too_small(self, service):
        q = make_quality(face_height_px=40)  # < min_face_size_px=80
        failures = service.check_quality_gate(q, threshold=0.5)
        codes = [f["check"] for f in failures]
        assert "face_height_px" in codes

    def test_yaw_too_extreme_positive(self, service):
        q = make_quality(yaw_deg=40.0)  # > max_yaw_deg=35
        failures = service.check_quality_gate(q, threshold=0.5)
        codes = [f["check"] for f in failures]
        assert "yaw_deg" in codes

    def test_yaw_too_extreme_negative(self, service):
        q = make_quality(yaw_deg=-40.0)  # abs > max_yaw_deg=35
        failures = service.check_quality_gate(q, threshold=0.5)
        codes = [f["check"] for f in failures]
        assert "yaw_deg" in codes

    def test_yaw_exactly_at_limit_passes(self, service):
        q = make_quality(yaw_deg=35.0)  # == max_yaw_deg, not strictly greater
        failures = service.check_quality_gate(q, threshold=0.5)
        assert all(f["check"] != "yaw_deg" for f in failures)

    def test_pitch_too_extreme(self, service):
        q = make_quality(pitch_deg=35.0)  # > max_pitch_deg=30
        failures = service.check_quality_gate(q, threshold=0.5)
        codes = [f["check"] for f in failures]
        assert "pitch_deg" in codes

    def test_pitch_exactly_at_limit_passes(self, service):
        q = make_quality(pitch_deg=30.0)
        failures = service.check_quality_gate(q, threshold=0.5)
        assert all(f["check"] != "pitch_deg" for f in failures)

    def test_roll_too_extreme(self, service):
        q = make_quality(roll_deg=30.0)  # > max_roll_deg=25
        failures = service.check_quality_gate(q, threshold=0.5)
        codes = [f["check"] for f in failures]
        assert "roll_deg" in codes

    def test_roll_exactly_at_limit_passes(self, service):
        q = make_quality(roll_deg=25.0)
        failures = service.check_quality_gate(q, threshold=0.5)
        assert all(f["check"] != "roll_deg" for f in failures)

    def test_multiple_failures_returned(self, service):
        q = make_quality(face_width_px=20, face_height_px=20, yaw_deg=50.0)
        failures = service.check_quality_gate(q, threshold=0.5)
        assert len(failures) >= 3

    def test_failure_dict_has_required_keys(self, service):
        q = make_quality(overall_score=0.1)
        failures = service.check_quality_gate(q, threshold=0.5)
        assert len(failures) > 0
        for f in failures:
            assert "check" in f
            assert "value" in f
            assert "minimum" in f
            assert "reason" in f

    def test_failure_value_is_numeric(self, service):
        q = make_quality(overall_score=0.1)
        failures = service.check_quality_gate(q, threshold=0.5)
        for f in failures:
            assert isinstance(f["value"], (int, float))
            assert isinstance(f["minimum"], (int, float))


# ---------------------------------------------------------------------------
# TestCheckDuplicate
# ---------------------------------------------------------------------------


class TestCheckDuplicate:
    def test_no_stored_embeddings(self, service):
        emb = unit_vec(1)
        result = service.check_duplicate(emb, [], threshold=0.95)
        assert result.is_duplicate is False
        assert result.existing_face_id is None
        assert result.similarity is None

    def test_identical_embedding_is_duplicate(self, service):
        emb = unit_vec(1)
        stored = [("face_id_1", emb.copy())]
        result = service.check_duplicate(emb, stored, threshold=0.95)
        assert result.is_duplicate is True
        assert result.existing_face_id == "face_id_1"
        assert result.similarity == pytest.approx(1.0, abs=1e-5)

    def test_dissimilar_embedding_not_duplicate(self, service):
        emb1 = unit_vec(1)
        emb2 = unit_vec(999)
        stored = [("face_id_1", emb2)]
        result = service.check_duplicate(emb1, stored, threshold=0.95)
        assert result.is_duplicate is False

    def test_picks_most_similar(self, service):
        query = unit_vec(1)
        close = query.copy()  # sim = 1.0
        far = unit_vec(999)
        stored = [("face_a", far), ("face_b", close)]
        result = service.check_duplicate(query, stored, threshold=0.99)
        assert result.is_duplicate is True
        assert result.existing_face_id == "face_b"

    def test_threshold_boundary_below(self, service):
        emb = unit_vec(1)
        # Create a slightly different embedding
        noise = unit_vec(2) * 0.01
        other = emb + noise
        other /= np.linalg.norm(other)
        sim = FaceMLService.cosine_similarity(emb, other)
        # Use threshold just above the actual similarity → not duplicate
        result = service.check_duplicate(emb, [("face_x", other)], threshold=sim + 0.01)
        assert result.is_duplicate is False

    def test_threshold_boundary_at(self, service):
        emb = unit_vec(1)
        other = unit_vec(2)
        sim = FaceMLService.cosine_similarity(emb, other)
        # Use threshold exactly at the similarity → duplicate (>=)
        result = service.check_duplicate(emb, [("face_x", other)], threshold=sim)
        assert result.is_duplicate is True

    def test_returns_duplicate_check_result_type(self, service):
        emb = unit_vec(1)
        result = service.check_duplicate(emb, [], threshold=0.95)
        assert isinstance(result, DuplicateCheckResult)

    def test_multiple_stored_picks_best(self, service):
        query = unit_vec(1)
        stored = [(f"face_{i}", unit_vec(i + 10)) for i in range(5)]
        # Find the one most similar to query
        sims = [(fid, FaceMLService.cosine_similarity(query, emb)) for fid, emb in stored]
        best_fid, best_sim = max(sims, key=lambda x: x[1])

        result = service.check_duplicate(query, stored, threshold=best_sim)
        assert result.existing_face_id == best_fid


# ---------------------------------------------------------------------------
# TestGetEmbedding
# ---------------------------------------------------------------------------


class TestGetEmbedding:
    def test_normalized_output(self, service):
        """get_embedding should return an L2-normalized float32 vector."""
        raw_emb = np.random.randn(512).astype(np.float32) * 10.0  # unnormalized
        fake_face = MagicMock()
        fake_face.embedding = raw_emb

        result = service.get_embedding(fake_face)
        norm = np.linalg.norm(result)
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_zero_norm_returned_as_is(self, service):
        """A zero vector should be returned without division by zero."""
        fake_face = MagicMock()
        fake_face.embedding = np.zeros(512, dtype=np.float32)
        result = service.get_embedding(fake_face)
        assert result is not None
        assert not np.any(np.isnan(result))

    def test_output_dtype_is_float32(self, service):
        fake_face = MagicMock()
        fake_face.embedding = np.random.randn(512).astype(np.float64)
        result = service.get_embedding(fake_face)
        assert result.dtype == np.float32
