"""Integration tests for the face verification endpoint (POST /v1/users/{user_id}/verify)."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from app.services.face_ml import DuplicateCheckResult, FaceDetectionResult, MLProcessResult, QualityMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_embedding(seed: int = 1) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


def make_ml_result(seed: int = 1) -> tuple[list, MLProcessResult]:
    emb = make_embedding(seed)
    q = QualityMetrics(
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
    face = FaceDetectionResult(
        embedding=emb,
        quality=q,
        bbox=[100.0, 80.0, 300.0, 320.0],
        landmarks=[],
    )
    return [object()], MLProcessResult(face=face)


def make_unique_image_b64(seed: int) -> str:
    img = Image.new("RGB", (200, 200), color=(seed * 10 % 256, seed * 5 % 256, seed * 3 % 256))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


async def enroll_face(
    client, auth_headers, mock_face_ml, user_id: str, seed: int = 1
) -> str:
    """Helper: create user + enroll a face and return face_id."""
    mock_face_ml.process_image.return_value = make_ml_result(seed)
    mock_face_ml.check_quality_gate.return_value = []
    mock_face_ml.check_duplicate.return_value = DuplicateCheckResult(is_duplicate=False)
    await client.post("/v1/users", json={"external_id": user_id}, headers=auth_headers)
    resp = await client.post(
        f"/v1/users/{user_id}/faces",
        json={"image": make_unique_image_b64(seed)},
        headers=auth_headers,
    )
    assert resp.status_code == 201, f"Enroll failed: {resp.json()}"
    return resp.json()["data"]["face_id"]


# ---------------------------------------------------------------------------
# TestVerify
# ---------------------------------------------------------------------------


class TestVerify:
    async def test_verify_match(self, client, auth_headers, mock_face_ml):
        """Same embedding seed → high similarity → match."""
        await enroll_face(client, auth_headers, mock_face_ml, "verify_match_user", seed=1)

        # Verify with the same embedding
        mock_face_ml.process_image.return_value = make_ml_result(seed=1)
        resp = await client.post(
            "/v1/users/verify_match_user/verify",
            json={"image": make_unique_image_b64(1), "threshold": 0.5},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["match"] is True
        assert data["confidence"] >= 0.5
        assert data["threshold_used"] == pytest.approx(0.5)

    async def test_verify_no_match(self, client, auth_headers, mock_face_ml):
        """Different embedding seed + strict threshold → no match."""
        await enroll_face(client, auth_headers, mock_face_ml, "verify_nomatch_user", seed=1)

        # Verify with a very different embedding and strict threshold
        mock_face_ml.process_image.return_value = make_ml_result(seed=999)
        resp = await client.post(
            "/v1/users/verify_nomatch_user/verify",
            json={"image": make_unique_image_b64(2), "threshold": 0.99},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["match"] is False

    async def test_verify_user_not_found(self, client, auth_headers):
        resp = await client.post(
            "/v1/users/ghost_verify_user/verify",
            json={"image": make_unique_image_b64(3)},
            headers=auth_headers,
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "USER_NOT_FOUND"

    async def test_verify_user_has_no_faces(self, client, auth_headers, mock_face_ml):
        await client.post(
            "/v1/users",
            json={"external_id": "nofaces_verify_user"},
            headers=auth_headers,
        )
        mock_face_ml.process_image.return_value = make_ml_result(seed=1)
        resp = await client.post(
            "/v1/users/nofaces_verify_user/verify",
            json={"image": make_unique_image_b64(4)},
            headers=auth_headers,
        )
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "USER_HAS_NO_FACES"

    async def test_verify_no_face_in_image(self, client, auth_headers, mock_face_ml):
        await enroll_face(client, auth_headers, mock_face_ml, "noface_img_verify_user", seed=1)

        mock_face_ml.process_image.return_value = ([], None)
        resp = await client.post(
            "/v1/users/noface_img_verify_user/verify",
            json={"image": make_unique_image_b64(5)},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "NO_FACE_DETECTED"

    async def test_verify_multiple_faces_in_image(self, client, auth_headers, mock_face_ml):
        await enroll_face(client, auth_headers, mock_face_ml, "multi_verify_user", seed=1)

        mock_face_ml.process_image.return_value = ([object(), object()], None)
        resp = await client.post(
            "/v1/users/multi_verify_user/verify",
            json={"image": make_unique_image_b64(6)},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "MULTIPLE_FACES"

    async def test_verify_returns_all_scores(self, client, auth_headers, mock_face_ml):
        await enroll_face(client, auth_headers, mock_face_ml, "allscores_verify_user", seed=1)

        mock_face_ml.process_image.return_value = make_ml_result(seed=1)
        resp = await client.post(
            "/v1/users/allscores_verify_user/verify",
            json={"image": make_unique_image_b64(1)},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert isinstance(data["all_scores"], list)
        assert len(data["all_scores"]) == 1
        assert "face_id" in data["all_scores"][0]
        assert "similarity" in data["all_scores"][0]

    async def test_verify_all_scores_sorted_descending(self, client, auth_headers, mock_face_ml):
        """When a user has multiple faces, all_scores must be sorted by similarity descending."""
        user_id = "multifaced_verify_user"
        # Enroll 3 faces with different seeds
        for i in range(3):
            mock_face_ml.process_image.return_value = make_ml_result(seed=i + 50)
            mock_face_ml.check_quality_gate.return_value = []
            mock_face_ml.check_duplicate.return_value = DuplicateCheckResult(is_duplicate=False)
            if i == 0:
                await client.post("/v1/users", json={"external_id": user_id}, headers=auth_headers)
            await client.post(
                f"/v1/users/{user_id}/faces",
                json={"image": make_unique_image_b64(i + 50)},
                headers=auth_headers,
            )

        mock_face_ml.process_image.return_value = make_ml_result(seed=50)  # Same as first enrolled
        resp = await client.post(
            f"/v1/users/{user_id}/verify",
            json={"image": make_unique_image_b64(50)},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        scores = resp.json()["data"]["all_scores"]
        sims = [s["similarity"] for s in scores]
        assert sims == sorted(sims, reverse=True)

    async def test_verify_response_has_processing_time(self, client, auth_headers, mock_face_ml):
        await enroll_face(client, auth_headers, mock_face_ml, "timing_verify_user", seed=1)

        mock_face_ml.process_image.return_value = make_ml_result(seed=1)
        resp = await client.post(
            "/v1/users/timing_verify_user/verify",
            json={"image": make_unique_image_b64(7)},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["processing_time_ms"] >= 0

    async def test_verify_uses_default_threshold_when_not_provided(
        self, client, auth_headers, mock_face_ml, test_settings
    ):
        await enroll_face(client, auth_headers, mock_face_ml, "default_thresh_user", seed=1)

        mock_face_ml.process_image.return_value = make_ml_result(seed=1)
        resp = await client.post(
            "/v1/users/default_thresh_user/verify",
            json={"image": make_unique_image_b64(8)},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["threshold_used"] == pytest.approx(
            test_settings.default_verification_threshold, abs=0.01
        )

    async def test_verify_custom_threshold_used(self, client, auth_headers, mock_face_ml):
        await enroll_face(client, auth_headers, mock_face_ml, "custom_thresh_user", seed=1)

        mock_face_ml.process_image.return_value = make_ml_result(seed=1)
        resp = await client.post(
            "/v1/users/custom_thresh_user/verify",
            json={"image": make_unique_image_b64(9), "threshold": 0.75},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["threshold_used"] == pytest.approx(0.75)

    async def test_verify_same_embedding_matches_low_threshold(
        self, client, auth_headers, mock_face_ml
    ):
        await enroll_face(client, auth_headers, mock_face_ml, "strict_low_thresh_user", seed=1)

        mock_face_ml.process_image.return_value = make_ml_result(seed=1)  # Identical embedding
        resp = await client.post(
            "/v1/users/strict_low_thresh_user/verify",
            json={"image": make_unique_image_b64(1), "threshold": 0.01},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["match"] is True  # sim ~1.0 >= 0.01

    async def test_verify_invalid_image(self, client, auth_headers, mock_face_ml):
        await enroll_face(client, auth_headers, mock_face_ml, "badimg_verify_user", seed=1)

        resp = await client.post(
            "/v1/users/badimg_verify_user/verify",
            json={"image": "dGhpcyBpcyBub3QgYW4gaW1hZ2U="},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "INVALID_IMAGE"

    async def test_verify_no_auth(self, client):
        resp = await client.post(
            "/v1/users/some_user/verify",
            json={"image": make_unique_image_b64(10)},
        )
        assert resp.status_code == 401

    async def test_verify_response_has_query_face_quality(self, client, auth_headers, mock_face_ml):
        await enroll_face(client, auth_headers, mock_face_ml, "quality_resp_verify_user", seed=1)

        mock_face_ml.process_image.return_value = make_ml_result(seed=1)
        resp = await client.post(
            "/v1/users/quality_resp_verify_user/verify",
            json={"image": make_unique_image_b64(11)},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        qfq = resp.json()["data"]["query_face_quality"]
        assert "overall_score" in qfq
        assert "blur_score" in qfq
        assert "brightness" in qfq
        assert "face_confidence" in qfq

    async def test_verify_response_has_best_matching_face_id(
        self, client, auth_headers, mock_face_ml
    ):
        face_id = await enroll_face(
            client, auth_headers, mock_face_ml, "best_face_verify_user", seed=1
        )

        mock_face_ml.process_image.return_value = make_ml_result(seed=1)
        resp = await client.post(
            "/v1/users/best_face_verify_user/verify",
            json={"image": make_unique_image_b64(1)},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["best_matching_face_id"] == face_id

    async def test_verify_returns_user_id_in_response(self, client, auth_headers, mock_face_ml):
        await enroll_face(client, auth_headers, mock_face_ml, "userid_resp_verify_user", seed=1)

        mock_face_ml.process_image.return_value = make_ml_result(seed=1)
        resp = await client.post(
            "/v1/users/userid_resp_verify_user/verify",
            json={"image": make_unique_image_b64(12)},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["user_id"] == "userid_resp_verify_user"

    async def test_verify_envelope_format(self, client, auth_headers, mock_face_ml):
        await enroll_face(client, auth_headers, mock_face_ml, "envelope_verify_user", seed=1)

        mock_face_ml.process_image.return_value = make_ml_result(seed=1)
        resp = await client.post(
            "/v1/users/envelope_verify_user/verify",
            json={"image": make_unique_image_b64(13)},
            headers=auth_headers,
        )
        body = resp.json()
        assert "success" in body
        assert "data" in body
        assert "error" in body
        assert "request_id" in body
        assert body["request_id"].startswith("req_")
        assert body["success"] is True
        assert body["error"] is None
