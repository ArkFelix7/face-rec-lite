"""Integration tests for the Face enrollment, listing, and deletion endpoints."""

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


def make_good_quality() -> QualityMetrics:
    return QualityMetrics(
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


def make_ml_result_for_mock(seed: int = 42) -> tuple[list, MLProcessResult]:
    rng = np.random.RandomState(seed)
    emb = rng.randn(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    face = FaceDetectionResult(
        embedding=emb,
        quality=make_good_quality(),
        bbox=[100.0, 80.0, 300.0, 320.0],
        landmarks=[],
    )
    return [object()], MLProcessResult(face=face)


def make_unique_image_b64(seed: int) -> str:
    """Create a unique JPEG image for a given seed."""
    img = Image.new("RGB", (200, 200), color=(seed * 10 % 256, seed * 5 % 256, seed * 3 % 256))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# TestEnrollFace
# ---------------------------------------------------------------------------


class TestEnrollFace:
    async def test_enroll_success(self, client, auth_headers, valid_image_b64, mock_face_ml):
        mock_face_ml.process_image.return_value = make_ml_result_for_mock()
        mock_face_ml.check_quality_gate.return_value = []
        mock_face_ml.check_duplicate.return_value = DuplicateCheckResult(is_duplicate=False)

        await client.post(
            "/v1/users", json={"external_id": "enroll_success_user"}, headers=auth_headers
        )
        resp = await client.post(
            "/v1/users/enroll_success_user/faces",
            json={"image": valid_image_b64, "label": "front"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert "face_id" in data
        assert data["user_id"] == "enroll_success_user"
        assert data["label"] == "front"
        assert data["duplicate"] is False
        assert data["quality_metrics"]["overall_score"] == pytest.approx(0.85, abs=0.01)

    async def test_enroll_response_envelope(self, client, auth_headers, valid_image_b64, mock_face_ml):
        mock_face_ml.process_image.return_value = make_ml_result_for_mock(seed=200)
        mock_face_ml.check_quality_gate.return_value = []
        mock_face_ml.check_duplicate.return_value = DuplicateCheckResult(is_duplicate=False)

        await client.post(
            "/v1/users", json={"external_id": "enroll_envelope_user"}, headers=auth_headers
        )
        resp = await client.post(
            "/v1/users/enroll_envelope_user/faces",
            json={"image": make_unique_image_b64(200)},
            headers=auth_headers,
        )
        body = resp.json()
        assert "success" in body
        assert "data" in body
        assert "request_id" in body
        assert body["request_id"].startswith("req_")

    async def test_enroll_user_not_found(self, client, auth_headers, valid_image_b64):
        resp = await client.post(
            "/v1/users/ghost_enroll_user/faces",
            json={"image": valid_image_b64},
            headers=auth_headers,
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "USER_NOT_FOUND"

    async def test_enroll_invalid_image(self, client, auth_headers):
        await client.post(
            "/v1/users", json={"external_id": "invalid_img_user"}, headers=auth_headers
        )
        # "this is not an image" in base64
        resp = await client.post(
            "/v1/users/invalid_img_user/faces",
            json={"image": "dGhpcyBpcyBub3QgYW4gaW1hZ2U="},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "INVALID_IMAGE"

    async def test_enroll_no_face_detected(self, client, auth_headers, valid_image_b64, mock_face_ml):
        mock_face_ml.process_image.return_value = ([], None)

        await client.post(
            "/v1/users", json={"external_id": "noface_enroll_user"}, headers=auth_headers
        )
        resp = await client.post(
            "/v1/users/noface_enroll_user/faces",
            json={"image": valid_image_b64},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "NO_FACE_DETECTED"

    async def test_enroll_multiple_faces(self, client, auth_headers, valid_image_b64, mock_face_ml):
        mock_face_ml.process_image.return_value = ([object(), object()], None)

        await client.post(
            "/v1/users", json={"external_id": "multi_face_user"}, headers=auth_headers
        )
        resp = await client.post(
            "/v1/users/multi_face_user/faces",
            json={"image": valid_image_b64},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "MULTIPLE_FACES"

    async def test_enroll_multiple_faces_returns_count(self, client, auth_headers, valid_image_b64, mock_face_ml):
        mock_face_ml.process_image.return_value = ([object(), object(), object()], None)

        await client.post(
            "/v1/users", json={"external_id": "multi3_face_user"}, headers=auth_headers
        )
        resp = await client.post(
            "/v1/users/multi3_face_user/faces",
            json={"image": valid_image_b64},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["details"]["face_count"] == 3

    async def test_enroll_low_quality(self, client, auth_headers, valid_image_b64, mock_face_ml):
        mock_face_ml.process_image.return_value = make_ml_result_for_mock()
        mock_face_ml.check_quality_gate.return_value = [
            {"check": "blur_score", "value": 0.10, "minimum": 0.40, "reason": "Too blurry"}
        ]

        await client.post(
            "/v1/users", json={"external_id": "low_quality_user"}, headers=auth_headers
        )
        resp = await client.post(
            "/v1/users/low_quality_user/faces",
            json={"image": valid_image_b64},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "LOW_QUALITY"
        details = resp.json()["error"]["details"]
        assert "failing_checks" in details
        assert len(details["failing_checks"]) == 1
        assert details["failing_checks"][0]["check"] == "blur_score"

    async def test_enroll_low_quality_multiple_checks(self, client, auth_headers, valid_image_b64, mock_face_ml):
        mock_face_ml.process_image.return_value = make_ml_result_for_mock()
        mock_face_ml.check_quality_gate.return_value = [
            {"check": "blur_score", "value": 0.10, "minimum": 0.40, "reason": "Too blurry"},
            {"check": "overall_score", "value": 0.20, "minimum": 0.50, "reason": "Too low"},
        ]

        await client.post(
            "/v1/users", json={"external_id": "low_quality_multi_user"}, headers=auth_headers
        )
        resp = await client.post(
            "/v1/users/low_quality_multi_user/faces",
            json={"image": valid_image_b64},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "LOW_QUALITY"
        assert len(resp.json()["error"]["details"]["failing_checks"]) == 2

    async def test_enroll_duplicate_by_hash(self, client, auth_headers, valid_image_b64, mock_face_ml):
        """Submitting the exact same image twice returns 200 with duplicate=True."""
        mock_face_ml.process_image.return_value = make_ml_result_for_mock()
        mock_face_ml.check_quality_gate.return_value = []
        mock_face_ml.check_duplicate.return_value = DuplicateCheckResult(is_duplicate=False)

        await client.post(
            "/v1/users", json={"external_id": "dedup_hash_user"}, headers=auth_headers
        )
        # First enrollment
        resp1 = await client.post(
            "/v1/users/dedup_hash_user/faces",
            json={"image": valid_image_b64},
            headers=auth_headers,
        )
        assert resp1.status_code == 201

        # Second enrollment with exact same image bytes (same hash)
        resp2 = await client.post(
            "/v1/users/dedup_hash_user/faces",
            json={"image": valid_image_b64},
            headers=auth_headers,
        )
        assert resp2.status_code == 200
        assert resp2.json()["data"]["duplicate"] is True
        assert resp2.json()["data"]["similarity"] == pytest.approx(1.0)

    async def test_enroll_max_faces_reached(
        self, client, auth_headers, mock_face_ml, test_settings
    ):
        await client.post(
            "/v1/users", json={"external_id": "maxface_enroll_user"}, headers=auth_headers
        )

        for i in range(test_settings.max_faces_per_user):
            mock_face_ml.process_image.return_value = make_ml_result_for_mock(seed=i + 100)
            mock_face_ml.check_quality_gate.return_value = []
            mock_face_ml.check_duplicate.return_value = DuplicateCheckResult(is_duplicate=False)
            img_b64 = make_unique_image_b64(seed=i + 100)
            enroll_resp = await client.post(
                "/v1/users/maxface_enroll_user/faces",
                json={"image": img_b64},
                headers=auth_headers,
            )
            assert enroll_resp.status_code == 201

        # One more should fail
        mock_face_ml.process_image.return_value = make_ml_result_for_mock(seed=999)
        mock_face_ml.check_duplicate.return_value = DuplicateCheckResult(is_duplicate=False)
        resp = await client.post(
            "/v1/users/maxface_enroll_user/faces",
            json={"image": make_unique_image_b64(999)},
            headers=auth_headers,
        )
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "MAX_FACES_REACHED"

    async def test_enroll_max_faces_error_has_details(
        self, client, auth_headers, mock_face_ml, test_settings
    ):
        await client.post(
            "/v1/users", json={"external_id": "maxface_details_user"}, headers=auth_headers
        )

        for i in range(test_settings.max_faces_per_user):
            mock_face_ml.process_image.return_value = make_ml_result_for_mock(seed=i + 200)
            mock_face_ml.check_quality_gate.return_value = []
            mock_face_ml.check_duplicate.return_value = DuplicateCheckResult(is_duplicate=False)
            await client.post(
                "/v1/users/maxface_details_user/faces",
                json={"image": make_unique_image_b64(i + 200)},
                headers=auth_headers,
            )

        resp = await client.post(
            "/v1/users/maxface_details_user/faces",
            json={"image": make_unique_image_b64(300)},
            headers=auth_headers,
        )
        assert resp.status_code == 409
        details = resp.json()["error"]["details"]
        assert "max" in details
        assert details["max"] == test_settings.max_faces_per_user

    async def test_enroll_no_auth(self, client, valid_image_b64):
        resp = await client.post(
            "/v1/users/some_user/faces",
            json={"image": valid_image_b64},
        )
        assert resp.status_code == 401

    async def test_enroll_data_uri_prefix_accepted(
        self, client, auth_headers, valid_image_b64_with_prefix, mock_face_ml
    ):
        mock_face_ml.process_image.return_value = make_ml_result_for_mock(seed=500)
        mock_face_ml.check_quality_gate.return_value = []
        mock_face_ml.check_duplicate.return_value = DuplicateCheckResult(is_duplicate=False)

        await client.post(
            "/v1/users", json={"external_id": "data_uri_user"}, headers=auth_headers
        )
        resp = await client.post(
            "/v1/users/data_uri_user/faces",
            json={"image": valid_image_b64_with_prefix},
            headers=auth_headers,
        )
        # Either 201 (success) or 200 (dup if hash matches) or 400 if different hash
        assert resp.status_code in (200, 201)

    async def test_enroll_without_label(self, client, auth_headers, valid_image_b64, mock_face_ml):
        mock_face_ml.process_image.return_value = make_ml_result_for_mock(seed=600)
        mock_face_ml.check_quality_gate.return_value = []
        mock_face_ml.check_duplicate.return_value = DuplicateCheckResult(is_duplicate=False)

        await client.post(
            "/v1/users", json={"external_id": "nolabel_user"}, headers=auth_headers
        )
        resp = await client.post(
            "/v1/users/nolabel_user/faces",
            json={"image": make_unique_image_b64(600)},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["data"]["label"] is None

    async def test_enroll_returns_quality_metrics_structure(
        self, client, auth_headers, valid_image_b64, mock_face_ml
    ):
        mock_face_ml.process_image.return_value = make_ml_result_for_mock(seed=700)
        mock_face_ml.check_quality_gate.return_value = []
        mock_face_ml.check_duplicate.return_value = DuplicateCheckResult(is_duplicate=False)

        await client.post(
            "/v1/users", json={"external_id": "qmetrics_user"}, headers=auth_headers
        )
        resp = await client.post(
            "/v1/users/qmetrics_user/faces",
            json={"image": make_unique_image_b64(700)},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        qm = resp.json()["data"]["quality_metrics"]
        assert "overall_score" in qm
        assert "blur_score" in qm
        assert "brightness" in qm
        assert "face_confidence" in qm
        assert "face_size" in qm
        assert "head_pose" in qm

    async def test_enroll_returns_bounding_box(
        self, client, auth_headers, mock_face_ml
    ):
        mock_face_ml.process_image.return_value = make_ml_result_for_mock(seed=800)
        mock_face_ml.check_quality_gate.return_value = []
        mock_face_ml.check_duplicate.return_value = DuplicateCheckResult(is_duplicate=False)

        await client.post(
            "/v1/users", json={"external_id": "bbox_user"}, headers=auth_headers
        )
        resp = await client.post(
            "/v1/users/bbox_user/faces",
            json={"image": make_unique_image_b64(800)},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        bb = resp.json()["data"]["bounding_box"]
        assert "x" in bb
        assert "y" in bb
        assert "width" in bb
        assert "height" in bb


# ---------------------------------------------------------------------------
# TestListFaces
# ---------------------------------------------------------------------------


class TestListFaces:
    async def test_list_faces_empty(self, client, auth_headers):
        await client.post(
            "/v1/users", json={"external_id": "listfaces_empty_user"}, headers=auth_headers
        )
        resp = await client.get("/v1/users/listfaces_empty_user/faces", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["total"] == 0
        assert body["data"]["faces"] == []

    async def test_list_faces_after_enroll(self, client, auth_headers, mock_face_ml):
        mock_face_ml.process_image.return_value = make_ml_result_for_mock(seed=10)
        mock_face_ml.check_quality_gate.return_value = []
        mock_face_ml.check_duplicate.return_value = DuplicateCheckResult(is_duplicate=False)

        await client.post(
            "/v1/users", json={"external_id": "listfaces_filled_user"}, headers=auth_headers
        )
        await client.post(
            "/v1/users/listfaces_filled_user/faces",
            json={"image": make_unique_image_b64(10), "label": "front"},
            headers=auth_headers,
        )
        resp = await client.get("/v1/users/listfaces_filled_user/faces", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total"] == 1
        assert data["faces"][0]["label"] == "front"

    async def test_list_faces_includes_quality_score(self, client, auth_headers, mock_face_ml):
        mock_face_ml.process_image.return_value = make_ml_result_for_mock(seed=11)
        mock_face_ml.check_quality_gate.return_value = []
        mock_face_ml.check_duplicate.return_value = DuplicateCheckResult(is_duplicate=False)

        await client.post(
            "/v1/users", json={"external_id": "listfaces_quality_user"}, headers=auth_headers
        )
        await client.post(
            "/v1/users/listfaces_quality_user/faces",
            json={"image": make_unique_image_b64(11)},
            headers=auth_headers,
        )
        resp = await client.get("/v1/users/listfaces_quality_user/faces", headers=auth_headers)
        face_item = resp.json()["data"]["faces"][0]
        assert "quality_score" in face_item
        assert "face_size" in face_item
        assert "enrolled_at" in face_item

    async def test_list_faces_user_not_found(self, client, auth_headers):
        resp = await client.get("/v1/users/ghost_listfaces_user/faces", headers=auth_headers)
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "USER_NOT_FOUND"

    async def test_list_faces_has_max_allowed(self, client, auth_headers, test_settings):
        await client.post(
            "/v1/users", json={"external_id": "listfaces_max_user"}, headers=auth_headers
        )
        resp = await client.get("/v1/users/listfaces_max_user/faces", headers=auth_headers)
        assert resp.json()["data"]["max_allowed"] == test_settings.max_faces_per_user

    async def test_list_faces_no_auth(self, client):
        resp = await client.get("/v1/users/some_user/faces")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# TestDeleteFace
# ---------------------------------------------------------------------------


class TestDeleteFace:
    async def test_delete_face_success(self, client, auth_headers, mock_face_ml):
        mock_face_ml.process_image.return_value = make_ml_result_for_mock(seed=20)
        mock_face_ml.check_quality_gate.return_value = []
        mock_face_ml.check_duplicate.return_value = DuplicateCheckResult(is_duplicate=False)

        await client.post(
            "/v1/users", json={"external_id": "delface_success_user"}, headers=auth_headers
        )
        enroll_resp = await client.post(
            "/v1/users/delface_success_user/faces",
            json={"image": make_unique_image_b64(20)},
            headers=auth_headers,
        )
        face_id = enroll_resp.json()["data"]["face_id"]

        resp = await client.delete(
            f"/v1/users/delface_success_user/faces/{face_id}", headers=auth_headers
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["deleted_face_id"] == face_id
        assert data["user_id"] == "delface_success_user"
        assert "remaining_faces" in data
        assert data["remaining_faces"] == 0

    async def test_delete_face_not_found(self, client, auth_headers):
        await client.post(
            "/v1/users", json={"external_id": "delface_notfound_user"}, headers=auth_headers
        )
        resp = await client.delete(
            "/v1/users/delface_notfound_user/faces/00000000-0000-0000-0000-000000000000",
            headers=auth_headers,
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "FACE_NOT_FOUND"

    async def test_delete_face_invalid_uuid(self, client, auth_headers):
        await client.post(
            "/v1/users", json={"external_id": "delface_uuid_user"}, headers=auth_headers
        )
        resp = await client.delete(
            "/v1/users/delface_uuid_user/faces/not-a-valid-uuid",
            headers=auth_headers,
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "FACE_NOT_FOUND"

    async def test_delete_face_then_list_is_empty(self, client, auth_headers, mock_face_ml):
        mock_face_ml.process_image.return_value = make_ml_result_for_mock(seed=30)
        mock_face_ml.check_quality_gate.return_value = []
        mock_face_ml.check_duplicate.return_value = DuplicateCheckResult(is_duplicate=False)

        await client.post(
            "/v1/users", json={"external_id": "delface_list_user"}, headers=auth_headers
        )
        enroll_resp = await client.post(
            "/v1/users/delface_list_user/faces",
            json={"image": make_unique_image_b64(30)},
            headers=auth_headers,
        )
        face_id = enroll_resp.json()["data"]["face_id"]

        await client.delete(
            f"/v1/users/delface_list_user/faces/{face_id}", headers=auth_headers
        )

        list_resp = await client.get(
            "/v1/users/delface_list_user/faces", headers=auth_headers
        )
        assert list_resp.json()["data"]["total"] == 0

    async def test_delete_face_user_not_found(self, client, auth_headers):
        resp = await client.delete(
            "/v1/users/ghost_face_user/faces/00000000-0000-0000-0000-000000000000",
            headers=auth_headers,
        )
        assert resp.status_code == 404
        # Either USER_NOT_FOUND or FACE_NOT_FOUND
        assert resp.json()["error"]["code"] in ("USER_NOT_FOUND", "FACE_NOT_FOUND")

    async def test_delete_face_no_auth(self, client):
        resp = await client.delete(
            "/v1/users/some_user/faces/00000000-0000-0000-0000-000000000000"
        )
        assert resp.status_code == 401

    async def test_delete_face_decrements_count(self, client, auth_headers, mock_face_ml):
        mock_face_ml.process_image.return_value = make_ml_result_for_mock(seed=40)
        mock_face_ml.check_quality_gate.return_value = []
        mock_face_ml.check_duplicate.return_value = DuplicateCheckResult(is_duplicate=False)

        await client.post(
            "/v1/users", json={"external_id": "delface_count_user"}, headers=auth_headers
        )
        enroll_resp = await client.post(
            "/v1/users/delface_count_user/faces",
            json={"image": make_unique_image_b64(40)},
            headers=auth_headers,
        )
        face_id = enroll_resp.json()["data"]["face_id"]

        # Verify user has 1 face
        get_resp = await client.get(
            "/v1/users/delface_count_user", headers=auth_headers
        )
        assert get_resp.json()["data"]["face_count"] == 1

        await client.delete(
            f"/v1/users/delface_count_user/faces/{face_id}", headers=auth_headers
        )

        # Verify user now has 0 faces
        get_resp2 = await client.get(
            "/v1/users/delface_count_user", headers=auth_headers
        )
        assert get_resp2.json()["data"]["face_count"] == 0
