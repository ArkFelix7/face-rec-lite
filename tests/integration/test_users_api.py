"""Integration tests for the User API endpoints.

All tests use a real PostgreSQL database (facedb_test) and a real Redis instance,
but the ML model is mocked out via the mock_face_ml fixture.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


class TestCreateUser:
    async def test_create_user_success(self, client: AsyncClient, auth_headers):
        resp = await client.post(
            "/v1/users",
            json={
                "external_id": "user_test_create_001",
                "display_name": "Test User",
                "metadata": {"dept": "eng"},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["external_id"] == "user_test_create_001"
        assert body["data"]["display_name"] == "Test User"
        assert body["data"]["face_count"] == 0
        assert "request_id" in body

    async def test_create_user_response_has_id(self, client, auth_headers):
        resp = await client.post(
            "/v1/users",
            json={"external_id": "user_with_id_check"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert "id" in data
        assert len(data["id"]) == 36  # UUID string length

    async def test_create_user_duplicate_external_id(self, client, auth_headers):
        await client.post(
            "/v1/users", json={"external_id": "dup_user_ext"}, headers=auth_headers
        )
        resp = await client.post(
            "/v1/users", json={"external_id": "dup_user_ext"}, headers=auth_headers
        )
        assert resp.status_code == 409
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "USER_ALREADY_EXISTS"

    async def test_create_user_no_auth(self, client):
        resp = await client.post("/v1/users", json={"external_id": "no_auth_user"})
        assert resp.status_code == 401
        body = resp.json()
        assert body["success"] is False

    async def test_create_user_invalid_api_key(self, client):
        resp = await client.post(
            "/v1/users",
            json={"external_id": "bad_key_user"},
            headers={"Authorization": "Bearer sk_live_invalid_key_000000000000"},
        )
        assert resp.status_code == 401
        body = resp.json()
        assert body["success"] is False
        # Auth middleware returns UNAUTHORIZED for invalid keys
        assert body["error"]["code"] in ("UNAUTHORIZED", "INVALID_API_KEY")

    async def test_create_user_minimal_body(self, client, auth_headers):
        resp = await client.post(
            "/v1/users", json={"external_id": "minimal_user_body"}, headers=auth_headers
        )
        assert resp.status_code == 201
        assert resp.json()["data"]["display_name"] is None
        assert resp.json()["data"]["metadata"] == {}

    async def test_create_user_response_has_request_id(self, client, auth_headers):
        resp = await client.post(
            "/v1/users", json={"external_id": "req_id_check_user"}, headers=auth_headers
        )
        assert resp.json()["request_id"].startswith("req_")

    async def test_create_user_with_metadata(self, client, auth_headers):
        resp = await client.post(
            "/v1/users",
            json={"external_id": "meta_user", "metadata": {"key": "value", "num": 42}},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["data"]["metadata"] == {"key": "value", "num": 42}

    async def test_create_user_missing_external_id_fails(self, client, auth_headers):
        resp = await client.post(
            "/v1/users",
            json={"display_name": "No External ID"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_create_user_empty_external_id_fails(self, client, auth_headers):
        resp = await client.post(
            "/v1/users",
            json={"external_id": ""},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_create_user_response_has_timestamps(self, client, auth_headers):
        resp = await client.post(
            "/v1/users", json={"external_id": "timestamp_user"}, headers=auth_headers
        )
        data = resp.json()["data"]
        assert "created_at" in data
        assert data["created_at"] is not None

    async def test_create_user_without_bearer_prefix_fails(self, client):
        resp = await client.post(
            "/v1/users",
            json={"external_id": "no_bearer"},
            headers={"Authorization": "sk_live_somekey"},
        )
        assert resp.status_code == 401


class TestGetUser:
    async def test_get_user_success(self, client, auth_headers):
        await client.post(
            "/v1/users", json={"external_id": "get_me_user"}, headers=auth_headers
        )
        resp = await client.get("/v1/users/get_me_user", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["external_id"] == "get_me_user"

    async def test_get_user_not_found(self, client, auth_headers):
        resp = await client.get("/v1/users/nonexistent_xyz_user", headers=auth_headers)
        assert resp.status_code == 404
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "USER_NOT_FOUND"

    async def test_get_user_has_face_count_zero(self, client, auth_headers):
        await client.post(
            "/v1/users", json={"external_id": "face_count_check_user"}, headers=auth_headers
        )
        resp = await client.get("/v1/users/face_count_check_user", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["data"]["face_count"] == 0

    async def test_get_user_no_auth(self, client):
        resp = await client.get("/v1/users/some_user")
        assert resp.status_code == 401

    async def test_get_user_response_envelope(self, client, auth_headers):
        await client.post(
            "/v1/users", json={"external_id": "envelope_get_user"}, headers=auth_headers
        )
        resp = await client.get("/v1/users/envelope_get_user", headers=auth_headers)
        body = resp.json()
        assert "success" in body
        assert "data" in body
        assert "error" in body
        assert "request_id" in body

    async def test_get_user_returns_metadata(self, client, auth_headers):
        await client.post(
            "/v1/users",
            json={"external_id": "meta_get_user", "metadata": {"role": "admin"}},
            headers=auth_headers,
        )
        resp = await client.get("/v1/users/meta_get_user", headers=auth_headers)
        assert resp.json()["data"]["metadata"] == {"role": "admin"}

    async def test_get_user_returns_display_name(self, client, auth_headers):
        await client.post(
            "/v1/users",
            json={"external_id": "named_user", "display_name": "John Doe"},
            headers=auth_headers,
        )
        resp = await client.get("/v1/users/named_user", headers=auth_headers)
        assert resp.json()["data"]["display_name"] == "John Doe"


class TestDeleteUser:
    async def test_delete_user_success(self, client, auth_headers):
        await client.post(
            "/v1/users", json={"external_id": "delete_me_user"}, headers=auth_headers
        )
        resp = await client.delete("/v1/users/delete_me_user", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["deleted_user_id"] == "delete_me_user"

    async def test_delete_user_not_found(self, client, auth_headers):
        resp = await client.delete("/v1/users/nonexistent_abc_user", headers=auth_headers)
        assert resp.status_code == 404
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "USER_NOT_FOUND"

    async def test_delete_user_then_get_fails(self, client, auth_headers):
        await client.post(
            "/v1/users", json={"external_id": "gone_user_test"}, headers=auth_headers
        )
        await client.delete("/v1/users/gone_user_test", headers=auth_headers)
        resp = await client.get("/v1/users/gone_user_test", headers=auth_headers)
        assert resp.status_code == 404

    async def test_delete_user_returns_faces_deleted_count(self, client, auth_headers):
        await client.post(
            "/v1/users", json={"external_id": "faces_del_count_user"}, headers=auth_headers
        )
        resp = await client.delete("/v1/users/faces_del_count_user", headers=auth_headers)
        assert resp.status_code == 200
        assert "faces_deleted" in resp.json()["data"]
        assert resp.json()["data"]["faces_deleted"] == 0

    async def test_delete_user_no_auth(self, client):
        resp = await client.delete("/v1/users/some_user")
        assert resp.status_code == 401

    async def test_delete_user_response_has_deleted_at(self, client, auth_headers):
        await client.post(
            "/v1/users", json={"external_id": "del_at_user"}, headers=auth_headers
        )
        resp = await client.delete("/v1/users/del_at_user", headers=auth_headers)
        assert "deleted_at" in resp.json()["data"]
        assert resp.json()["data"]["deleted_at"] is not None

    async def test_delete_user_then_recreate_succeeds(self, client, auth_headers):
        await client.post(
            "/v1/users", json={"external_id": "recreate_user"}, headers=auth_headers
        )
        await client.delete("/v1/users/recreate_user", headers=auth_headers)
        # Re-creating the same external_id should succeed after deletion
        resp = await client.post(
            "/v1/users", json={"external_id": "recreate_user"}, headers=auth_headers
        )
        assert resp.status_code == 201
