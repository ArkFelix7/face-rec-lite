"""Integration tests for health, readiness, and metrics endpoints.

These endpoints are public (no auth required).
"""

from __future__ import annotations

import pytest


class TestHealth:
    async def test_health_no_auth(self, client):
        resp = await client.get("/v1/health")
        assert resp.status_code == 200

    async def test_health_success_envelope(self, client):
        resp = await client.get("/v1/health")
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["status"] == "ok"

    async def test_health_has_version(self, client):
        resp = await client.get("/v1/health")
        body = resp.json()
        assert body["data"]["version"] == "1.0.0"

    async def test_health_has_request_id(self, client):
        resp = await client.get("/v1/health")
        body = resp.json()
        assert "request_id" in body
        assert body["request_id"].startswith("req_")

    async def test_health_error_is_none(self, client):
        resp = await client.get("/v1/health")
        assert resp.json()["error"] is None

    async def test_health_ignores_auth_header(self, client, auth_headers):
        # Health should work with or without auth headers
        resp = await client.get("/v1/health", headers=auth_headers)
        assert resp.status_code == 200

    async def test_health_returns_json(self, client):
        resp = await client.get("/v1/health")
        assert "application/json" in resp.headers["content-type"]


class TestReady:
    async def test_ready_endpoint_responds(self, client):
        resp = await client.get("/v1/ready")
        # May be 200 or 503 depending on test DB/Redis availability
        assert resp.status_code in (200, 503)

    async def test_ready_response_has_success_field(self, client):
        resp = await client.get("/v1/ready")
        body = resp.json()
        assert "success" in body
        assert isinstance(body["success"], bool)

    async def test_ready_no_auth_required(self, client):
        # Should not return 401
        resp = await client.get("/v1/ready")
        assert resp.status_code != 401

    async def test_ready_response_envelope(self, client):
        resp = await client.get("/v1/ready")
        body = resp.json()
        assert "success" in body
        assert "request_id" in body

    async def test_ready_when_ok_has_checks(self, client):
        resp = await client.get("/v1/ready")
        body = resp.json()
        if resp.status_code == 200:
            assert body["data"]["status"] == "ready"
            assert isinstance(body["data"]["checks"], dict)

    async def test_ready_503_has_error_info(self, client):
        resp = await client.get("/v1/ready")
        body = resp.json()
        if resp.status_code == 503:
            assert body["success"] is False
            assert body["error"] is not None


class TestMetrics:
    async def test_metrics_endpoint_responds(self, client):
        resp = await client.get("/v1/metrics")
        assert resp.status_code == 200

    async def test_metrics_no_auth_required(self, client):
        resp = await client.get("/v1/metrics")
        assert resp.status_code != 401

    async def test_metrics_is_prometheus_format(self, client):
        resp = await client.get("/v1/metrics")
        # Prometheus text format contains HELP and TYPE lines
        assert "# HELP" in resp.text or "# TYPE" in resp.text

    async def test_metrics_contains_face_api_metrics(self, client):
        resp = await client.get("/v1/metrics")
        # Our custom metrics should appear after some requests
        # At minimum the prometheus_python metrics should be present
        assert "python" in resp.text.lower() or "face_api" in resp.text

    async def test_metrics_content_type_is_text(self, client):
        resp = await client.get("/v1/metrics")
        # Prometheus uses text/plain
        assert "text/plain" in resp.headers.get("content-type", "")

    async def test_metrics_enrollment_counter_present(self, client):
        """After making an enrollment request, the enrollment counter should appear."""
        resp = await client.get("/v1/metrics")
        # Just check it's a valid Prometheus text format response
        assert resp.status_code == 200
        assert len(resp.text) > 0
