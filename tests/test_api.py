"""API tests with FastAPI's TestClient (no model or network needed)."""
from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_generate_rejects_blank_company():
    response = client.post("/api/generate", json={"company": "   ", "person": ""})
    assert response.status_code == 400


def test_unknown_job_returns_404():
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_metrics_endpoint_counts_requests():
    client.get("/api/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "brandpilot_http_requests_total" in body
    assert 'route="/api/health"' in body
    assert "brandpilot_generation_jobs_total" in body
