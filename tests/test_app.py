"""Integration test for FastAPI endpoints."""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_get_status():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "activitywatch" in data
    assert "classifier" in data
    assert "gemma_oracle" in data
    assert "confidence_threshold" in data
    assert "target_bucket_summary" in data
    assert "local_model" in data["gemma_oracle"]


def test_get_bucket_stats():
    response = client.get("/api/bucket-stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_events" in data
    assert "fast_classifier" in data
    assert "gemma_oracle" in data


def test_get_buckets():
    response = client.get("/api/buckets")
    assert response.status_code == 200
    data = response.json()
    assert "buckets" in data
    assert "source_buckets" in data
    assert "classified_buckets" in data


def test_update_settings():
    response = client.post("/api/settings", json={"confidence_threshold": 0.82})
    assert response.status_code == 200
    data = response.json()
    assert data["settings"]["confidence_threshold"] == 0.82


def test_correct_event():
    response = client.post(
        "/api/correct-event",
        json={
            "feature_text": "app:slack title:dev discussions",
            "correct_category": "Communication",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "trained"


def test_serve_ui():
    response = client.get("/")
    assert response.status_code == 200
    assert "ActivityWatch Classifier" in response.text


def test_llama_server_endpoints():
    response = client.post("/api/llama-server/download")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "installed"
    assert "binary_path" in data

    stop_resp = client.post("/api/llama-server/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["status"] == "stopped"
