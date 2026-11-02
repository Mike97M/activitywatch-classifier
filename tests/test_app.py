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


def test_get_buckets():
    response = client.get("/api/buckets")
    assert response.status_code == 200
    data = response.json()
    assert "buckets" in data


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
