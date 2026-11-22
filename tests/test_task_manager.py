"""Tests for TaskManager and background task endpoints."""

import time
from fastapi.testclient import TestClient
from app import app
from task_manager import TaskManager, task_manager

client = TestClient(app, base_url="http://localhost")


def test_task_manager_lifecycle():
    tm = TaskManager()

    def dummy_job(progress_callback=None):
        if progress_callback:
            progress_callback({"step": "step1", "percent": 30.0, "message": "Working..."})
        time.sleep(0.05)
        if progress_callback:
            progress_callback({"step": "step2", "percent": 70.0, "message": "Almost done..."})
        time.sleep(0.05)
        return {"items": 42}

    task = tm.start_task("dummy_test", dummy_job)
    assert task.id.startswith("task_")
    assert task.status in ("pending", "running")

    # While running, tm.get_active_task() should return the task
    active = tm.get_active_task()
    if active:
        assert active.id == task.id

    # Wait for completion
    timeout = time.time() + 2.0
    while task.status == "running" and time.time() < timeout:
        time.sleep(0.02)

    assert task.status == "completed"
    assert task.results == {"items": 42}
    assert task.progress["percent"] == 100.0


def test_api_task_endpoints():
    def quick_job(progress_callback=None):
        if progress_callback:
            progress_callback({"step": "running", "percent": 50.0, "message": "In flight..."})
        time.sleep(0.05)
        return {"status": "ok"}

    task = task_manager.start_task("api_test", quick_job)
    task_id = task.id

    # Test /api/tasks/active
    res = client.get("/api/tasks/active")
    assert res.status_code == 200

    # Test /api/tasks/{task_id}
    res_task = client.get(f"/api/tasks/{task_id}")
    assert res_task.status_code == 200
    data = res_task.json()
    assert data["id"] == task_id

    # Test /api/status contains active_task
    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert "active_task" in status_data

    # Wait for task completion
    timeout = time.time() + 2.0
    while task.status == "running" and time.time() < timeout:
        time.sleep(0.02)

    assert task.status == "completed"


def test_pipeline_run_background_task_and_refresh_persistence():
    from unittest.mock import MagicMock
    import app as app_module

    # Mock pipeline.run to simulate a job that reports progress over time
    mock_pipeline = MagicMock()
    def fake_pipeline_run(**kwargs):
        cb = kwargs.get("progress_callback")
        if cb:
            cb({"step": "fetching", "percent": 20.0, "message": "Fetching..."})
        time.sleep(0.05)
        if cb:
            cb({"step": "classifying", "percent": 60.0, "message": "Classifying 60/100..."})
        time.sleep(0.05)
        if cb:
            cb({"step": "completed", "percent": 100.0, "message": "Completed"})
        return {"total_events_processed": 100, "fast_classifier_calls": 80, "gemma_oracle_calls": 20}

    orig_pipeline = app_module.pipeline
    app_module.pipeline = mock_pipeline
    mock_pipeline.run.side_effect = fake_pipeline_run

    try:
        # Start pipeline via API
        req_payload = {
            "source_bucket_ids": ["test-bucket-1"],
            "target_bucket_id": "test-target",
            "limit_per_bucket": 100,
        }
        res = client.post("/api/run-pipeline", json=req_payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "started"
        task_id = data["task_id"]

        # Simulate page refresh 1: Immediate fetch /api/status
        res_status = client.get("/api/status")
        assert res_status.status_code == 200
        status_data = res_status.json()
        assert "active_task" in status_data
        active_task = status_data["active_task"]
        assert active_task is not None
        assert active_task["id"] == task_id

        # Simulate polling
        res_task = client.get(f"/api/tasks/{task_id}")
        assert res_task.status_code == 200
        task_data = res_task.json()
        assert task_data["task_type"] == "pipeline_run"
        assert task_data["status"] in ("running", "completed")

        # Wait for task completion
        timeout = time.time() + 2.0
        while task_data["status"] == "running" and time.time() < timeout:
            time.sleep(0.03)
            task_data = client.get(f"/api/tasks/{task_id}").json()

        assert task_data["status"] == "completed"
        assert task_data["results"]["total_events_processed"] == 100
        assert task_data["progress"]["percent"] == 100.0

        # Simulate page refresh 2: After completion, /api/tasks/active returns last_completed
        res_active = client.get("/api/tasks/active")
        assert res_active.status_code == 200
        active_data = res_active.json()
        assert active_data["has_active"] is False
        assert active_data["last_completed"]["id"] == task_id
        assert active_data["last_completed"]["status"] == "completed"
    finally:
        app_module.pipeline = orig_pipeline


def test_second_start_while_first_queued_returns_same_task():
    import threading
    from task_manager import TaskManager

    tm = TaskManager()
    release = threading.Event()
    first = tm.start_task("pipeline_run", lambda progress_callback=None: release.wait(5))
    second = tm.start_task("pipeline_run", lambda progress_callback=None: None)
    assert second is first
    assert tm.get_active_task() is first
    release.set()
