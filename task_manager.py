"""Background Task Manager for ActivityWatch Classifier.

Provides thread-safe background execution, real-time progress tracking,
and persistent task state across web UI refreshes and multiple client tabs.
"""

import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("task_manager")


class Task:
    """Represents a background task with live progress reporting."""

    def __init__(self, task_id: str, task_type: str, metadata: Optional[Dict[str, Any]] = None):
        self.id = task_id
        self.task_type = task_type
        self.metadata = metadata or {}
        self.status = "pending"  # "pending", "running", "completed", "failed"
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.progress: Dict[str, Any] = {
            "current": 0,
            "total": 0,
            "percent": 0.0,
            "message": "Task queued...",
            "step": "pending",
            "stats": {},
            "live_events": [],
        }
        self.results: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

    def update_progress(
        self,
        current: int = 0,
        total: int = 0,
        percent: Optional[float] = None,
        message: str = "",
        step: str = "running",
        stats: Optional[Dict[str, Any]] = None,
        live_events: Optional[List[Dict[str, Any]]] = None,
    ):
        """Update live task progress."""
        if percent is None:
            if total > 0:
                percent = round(min(100.0, max(0.0, (current / total) * 100.0)), 1)
            else:
                percent = self.progress.get("percent", 0.0)

        self.progress["current"] = current
        self.progress["total"] = total
        self.progress["percent"] = percent
        if message:
            self.progress["message"] = message
        if step:
            self.progress["step"] = step
        if stats:
            self.progress["stats"].update(stats)
        if live_events is not None:
            self.progress["live_events"] = live_events

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task state for JSON API."""
        return {
            "id": self.id,
            "task_type": self.task_type,
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": (
                round((self.completed_at or time.time()) - (self.started_at or self.created_at), 1)
            ),
            "progress": self.progress,
            "results": self.results,
            "error": self.error,
        }


class TaskManager:
    """Manages active and historical background tasks."""

    def __init__(self, max_history: int = 20):
        self._tasks: Dict[str, Task] = {}
        self._active_task_id: Optional[str] = None
        self._lock = threading.Lock()
        self._max_history = max_history

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve task by its ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def get_active_task(self) -> Optional[Task]:
        """Return the currently running task, if any."""
        with self._lock:
            if self._active_task_id:
                task = self._tasks.get(self._active_task_id)
                if task and task.status == "running":
                    return task
                self._active_task_id = None
            return None

    def get_last_completed_task(self, max_age_seconds: float = 60.0) -> Optional[Task]:
        """Return the most recently completed task if within max_age_seconds."""
        with self._lock:
            completed = [
                t for t in self._tasks.values()
                if t.status in ("completed", "failed") and t.completed_at
            ]
            if not completed:
                return None
            completed.sort(key=lambda t: t.completed_at or 0, reverse=True)
            last = completed[0]
            if (time.time() - (last.completed_at or 0)) <= max_age_seconds:
                return last
            return None

    def start_task(
        self,
        task_type: str,
        target_fn: Callable[..., Any],
        metadata: Optional[Dict[str, Any]] = None,
        fn_args: tuple = (),
        fn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Create and launch a background task in a worker thread."""
        with self._lock:
            # Check if another task is already running
            if self._active_task_id:
                active = self._tasks.get(self._active_task_id)
                if active and active.status == "running":
                    logger.warning(
                        f"Cannot start {task_type}: Task {active.id} ({active.task_type}) is already running."
                    )
                    return active

            task_id = f"task_{uuid.uuid4().hex[:8]}"
            task = Task(task_id=task_id, task_type=task_type, metadata=metadata)
            self._tasks[task_id] = task
            self._active_task_id = task_id

            # Trim history
            if len(self._tasks) > self._max_history:
                oldest = sorted(
                    [t for t in self._tasks.values() if t.status != "running"],
                    key=lambda t: t.created_at,
                )
                if oldest:
                    del self._tasks[oldest[0].id]

        def worker():
            task.status = "running"
            task.started_at = time.time()
            logger.info(f"Task {task.id} ({task.task_type}) started.")

            def progress_callback(update: Dict[str, Any]):
                task.update_progress(
                    current=update.get("current", update.get("processed", 0)),
                    total=update.get("total", 0),
                    percent=update.get("percent"),
                    message=update.get("message", ""),
                    step=update.get("step", "running"),
                    stats=update.get("stats", update),
                    live_events=update.get("live_events"),
                )

            kwargs = dict(fn_kwargs or {})
            kwargs["progress_callback"] = progress_callback

            try:
                result = target_fn(*fn_args, **kwargs)
                task.status = "completed"
                task.completed_at = time.time()
                task.results = result
                task.update_progress(
                    percent=100.0,
                    message="Task completed successfully.",
                    step="completed",
                )
                logger.info(f"Task {task.id} completed successfully.")
            except Exception as e:
                task.status = "failed"
                task.completed_at = time.time()
                task.error = str(e)
                task.update_progress(
                    message=f"Task failed: {e}",
                    step="failed",
                )
                logger.exception(f"Task {task.id} failed with error: {e}")
            finally:
                with self._lock:
                    if self._active_task_id == task.id:
                        self._active_task_id = None

        thread = threading.Thread(target=worker, daemon=True, name=f"Worker-{task_id}")
        thread.start()
        return task


# Global singleton instance
task_manager = TaskManager()
