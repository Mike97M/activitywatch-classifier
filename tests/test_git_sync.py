"""Unit tests for Git dataset synchronization."""

from pathlib import Path
import tempfile
import git_sync
from config import TRAINING_DATA_PATH


def test_get_training_data_status():
    status = git_sync.get_training_data_status()
    assert "file" in status
    assert "sample_count" in status
    assert "is_modified_or_untracked" in status
    assert "status_code" in status
    assert "last_commit" in status
    assert isinstance(status["sample_count"], int)
    assert status["sample_count"] > 0


def test_git_sync_nonexistent_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_path = Path(tmpdir) / "nonexistent.jsonl"
        res = git_sync.commit_training_data(data_path=fake_path)
        assert res["success"] is False
        assert "does not exist" in res["message"]


def test_git_sync_already_clean_or_commit():
    # If training data is gitignored (privacy mode), it should not be committed to Git
    status = git_sync.get_training_data_status()
    res = git_sync.commit_training_data()
    assert res["success"] is True
    if res.get("is_gitignored"):
        assert res["committed"] is False
        assert "gitignored" in res["message"].lower()
    elif not status["is_modified_or_untracked"]:
        assert res["committed"] is False
        assert "already up to date" in res["message"]
    else:
        assert res["committed"] is True
        assert res["commit_hash"] != ""
