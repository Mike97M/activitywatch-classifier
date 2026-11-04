"""Git synchronization utility for training dataset in ActivityWatch Classifier."""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
from config import BASE_DIR, TRAINING_DATA_PATH

logger = logging.getLogger("git_sync")


def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Execute git command in the project repository."""
    return subprocess.run(
        ["git"] + args,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        check=False,
    )


def get_training_data_status(
    data_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Check git status of data/training_data.jsonl and repository."""
    target_path = data_path or TRAINING_DATA_PATH
    try:
        rel_path = str(target_path.relative_to(BASE_DIR))
    except ValueError:
        rel_path = str(target_path)
    
    # 1. Count samples in file
    sample_count = 0
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            sample_count = sum(1 for line in f if line.strip())

    # 2. Check git status
    proc = run_git(["status", "--porcelain", rel_path])
    porcelain = proc.stdout.strip()
    is_modified = bool(porcelain)

    # 3. Last commit touching training_data.jsonl
    log_proc = run_git(["log", "-n", "1", "--oneline", "--", rel_path])
    last_commit = log_proc.stdout.strip() or "No commits yet"
    commit_hash = last_commit.split()[0] if last_commit and last_commit != "No commits yet" else ""

    return {
        "file": rel_path,
        "sample_count": sample_count,
        "is_modified_or_untracked": is_modified,
        "status_code": porcelain[:2] if porcelain else "clean",
        "last_commit": last_commit,
        "commit_hash": commit_hash,
    }


def commit_training_data(
    message: Optional[str] = None,
    author: Optional[str] = None,
    data_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Stage data/training_data.jsonl and commit it to git if there are changes."""
    target_path = data_path or TRAINING_DATA_PATH
    try:
        rel_path = str(target_path.relative_to(BASE_DIR))
    except ValueError:
        rel_path = str(target_path)

    if not target_path.exists():
        return {
            "success": False,
            "committed": False,
            "message": f"{rel_path} does not exist.",
        }

    # Count samples
    with open(target_path, "r", encoding="utf-8") as f:
        sample_count = sum(1 for line in f if line.strip())

    # Stage the training data file
    add_proc = run_git(["add", rel_path])
    if add_proc.returncode != 0:
        logger.error(f"git add failed: {add_proc.stderr}")
        return {"success": False, "committed": False, "error": add_proc.stderr}

    # Check if there is anything staged to commit
    diff_proc = run_git(["diff", "--cached", "--name-only", rel_path])
    staged_files = diff_proc.stdout.strip()

    if not staged_files:
        return {
            "success": True,
            "committed": False,
            "message": "Training data is already up to date in Git.",
            "sample_count": sample_count,
        }

    # Prepare commit message
    if not message:
        message = f"data: update training dataset ({sample_count} samples)"

    commit_args = ["commit", "-m", message, "--", rel_path]
    commit_proc = run_git(commit_args)

    if commit_proc.returncode != 0:
        logger.error(f"git commit failed: {commit_proc.stderr}")
        return {"success": False, "committed": False, "error": commit_proc.stderr}

    # Get new commit hash
    rev_proc = run_git(["rev-parse", "--short", "HEAD"])
    commit_hash = rev_proc.stdout.strip()

    logger.info(f"Committed training data to git: [{commit_hash}] {message}")
    return {
        "success": True,
        "committed": True,
        "commit_hash": commit_hash,
        "message": message,
        "sample_count": sample_count,
    }
