"""FastAPI Backend for ActivityWatch Classifier & Active Learning UI."""

import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import git_sync
from aw_client import AWClient
from classifier import FastClassifier
from config import (
    BASE_DIR,
    DEFAULT_CATEGORIES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_GEMMA_BACKEND,
    DEFAULT_GEMMA_MODEL,
    DEFAULT_LOCAL_GGUF_NAME,
    DEFAULT_LOCAL_GGUF_REPO,
    DEFAULT_TARGET_BUCKET_ID,
    LIGHT_LOCAL_GGUF_NAME,
    WEB_HOST,
    WEB_PORT,
)
from gemma_oracle import GemmaOracle, download_gemma_weights
from pipeline import ClassificationPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app")

app = FastAPI(title="ActivityWatch Gemma Classifier", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Pipeline components
aw_client = AWClient()
classifier = FastClassifier(categories=list(DEFAULT_CATEGORIES))
gemma_oracle = GemmaOracle(
    backend=DEFAULT_GEMMA_BACKEND,
    model_name=DEFAULT_GEMMA_MODEL,
    taxonomy=classifier.taxonomy,
)
pipeline = ClassificationPipeline(
    aw_client=aw_client,
    classifier=classifier,
    gemma_oracle=gemma_oracle,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
)


# Request & Response Schemas
class SettingsUpdateRequest(BaseModel):
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    gemma_backend: Optional[str] = None
    gemma_model: Optional[str] = None
    ollama_host: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openai_url: Optional[str] = None
    openai_key: Optional[str] = None
    local_gguf_path: Optional[str] = None
    categories: Optional[List[str]] = None


class RunPipelineRequest(BaseModel):
    source_bucket_ids: List[str]
    target_bucket_id: str = DEFAULT_TARGET_BUCKET_ID
    limit_per_bucket: int = 150
    time_range: Optional[str] = None  # e.g. "today", "24h", "7d", "30d", "all", "count"
    start_time: Optional[str] = None  # ISO timestamp
    end_time: Optional[str] = None    # ISO timestamp
    confidence_threshold: Optional[float] = None
    filter_afk: bool = True
    afk_bucket_id: Optional[str] = None
    collect_training_data: bool = True
    auto_commit_git: bool = False


class EventCorrectionRequest(BaseModel):
    feature_text: str
    correct_category: str
    correct_subcategory: Optional[str] = None
    auto_commit_git: bool = False


class GitCommitRequest(BaseModel):
    message: Optional[str] = None


class DownloadGemmaRequest(BaseModel):
    variant: str = "standard"  # "standard" (Q4_K_M) or "light" (Q3_K_M)


# API Routes
@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Return system statuses for ActivityWatch, FastClassifier, and Gemma Oracle."""
    aw_status = {"connected": False, "server_url": aw_client.base_url, "info": {}}
    try:
        info = aw_client.get_info()
        aw_status["connected"] = True
        aw_status["info"] = info
    except Exception as e:
        aw_status["error"] = str(e)

    classifier_stats = classifier.get_stats()

    oracle_status = {
        "backend": gemma_oracle.backend,
        "model_name": gemma_oracle.model_name,
        "ollama_host": gemma_oracle.ollama_host,
        "has_gemini_key": bool(gemma_oracle.gemini_api_key),
        "categories": classifier.categories,
        "taxonomy": classifier.taxonomy,
        "local_model": gemma_oracle.get_local_model_info(),
    }

    git_status = git_sync.get_training_data_status()

    return {
        "activitywatch": aw_status,
        "classifier": classifier_stats,
        "gemma_oracle": oracle_status,
        "confidence_threshold": pipeline.confidence_threshold,
        "last_run_stats": pipeline.last_run_stats,
        "git_status": git_status,
    }


@app.get("/api/git/status")
def get_git_status() -> Dict[str, Any]:
    """Return Git sync status of training_data.jsonl."""
    try:
        return git_sync.get_training_data_status()
    except Exception as e:
        logger.exception("Failed to get git status")
        raise HTTPException(status_code=500, detail=f"Git status error: {e}")


@app.post("/api/git/commit")
def commit_git(req: Optional[GitCommitRequest] = None) -> Dict[str, Any]:
    """Manually commit training data to Git repository."""
    try:
        msg = req.message if req else None
        return git_sync.commit_training_data(message=msg)
    except Exception as e:
        logger.exception("Manual git commit failed")
        raise HTTPException(status_code=500, detail=f"Git commit error: {e}")


@app.post("/api/download-gemma")
def download_gemma_api(req: DownloadGemmaRequest) -> Dict[str, Any]:
    """Download Gemma 4 E2B weights directly from Hugging Face for local in-process inference."""
    try:
        repo = DEFAULT_LOCAL_GGUF_REPO
        filename = (
            LIGHT_LOCAL_GGUF_NAME if req.variant == "light" else DEFAULT_LOCAL_GGUF_NAME
        )

        path = download_gemma_weights(repo_id=repo, filename=filename)
        gemma_oracle.local_gguf_path = str(path)
        return {
            "status": "success",
            "path": str(path),
            "local_model": gemma_oracle.get_local_model_info(),
        }
    except Exception as e:
        logger.exception("Failed to download Gemma 4 E2B model weights")
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")


@app.get("/api/buckets")
def get_buckets() -> Dict[str, Any]:
    """List all ActivityWatch buckets and detected AFK buckets."""
    try:
        buckets = aw_client.list_buckets()
        afk_buckets = aw_client.list_afk_buckets()
        return {"buckets": buckets, "afk_buckets": afk_buckets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch buckets: {e}")


@app.post("/api/settings")
def update_settings(req: SettingsUpdateRequest) -> Dict[str, Any]:
    """Update active settings for the pipeline and models."""
    if req.confidence_threshold is not None:
        pipeline.confidence_threshold = req.confidence_threshold

    oracle_kwargs = {}
    if req.gemma_backend is not None:
        oracle_kwargs["backend"] = req.gemma_backend
    if req.gemma_model is not None:
        oracle_kwargs["model_name"] = req.gemma_model
    if req.ollama_host is not None:
        oracle_kwargs["ollama_host"] = req.ollama_host
    if req.gemini_api_key is not None:
        oracle_kwargs["gemini_api_key"] = req.gemini_api_key
    if req.openai_url is not None:
        oracle_kwargs["openai_url"] = req.openai_url
    if req.openai_key is not None:
        oracle_kwargs["openai_key"] = req.openai_key
    if req.local_gguf_path is not None:
        oracle_kwargs["local_gguf_path"] = req.local_gguf_path
    if req.categories is not None and len(req.categories) > 0:
        clean_cats = [c.strip() for c in req.categories if c.strip()]
        classifier.categories = sorted(clean_cats)
        oracle_kwargs["categories"] = classifier.categories

    if oracle_kwargs:
        gemma_oracle.update_settings(**oracle_kwargs)

    return {"status": "updated", "settings": get_status()}


@app.post("/api/run-pipeline")
def run_pipeline(req: RunPipelineRequest) -> Dict[str, Any]:
    """Execute non-destructive cloning and active learning classification."""
    if not req.source_bucket_ids:
        raise HTTPException(status_code=400, detail="Must select at least one source bucket.")

    try:
        stats = pipeline.run(
            source_bucket_ids=req.source_bucket_ids,
            target_bucket_id=req.target_bucket_id.strip() or DEFAULT_TARGET_BUCKET_ID,
            limit_per_bucket=req.limit_per_bucket,
            time_range=req.time_range,
            start_time=req.start_time,
            end_time=req.end_time,
            confidence_threshold=req.confidence_threshold,
            filter_afk=req.filter_afk,
            afk_bucket_id=req.afk_bucket_id,
            collect_training_data=req.collect_training_data,
            auto_commit_git=req.auto_commit_git,
        )
        return {"status": "completed", "results": stats}
    except Exception as e:
        logger.exception("Pipeline run encountered an error")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")


@app.post("/api/correct-event")
def correct_event(req: EventCorrectionRequest) -> Dict[str, Any]:
    """Submit human correction to retrain the fast classifier on the fly with hierarchical taxonomy."""
    if not req.feature_text or not req.correct_category:
        raise HTTPException(status_code=400, detail="Missing feature_text or correct_category")

    changed = classifier.add_training_sample(
        feature_text=req.feature_text,
        category=req.correct_category,
        subcategory=req.correct_subcategory,
        source="human_correction",
        retrain_now=True,
    )

    git_res = None
    if req.auto_commit_git and changed:
        sub_desc = f" > {req.correct_subcategory}" if req.correct_subcategory else ""
        try:
            git_res = git_sync.commit_training_data(
                message=f"data: human correction for '{req.correct_category}{sub_desc}'"
            )
        except Exception as e:
            logger.warning(f"Git auto-commit on correction failed: {e}")

    return {
        "status": "trained",
        "changed": changed,
        "classifier_stats": classifier.get_stats(),
        "git_commit": git_res,
        "git_status": git_sync.get_training_data_status(),
    }


@app.post("/api/sync-categories")
def sync_categories() -> Dict[str, Any]:
    """Sync classifier taxonomy and learned patterns into ActivityWatch native categorization settings."""
    try:
        from category_sync import sync_taxonomy_to_activitywatch
        res = sync_taxonomy_to_activitywatch(aw_client, classifier)
        return {"status": "success", "result": res}
    except Exception as e:
        logger.exception("Failed to sync categories to ActivityWatch")
        raise HTTPException(status_code=500, detail=f"Category sync failed: {e}")


@app.get("/api/events/target")
def get_target_events(
    bucket_id: str = DEFAULT_TARGET_BUCKET_ID, limit: int = 50
) -> Dict[str, Any]:
    """Retrieve recently categorized events from the target bucket."""
    try:
        events = aw_client.get_events(bucket_id, limit=limit)
        return {"bucket_id": bucket_id, "events": events}
    except Exception as e:
        return {"bucket_id": bucket_id, "events": [], "message": str(e)}


# Serve Static UI
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def serve_ui():
    index_file = BASE_DIR / "static" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "ActivityWatch Classifier API is running. UI index.html not yet built."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=WEB_HOST, port=WEB_PORT, reload=True)
