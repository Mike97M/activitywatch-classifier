"""FastAPI Backend for ActivityWatch Classifier & Active Learning UI."""

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
    WEB_ALLOWED_HOSTS,
    WEB_HOST,
    WEB_PORT,
)
from gemma_oracle import GemmaOracle, download_gemma_weights
from pipeline import ClassificationPipeline
from task_manager import task_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app")

app = FastAPI(title="ActivityWatch Gemma Classifier", version="1.0.0")

# The UI is same-origin, so no CORS. Reject foreign Host headers (DNS rebinding) and
# cross-site writes (CSRF: a page you visit could otherwise POST new LLM endpoints and run the pipeline).
app.add_middleware(TrustedHostMiddleware, allowed_hosts=WEB_ALLOWED_HOSTS)


@app.middleware("http")
async def reject_cross_origin_writes(request: Request, call_next):
    origin = request.headers.get("origin")
    if (
        request.method not in ("GET", "HEAD", "OPTIONS")
        and origin
        and urlparse(origin).netloc != request.headers.get("host")
    ):
        return JSONResponse({"detail": "Cross-origin request rejected"}, status_code=403)
    return await call_next(request)

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
    limit_per_bucket: Optional[int] = None  # None: all events in time range, else 150
    time_range: Optional[str] = None  # e.g. "today", "24h", "7d", "30d", "all", "count"
    start_time: Optional[str] = None  # ISO timestamp
    end_time: Optional[str] = None    # ISO timestamp
    confidence_threshold: Optional[float] = None
    filter_afk: bool = True
    include_audible: bool = True
    afk_bucket_id: Optional[str] = None
    collect_training_data: bool = True
    max_streams: Optional[int] = None
    force_reclassify: bool = False


class EventCorrectionRequest(BaseModel):
    feature_text: str
    correct_category: str
    correct_subcategory: Optional[str] = None


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

    active_task = task_manager.get_active_task()
    last_task = task_manager.get_last_completed_task(max_age_seconds=60.0)
    active_dict = (
        active_task.to_dict()
        if active_task
        else (last_task.to_dict() if last_task else None)
    )

    from llama_server_manager import llama_server
    target_summary = aw_client.get_bucket_summary(DEFAULT_TARGET_BUCKET_ID)

    return {
        "activitywatch": aw_status,
        "classifier": classifier_stats,
        "gemma_oracle": oracle_status,
        "llama_server": llama_server.get_status(),
        "confidence_threshold": pipeline.confidence_threshold,
        "last_run_stats": pipeline.last_run_stats,
        "target_bucket_summary": target_summary,
        "active_task": active_dict,
    }


@app.get("/api/bucket-stats")
def get_bucket_stats_api(bucket_id: Optional[str] = None) -> Dict[str, Any]:
    """Return classified activity summary for target bucket."""
    bid = bucket_id or DEFAULT_TARGET_BUCKET_ID
    return aw_client.get_bucket_summary(bid)


@app.get("/api/tasks/active")
def get_active_task_api() -> Dict[str, Any]:
    """Return currently active background task or recently completed task."""
    active = task_manager.get_active_task()
    if active:
        return {"has_active": True, "task": active.to_dict()}
    last = task_manager.get_last_completed_task(max_age_seconds=60.0)
    if last:
        return {"has_active": False, "last_completed": last.to_dict()}
    return {"has_active": False, "task": None}


@app.get("/api/tasks/{task_id}")
def get_task_status_api(task_id: str) -> Dict[str, Any]:
    """Retrieve progress and completion state of a specific background task."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return task.to_dict()


@app.post("/api/download-gemma")
def download_gemma_api(req: DownloadGemmaRequest, sync: bool = False) -> Dict[str, Any]:
    """Download Gemma 4 E2B weights directly from Hugging Face in a background task."""
    repo = DEFAULT_LOCAL_GGUF_REPO
    filename = (
        LIGHT_LOCAL_GGUF_NAME if req.variant == "light" else DEFAULT_LOCAL_GGUF_NAME
    )

    active = task_manager.get_active_task()
    if active and active.task_type == "download_gemma":
        return {
            "status": "already_running",
            "task_id": active.id,
            "task": active.to_dict(),
        }

    def do_download(progress_callback=None):
        if progress_callback:
            progress_callback({
                "step": "downloading",
                "percent": 25.0,
                "message": f"Downloading {filename} from Hugging Face ({repo})...",
            })
        path = download_gemma_weights(repo_id=repo, filename=filename)
        gemma_oracle.local_gguf_path = str(path)
        return {
            "status": "success",
            "path": str(path),
            "local_model": gemma_oracle.get_local_model_info(),
        }

    if sync:
        try:
            return do_download()
        except Exception as e:
            logger.exception("Failed to download Gemma 4 E2B model weights")
            raise HTTPException(status_code=500, detail=f"Download failed: {e}")

    task = task_manager.start_task(
        task_type="download_gemma",
        target_fn=do_download,
        metadata={"variant": req.variant, "filename": filename},
    )
    return {
        "status": "started",
        "task_id": task.id,
        "task": task.to_dict(),
    }


@app.post("/api/llama-server/download")
def download_llama_server_endpoint() -> Dict[str, Any]:
    """Download native llama-server C++ binary if not already present."""
    from llama_server_manager import download_llama_server, llama_server
    bin_path = download_llama_server()
    return {"status": "installed", "binary_path": str(bin_path), "server_status": llama_server.get_status()}


@app.post("/api/llama-server/start")
def start_llama_server_endpoint() -> Dict[str, Any]:
    """Start local llama-server process using available local model."""
    from llama_server_manager import llama_server
    model_path = gemma_oracle.find_local_model_path()
    if not model_path:
        raise HTTPException(status_code=400, detail="No local model found. Download weights first.")
    base_url = llama_server.ensure_server(model_path)
    return {"status": "started", "base_url": base_url, "server_status": llama_server.get_status()}


@app.post("/api/llama-server/stop")
def stop_llama_server_endpoint() -> Dict[str, Any]:
    """Stop local llama-server process."""
    from llama_server_manager import llama_server
    llama_server.stop()
    return {"status": "stopped", "server_status": llama_server.get_status()}


@app.get("/api/buckets")
def get_buckets() -> Dict[str, Any]:
    """List ActivityWatch buckets, detected AFK buckets, and filtered source candidates."""
    try:
        buckets = aw_client.list_buckets()
        afk_buckets = aw_client.list_afk_buckets()
        classified_buckets = {}
        source_buckets = {}

        for bid, binfo in buckets.items():
            is_classified = (
                "classified" in bid.lower()
                or "classifier" in bid.lower()
                or binfo.get("client") in ("aw-gemma-classifier", "aw-classifier")
                or binfo.get("type") == "app.classified.activities"
                or binfo.get("hostname") == "classified"
            )
            is_afk = binfo.get("type") == "afkstatus" or "afk" in bid.lower()

            if is_classified:
                classified_buckets[bid] = binfo
            elif not is_afk:
                source_buckets[bid] = binfo

        return {
            "buckets": buckets,
            "afk_buckets": afk_buckets,
            "classified_buckets": classified_buckets,
            "source_buckets": source_buckets,
        }
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
def run_pipeline(req: RunPipelineRequest, sync: bool = False) -> Dict[str, Any]:
    """Execute non-destructive cloning and active learning classification in a tracked background task."""
    if not req.source_bucket_ids:
        raise HTTPException(status_code=400, detail="Must select at least one source bucket.")

    active_task = task_manager.get_active_task()
    if active_task and active_task.task_type == "pipeline_run":
        return {
            "status": "already_running",
            "task_id": active_task.id,
            "task": active_task.to_dict(),
        }

    if sync:
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
                include_audible=req.include_audible,
                afk_bucket_id=req.afk_bucket_id,
                collect_training_data=req.collect_training_data,
                max_streams=req.max_streams,
                force_reclassify=req.force_reclassify,
            )
            return {"status": "completed", "results": stats}
        except Exception as e:
            logger.exception("Pipeline run encountered an error")
            raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    task = task_manager.start_task(
        task_type="pipeline_run",
        target_fn=pipeline.run,
        metadata={
            "source_bucket_ids": req.source_bucket_ids,
            "target_bucket_id": req.target_bucket_id,
            "limit_per_bucket": req.limit_per_bucket,
            "time_range": req.time_range,
        },
        fn_kwargs={
            "source_bucket_ids": req.source_bucket_ids,
            "target_bucket_id": req.target_bucket_id.strip() or DEFAULT_TARGET_BUCKET_ID,
            "limit_per_bucket": req.limit_per_bucket,
            "time_range": req.time_range,
            "start_time": req.start_time,
            "end_time": req.end_time,
            "confidence_threshold": req.confidence_threshold,
            "filter_afk": req.filter_afk,
            "include_audible": req.include_audible,
            "afk_bucket_id": req.afk_bucket_id,
            "collect_training_data": req.collect_training_data,
            "max_streams": req.max_streams,
            "force_reclassify": req.force_reclassify,
        },
    )

    return {
        "status": "started",
        "task_id": task.id,
        "task": task.to_dict(),
    }


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

    return {
        "status": "trained",
        "changed": changed,
        "classifier_stats": classifier.get_stats(),
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
    uvicorn.run("app:app", host=WEB_HOST, port=WEB_PORT, reload=False)
