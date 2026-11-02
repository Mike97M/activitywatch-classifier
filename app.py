"""FastAPI Backend for ActivityWatch Classifier & Active Learning UI."""

import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
    DEFAULT_TARGET_BUCKET_ID,
    WEB_HOST,
    WEB_PORT,
)
from gemma_oracle import GemmaOracle
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
    categories=classifier.categories,
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
    confidence_threshold: Optional[float] = None


class EventCorrectionRequest(BaseModel):
    feature_text: str
    correct_category: str


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
    }

    return {
        "activitywatch": aw_status,
        "classifier": classifier_stats,
        "gemma_oracle": oracle_status,
        "confidence_threshold": pipeline.confidence_threshold,
        "last_run_stats": pipeline.last_run_stats,
    }


@app.get("/api/buckets")
def get_buckets() -> Dict[str, Any]:
    """List all ActivityWatch buckets."""
    try:
        buckets = aw_client.list_buckets()
        return {"buckets": buckets}
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
            confidence_threshold=req.confidence_threshold,
        )
        return {"status": "completed", "results": stats}
    except Exception as e:
        logger.exception("Pipeline run encountered an error")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")


@app.post("/api/correct-event")
def correct_event(req: EventCorrectionRequest) -> Dict[str, Any]:
    """Submit human correction to retrain the fast classifier on the fly."""
    if not req.feature_text or not req.correct_category:
        raise HTTPException(status_code=400, detail="Missing feature_text or correct_category")

    classifier.add_training_sample(
        feature_text=req.feature_text,
        category=req.correct_category,
        source="human_correction",
        retrain_now=True,
    )
    return {"status": "trained", "classifier_stats": classifier.get_stats()}


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
