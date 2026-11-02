"""Configuration settings for ActivityWatch Classifier & Active Learning Pipeline."""

import os
from pathlib import Path
from typing import List

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ActivityWatch Server Connection
# Auto-detects potential endpoints (VM host, localhost, or env var)
DEFAULT_AW_URL = os.environ.get("AW_SERVER_URL", "http://127.0.0.1:5600")
AW_CANDIDATE_URLS = [
    os.environ.get("AW_SERVER_URL"),
    "http://127.0.0.1:5600",
    "http://localhost:5600",
    "http://host.lima.internal:5600",
    "http://192.168.5.2:5600",
]
AW_CANDIDATE_URLS = [u for u in AW_CANDIDATE_URLS if u]

# Target Bucket defaults
DEFAULT_TARGET_BUCKET_ID = "aw-classifier-categorized"
DEFAULT_TARGET_BUCKET_TYPE = "app.classified.activities"
DEFAULT_CLIENT_NAME = "aw-gemma-classifier"

# Default Taxonomy / Categories
DEFAULT_CATEGORIES: List[str] = [
    "Development",
    "Communication",
    "Browsing & Research",
    "Productivity & Planning",
    "Media & Entertainment",
    "Social Media",
    "System & Admin",
    "Uncategorized",
]

# Active Learning & Fast Classifier Thresholds
DEFAULT_CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.75"))
MODEL_PATH = MODELS_DIR / "fast_classifier.joblib"
TRAINING_DATA_PATH = DATA_DIR / "training_data.jsonl"

# Gemma E2B / LLM Configuration
# Backends supported: 'ollama', 'gemini_api', 'openai_compatible', 'llama_cpp', 'heuristic'
DEFAULT_GEMMA_BACKEND = os.environ.get("GEMMA_BACKEND", "heuristic")
DEFAULT_GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "google/gemma-4-E2B-it")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_COMPATIBLE_URL = os.environ.get("OPENAI_COMPATIBLE_URL", "http://127.0.0.1:8000/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Server Web UI
WEB_HOST = os.environ.get("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("WEB_PORT", "5000"))
