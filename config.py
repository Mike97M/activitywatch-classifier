"""Configuration settings for ActivityWatch Classifier & Active Learning Pipeline."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_STATS_FILE = DATA_DIR / "pipeline_stats.json"

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

# Hierarchical Taxonomy: Main Categories & Granular Subcategories
DEFAULT_TAXONOMY: Dict[str, List[str]] = {
    "Development": [
        "Coding & IDE",
        "DevOps & Terminal",
        "Git & Code Review",
        "Documentation & API",
        "Debugging & Testing",
    ],
    "Communication": [
        "Team Chat",
        "Email",
        "Calls & Meetings",
    ],
    "Education & Learning": [
        "Language Learning",
        "Courses & Tutorials",
        "Reading & Reference",
    ],
    "Finance & Banking": [
        "Banking & Payments",
        "Investments & Trading",
    ],
    "Travel & Navigation": [
        "Public Transit & Tickets",
        "Maps & Rides",
        "Car & EV Utilities",
        "Lodging & Trips",
    ],
    "Health & Fitness": [
        "Workouts & Gym",
        "Health Trackers",
    ],
    "Browsing & Research": [
        "Search & Exploration",
        "AI & Research",
        "Technical Articles & Papers",
        "Shopping & E-Commerce",
    ],
    "Productivity & Planning": [
        "Notes & Knowledge Base",
        "Task & Project Management",
        "Docs & Spreadsheets",
    ],
    "Media & Entertainment": [
        "Video & Streaming",
        "Music & Audio",
        "Gaming",
    ],
    "Social Media": [
        "X / Twitter",
        "Reddit & Communities",
        "LinkedIn & Professional",
    ],
    "System & Admin": [
        "OS & Desktop Settings",
        "Hardware & System Monitor",
        "Package Management",
    ],
    "Uncategorized": [
        "General Activity",
    ],
}

DEFAULT_CATEGORIES: List[str] = list(DEFAULT_TAXONOMY.keys())


def flatten_taxonomy(taxonomy: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """Return flat list of 'Category > Subcategory' paths."""
    tax = taxonomy or DEFAULT_TAXONOMY
    items: List[str] = []
    for cat, subcats in tax.items():
        for sub in subcats:
            items.append(f"{cat} > {sub}")
    return items


def split_category_path(path: str) -> Tuple[str, str]:
    """Parse 'Category > Subcategory' string into (category, subcategory)."""
    if " > " in path:
        parts = path.split(" > ", 1)
        return parts[0].strip(), parts[1].strip()
    return path.strip(), "General"


# Active Learning & Fast Classifier Thresholds
# Calibrated for hierarchical category confidence across granular subcategories
DEFAULT_CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.60"))
# Teacher labels below this confidence are used for the current run but never cached/learned:
# a low-confidence LLM guess must not become a permanent 1.0-confidence memory match.
LABEL_MIN_CONFIDENCE = float(os.environ.get("LABEL_MIN_CONFIDENCE", "0.70"))
# Confidence assigned to LLM labels from backends that return no token logprobs (ollama, gemini_api,
# in-process local). The model's self-reported number is ignored: it just echoes the prompt example.
# ponytail: fixed value, not calibrated; set below LABEL_MIN_CONFIDENCE to stop learning from those backends.
LLM_DEFAULT_CONFIDENCE = float(os.environ.get("LLM_DEFAULT_CONFIDENCE", "0.80"))
MODEL_PATH = MODELS_DIR / "fast_classifier.joblib"
TRAINING_DATA_PATH = DATA_DIR / "training_data.jsonl"

# Gemma E2B / LLM Configuration
# Backends supported: 'llama_server', 'local' (in-process), 'ollama', 'gemini_api', 'openai_compatible', 'heuristic'
DEFAULT_GEMMA_BACKEND = os.environ.get("GEMMA_BACKEND", "llama_server")
DEFAULT_GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "google/gemma-4-E2B-it")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_COMPATIBLE_URL = os.environ.get("OPENAI_COMPATIBLE_URL", "http://127.0.0.1:8000/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Local In-Process Gemma 4 E2B GGUF Model Settings
DEFAULT_LOCAL_GGUF_REPO = "unsloth/gemma-4-E2B-it-GGUF"
DEFAULT_LOCAL_GGUF_NAME = "gemma-4-E2B-it-Q4_K_M.gguf"
LIGHT_LOCAL_GGUF_NAME = "gemma-4-E2B-it-Q3_K_M.gguf"
LOCAL_GGUF_PATH = os.environ.get("LOCAL_GGUF_PATH", "")

# Server Web UI
WEB_HOST = os.environ.get("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("WEB_PORT", "5000"))
