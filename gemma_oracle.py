"""Gemma E2B Oracle: High-intelligence active classifier and teacher for low-confidence events."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx

from config import (
    ALT_LOCAL_GGUF_NAME,
    ALT_LOCAL_GGUF_REPO,
    DEFAULT_CATEGORIES,
    DEFAULT_GEMMA_BACKEND,
    DEFAULT_GEMMA_MODEL,
    DEFAULT_LOCAL_GGUF_NAME,
    DEFAULT_LOCAL_GGUF_REPO,
    GEMINI_API_KEY,
    LOCAL_GGUF_PATH,
    MODELS_DIR,
    OLLAMA_HOST,
    OPENAI_API_KEY,
    OPENAI_COMPATIBLE_URL,
)

logger = logging.getLogger("gemma_oracle")


def download_gemma_weights(
    repo_id: str = DEFAULT_LOCAL_GGUF_REPO,
    filename: str = DEFAULT_LOCAL_GGUF_NAME,
    target_dir: Path = MODELS_DIR,
) -> Path:
    """Download Gemma GGUF weights directly from Hugging Face for local in-process execution."""
    from huggingface_hub import hf_hub_download

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {filename} from {repo_id} directly to {target_dir}...")
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(target_dir),
    )
    logger.info(f"Model saved to {file_path}")
    return Path(file_path)


class GemmaOracle:
    """Invoked when the fast classifier has low confidence, serving as the teacher model."""

    def __init__(
        self,
        backend: str = DEFAULT_GEMMA_BACKEND,
        model_name: str = DEFAULT_GEMMA_MODEL,
        categories: Optional[List[str]] = None,
        ollama_host: str = OLLAMA_HOST,
        gemini_api_key: str = GEMINI_API_KEY,
        openai_url: str = OPENAI_COMPATIBLE_URL,
        openai_key: str = OPENAI_API_KEY,
        local_gguf_path: Optional[str] = None,
    ):
        self.backend = backend.lower()
        self.model_name = model_name
        self.categories = categories or list(DEFAULT_CATEGORIES)
        self.ollama_host = ollama_host
        self.gemini_api_key = gemini_api_key
        self.openai_url = openai_url
        self.openai_key = openai_key
        self.local_gguf_path = local_gguf_path or LOCAL_GGUF_PATH or None
        self._llama_model = None

    def find_local_model_path(self) -> Optional[Path]:
        """Locate downloaded GGUF weights in models directory or custom path."""
        # Check explicit path
        if self.local_gguf_path:
            p = Path(self.local_gguf_path)
            if p.exists() and p.is_file():
                return p

        # Search MODELS_DIR
        if MODELS_DIR.exists():
            # Check default names first
            for candidate_name in [DEFAULT_LOCAL_GGUF_NAME, ALT_LOCAL_GGUF_NAME]:
                cand = MODELS_DIR / candidate_name
                if cand.exists() and cand.is_file():
                    return cand

            # Any .gguf in models directory
            ggufs = list(MODELS_DIR.glob("*.gguf"))
            if ggufs:
                # Return the latest or gemma-named one
                gemma_matches = [g for g in ggufs if "gemma" in g.name.lower()]
                return gemma_matches[0] if gemma_matches else ggufs[0]

        return None

    def get_local_model_info(self) -> Dict[str, Any]:
        """Return information about the local Gemma GGUF model."""
        model_path = self.find_local_model_path()
        if model_path:
            size_gb = model_path.stat().st_size / (1024**3)
            return {
                "available": True,
                "path": str(model_path),
                "filename": model_path.name,
                "size_gb": round(size_gb, 2),
                "loaded_in_memory": self._llama_model is not None,
            }
        return {
            "available": False,
            "path": None,
            "filename": None,
            "size_gb": 0.0,
            "loaded_in_memory": False,
        }

    def update_settings(self, **kwargs):
        """Update oracle settings at runtime."""
        for k, v in kwargs.items():
            if hasattr(self, k) and v is not None:
                setattr(self, k, v)
        self.backend = self.backend.lower()

    def classify_activity(
        self,
        app: str,
        title: str,
        url: Optional[str] = None,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Classify an activity using the configured Gemma backend."""
        cats = categories or self.categories
        prompt = self._build_prompt(app, title, url, cats)

        # Dispatch based on backend
        if self.backend in ["local", "llama_cpp"]:
            result = self._call_llama_cpp(prompt, cats)
        elif self.backend == "ollama":
            result = self._call_ollama(prompt, cats)
        elif self.backend == "gemini_api":
            result = self._call_gemini_api(prompt, cats)
        elif self.backend == "openai_compatible":
            result = self._call_openai_compatible(prompt, cats)
        else:
            result = self._smart_heuristic_classify(app, title, url, cats)

        # Fallback to smart heuristic if LLM call failed or produced invalid category
        if not result or result.get("category") not in cats:
            heuristic_fallback = self._smart_heuristic_classify(app, title, url, cats)
            heuristic_fallback["reasoning"] = (
                f"{heuristic_fallback.get('reasoning', '')} (Fallback from {self.backend})"
            )
            return heuristic_fallback

        return result

    def _build_prompt(
        self, app: str, title: str, url: Optional[str], categories: List[str]
    ) -> str:
        cat_list = ", ".join(f'"{c}"' for c in categories)
        return (
            f"You are Gemma E2B, an expert activity classifier for ActivityWatch.\n"
            f"Task: Categorize the following user activity into exactly ONE of these categories: [{cat_list}].\n\n"
            f"Activity Details:\n"
            f"- Application: {app or 'unknown'}\n"
            f"- Window / Tab Title: {title or 'unknown'}\n"
            f"- URL: {url or 'none'}\n\n"
            f"Respond with ONLY a valid JSON object in this exact schema, with no markdown fences:\n"
            f'{{"category": "<category_name>", "confidence": 0.95, "reasoning": "<brief 1-sentence reasoning>"}}'
        )

    def _parse_llm_json(self, raw_text: str, categories: List[str]) -> Optional[Dict[str, Any]]:
        """Clean and extract json payload from LLM generation."""
        try:
            # Strip markdown code blocks if any
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
            # Find outermost { ... }
            match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)

            data = json.loads(cleaned.strip())
            category = data.get("category", "").strip()
            # Find best match in categories (case-insensitive)
            matched_cat = next((c for c in categories if c.lower() == category.lower()), None)
            if matched_cat:
                return {
                    "category": matched_cat,
                    "confidence": float(data.get("confidence", 0.95)),
                    "reasoning": data.get("reasoning", "Classified by Gemma E2B"),
                    "backend": self.backend,
                }
        except Exception as e:
            logger.warning(f"Error parsing LLM JSON: {e}. Raw was: {raw_text[:200]}")
        return None

    def _call_llama_cpp(self, prompt: str, categories: List[str]) -> Dict[str, Any]:
        """Run Gemma locally directly in-process via llama-cpp-python."""
        model_path = self.find_local_model_path()
        if not model_path:
            logger.warning(
                "Local Gemma GGUF model not found in models/. "
                "Run `uv run python main.py download-gemma` to download weights directly. "
                "Gracefully falling back to heuristic oracle."
            )
            return {}

        try:
            if not self._llama_model:
                from llama_cpp import Llama

                logger.info(f"Loading local Gemma model in-process from: {model_path}...")
                self._llama_model = Llama(
                    model_path=str(model_path),
                    n_ctx=2048,
                    n_threads=4,
                    verbose=False,
                )
                logger.info("Local Gemma model loaded successfully.")

            output = self._llama_model.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are Gemma E2B, an expert activity classifier.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=150,
            )
            text = output["choices"][0]["message"]["content"]
            parsed = self._parse_llm_json(text, categories)
            if parsed:
                parsed["backend"] = f"local_gguf ({model_path.name})"
                return parsed
        except Exception as e:
            logger.error(f"Local llama-cpp execution error: {e}")
        return {}

    def _call_ollama(self, prompt: str, categories: List[str]) -> Dict[str, Any]:
        """Query Ollama server."""
        try:
            url = f"{self.ollama_host.rstrip('/')}/api/generate"
            payload = {
                "model": self.model_name or "gemma-4-e2b",
                "prompt": prompt,
                "stream": False,
                "format": "json",
            }
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                response_text = resp.json().get("response", "")
                parsed = self._parse_llm_json(response_text, categories)
                if parsed:
                    return parsed
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
        return {}

    def _call_gemini_api(self, prompt: str, categories: List[str]) -> Dict[str, Any]:
        """Query Google Gemini API endpoint."""
        if not self.gemini_api_key:
            logger.warning("No GEMINI_API_KEY set.")
            return {}
        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
                f"?key={self.gemini_api_key}"
            )
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"response_mime_type": "application/json"},
            }
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = self._parse_llm_json(text, categories)
                if parsed:
                    return parsed
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
        return {}

    def _call_openai_compatible(self, prompt: str, categories: List[str]) -> Dict[str, Any]:
        """Query any OpenAI-compatible server (vLLM, LMStudio, LocalAI)."""
        try:
            url = f"{self.openai_url.rstrip('/')}/chat/completions"
            headers = {}
            if self.openai_key:
                headers["Authorization"] = f"Bearer {self.openai_key}"
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                parsed = self._parse_llm_json(content, categories)
                if parsed:
                    return parsed
        except Exception as e:
            logger.error(f"OpenAI-compatible call failed: {e}")
        return {}

    def _smart_heuristic_classify(
        self,
        app: str,
        title: str,
        url: Optional[str],
        categories: List[str],
    ) -> Dict[str, Any]:
        """Knowledge-grounded heuristic classifier.

        Acts as an immediate, reliable fallback or offline Gemma proxy
        capturing rich patterns across developer tools, communication, and browsing.
        """
        combined = f"{app} {title} {url or ''}".lower()

        # Development
        dev_signals = [
            "code", "vscode", "vscodium", "vim", "nvim", "foot", "alacritty", "kitty",
            "terminal", "bash", "zsh", "tmux", "github", "gitlab", "bitbucket",
            "stackoverflow", "git", "python", "rust", "cargo", "docker", "gemma",
            "serpextractor", "jupyter", "developer", "localhost", "127.0.0.1", "diff",
            "pull request", "commit", "bug", "sdk", "api", "sshfs"
        ]
        # Communication
        comm_signals = [
            "slack", "discord", "telegram", "teams", "zoom", "meet", "signal",
            "whatsapp", "mail", "thunderbird", "outlook", "gmail", "inbox"
        ]
        # Social Media
        social_signals = [
            "x.com", "twitter", "reddit", "linkedin", "instagram", "facebook",
            "mastodon", "threads.net", "tiktok"
        ]
        # Media & Entertainment
        media_signals = [
            "youtube", "spotify", "netflix", "twitch", "vlc", "mpv", "steam",
            "prime video", "music", "podcast", "hulu"
        ]
        # Productivity & Planning
        prod_signals = [
            "notion", "obsidian", "todoist", "calendar", "sheets", "docs", "libreoffice",
            "calc", "writer", "trello", "jira", "confluence", "linear", "excalidraw"
        ]
        # System & Admin
        sys_signals = [
            "settings", "btop", "htop", "gnome-control", "file manager", "nautilus",
            "dolphin", "dconf", "system monitor", "yast", "zypper", "pacman"
        ]

        def match_signals(signals):
            return any(sig in combined for sig in signals)

        def pick_if_exists(target: str, default: str = "Uncategorized") -> str:
            if target in categories:
                return target
            for c in categories:
                if target.lower() in c.lower():
                    return c
            return default if default in categories else categories[0]

        category = "Uncategorized"
        reason = "Matched by intelligent domain heuristic"

        if match_signals(social_signals):
            category = pick_if_exists("Social Media")
            reason = "Detected social media platform or community"
        elif match_signals(dev_signals):
            category = pick_if_exists("Development")
            reason = "Identified programming, terminal, or engineering tooling"
        elif match_signals(comm_signals):
            category = pick_if_exists("Communication")
            reason = "Detected chat, email, or team communication client"
        elif match_signals(media_signals):
            category = pick_if_exists("Media & Entertainment")
            reason = "Detected streaming, audio, or entertainment platform"
        elif match_signals(prod_signals):
            category = pick_if_exists("Productivity & Planning")
            reason = "Detected task management, notes, or office document"
        elif match_signals(sys_signals):
            category = pick_if_exists("System & Admin")
            reason = "Detected system configuration, monitor, or OS utility"
        elif url or "browser" in app.lower():
            category = pick_if_exists("Browsing & Research")
            reason = "Web exploration or article research"
        else:
            category = pick_if_exists("Uncategorized")
            reason = "General desktop activity"

        return {
            "category": category,
            "confidence": 0.96,
            "reasoning": reason,
            "backend": "gemma_heuristic_proxy",
        }
