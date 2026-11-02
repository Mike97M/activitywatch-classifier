"""Gemma E2B Oracle: High-intelligence active classifier with hierarchical subcategories."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx

from config import (
    DEFAULT_GEMMA_BACKEND,
    DEFAULT_GEMMA_MODEL,
    DEFAULT_LOCAL_GGUF_NAME,
    DEFAULT_LOCAL_GGUF_REPO,
    DEFAULT_TAXONOMY,
    GEMINI_API_KEY,
    LIGHT_LOCAL_GGUF_NAME,
    LOCAL_GGUF_PATH,
    MODELS_DIR,
    OLLAMA_HOST,
    OPENAI_API_KEY,
    OPENAI_COMPATIBLE_URL,
    flatten_taxonomy,
    split_category_path,
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
    """Invoked when the fast classifier has low confidence, serving as the hierarchical teacher."""

    def __init__(
        self,
        backend: str = DEFAULT_GEMMA_BACKEND,
        model_name: str = DEFAULT_GEMMA_MODEL,
        taxonomy: Optional[Dict[str, List[str]]] = None,
        categories: Optional[List[str]] = None,
        ollama_host: str = OLLAMA_HOST,
        gemini_api_key: str = GEMINI_API_KEY,
        openai_url: str = OPENAI_COMPATIBLE_URL,
        openai_key: str = OPENAI_API_KEY,
        local_gguf_path: Optional[str] = None,
    ):
        self.backend = backend.lower()
        self.model_name = model_name
        self.taxonomy = taxonomy or dict(DEFAULT_TAXONOMY)
        self.categories = categories or list(self.taxonomy.keys())
        self.ollama_host = ollama_host
        self.gemini_api_key = gemini_api_key
        self.openai_url = openai_url
        self.openai_key = openai_key
        self.local_gguf_path = local_gguf_path or LOCAL_GGUF_PATH or None
        self._llama_model = None

    def find_local_model_path(self) -> Optional[Path]:
        """Locate downloaded GGUF weights in models directory or custom path."""
        if self.local_gguf_path:
            p = Path(self.local_gguf_path)
            if p.exists() and p.is_file():
                return p

        if MODELS_DIR.exists():
            for candidate_name in [DEFAULT_LOCAL_GGUF_NAME, LIGHT_LOCAL_GGUF_NAME]:
                cand = MODELS_DIR / candidate_name
                if cand.exists() and cand.is_file():
                    return cand

            ggufs = list(MODELS_DIR.glob("*.gguf"))
            if ggufs:
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
        taxonomy: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Classify an activity into Category and Subcategory."""
        tax = taxonomy or self.taxonomy
        prompt = self._build_prompt(app, title, url, tax)

        # Dispatch based on backend
        if self.backend in ["local", "llama_cpp"]:
            result = self._call_llama_cpp(prompt, tax)
        elif self.backend == "ollama":
            result = self._call_ollama(prompt, tax)
        elif self.backend == "gemini_api":
            result = self._call_gemini_api(prompt, tax)
        elif self.backend == "openai_compatible":
            result = self._call_openai_compatible(prompt, tax)
        else:
            result = self._smart_heuristic_classify(app, title, url, tax)

        # Fallback to smart heuristic if LLM call failed or produced invalid category
        if not result or result.get("category") not in tax:
            heuristic_fallback = self._smart_heuristic_classify(app, title, url, tax)
            heuristic_fallback["reasoning"] = (
                f"{heuristic_fallback.get('reasoning', '')} (Fallback from {self.backend})"
            )
            return heuristic_fallback

        return result

    def _build_prompt(
        self,
        app: str,
        title: str,
        url: Optional[str],
        taxonomy: Dict[str, List[str]],
    ) -> str:
        tax_lines = []
        for cat, subcats in taxonomy.items():
            sub_str = ", ".join(subcats)
            tax_lines.append(f"- {cat}: [{sub_str}]")
        taxonomy_text = "\n".join(tax_lines)

        return (
            f"You are Gemma E2B, an expert activity classifier for ActivityWatch.\n"
            f"Task: Categorize the following user activity into exactly ONE category and a matching subcategory from this taxonomy:\n\n"
            f"{taxonomy_text}\n\n"
            f"Activity Details:\n"
            f"- Application: {app or 'unknown'}\n"
            f"- Window / Tab Title: {title or 'unknown'}\n"
            f"- URL: {url or 'none'}\n\n"
            f"Respond with ONLY a valid JSON object in this exact schema, with no markdown fences:\n"
            f'{{"category": "<Category>", "subcategory": "<Subcategory>", "confidence": 0.95, "reasoning": "<1-sentence reasoning>"}}'
        )

    def _parse_llm_json(
        self, raw_text: str, taxonomy: Dict[str, List[str]]
    ) -> Optional[Dict[str, Any]]:
        """Clean and extract json payload from LLM generation."""
        try:
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
            match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)

            data = json.loads(cleaned.strip())
            category_raw = str(data.get("category", "")).strip()
            subcategory_raw = str(data.get("subcategory", "")).strip()

            # Match category (case-insensitive)
            matched_cat = next(
                (c for c in taxonomy.keys() if c.lower() == category_raw.lower()), None
            )
            if not matched_cat:
                return None

            # Match subcategory within category
            valid_subs = taxonomy[matched_cat]
            matched_sub = next(
                (s for s in valid_subs if s.lower() == subcategory_raw.lower()), None
            )
            if not matched_sub:
                # Substring match or default to first
                matched_sub = next(
                    (s for s in valid_subs if subcategory_raw.lower() in s.lower()),
                    valid_subs[0] if valid_subs else "General",
                )

            full_cat = f"{matched_cat} > {matched_sub}"
            return {
                "category": matched_cat,
                "subcategory": matched_sub,
                "full_category": full_cat,
                "confidence": float(data.get("confidence", 0.95)),
                "reasoning": data.get("reasoning", "Classified by Gemma E2B"),
                "backend": self.backend,
            }
        except Exception as e:
            logger.warning(f"Error parsing LLM JSON: {e}. Raw was: {raw_text[:200]}")
        return None

    def _call_llama_cpp(self, prompt: str, taxonomy: Dict[str, List[str]]) -> Dict[str, Any]:
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
            parsed = self._parse_llm_json(text, taxonomy)
            if parsed:
                parsed["backend"] = f"local_gguf ({model_path.name})"
                return parsed
        except Exception as e:
            logger.error(f"Local llama-cpp execution error: {e}")
        return {}

    def _call_ollama(self, prompt: str, taxonomy: Dict[str, List[str]]) -> Dict[str, Any]:
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
                parsed = self._parse_llm_json(response_text, taxonomy)
                if parsed:
                    return parsed
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
        return {}

    def _call_gemini_api(self, prompt: str, taxonomy: Dict[str, List[str]]) -> Dict[str, Any]:
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
                parsed = self._parse_llm_json(text, taxonomy)
                if parsed:
                    return parsed
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
        return {}

    def _call_openai_compatible(
        self, prompt: str, taxonomy: Dict[str, List[str]]
    ) -> Dict[str, Any]:
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
                parsed = self._parse_llm_json(content, taxonomy)
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
        taxonomy: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Granular heuristic classifier outputting Category and Subcategory."""
        combined = f"{app} {title} {url or ''}".lower()

        def pick_sub(cat: str, target_sub: str) -> str:
            subs = taxonomy.get(cat, ["General"])
            for s in subs:
                if target_sub.lower() in s.lower() or s.lower() in target_sub.lower():
                    return s
            return subs[0] if subs else "General"

        category = "Uncategorized"
        subcategory = "General Activity"
        reason = "Matched by domain heuristic"

        # 1. Social Media
        if any(sig in combined for sig in ["x.com", "twitter"]):
            category = "Social Media"
            subcategory = pick_sub("Social Media", "X / Twitter")
            reason = "Detected X / Twitter microblogging"
        elif any(sig in combined for sig in ["reddit"]):
            category = "Social Media"
            subcategory = pick_sub("Social Media", "Reddit & Communities")
            reason = "Detected Reddit community browsing"
        elif any(sig in combined for sig in ["linkedin"]):
            category = "Social Media"
            subcategory = pick_sub("Social Media", "LinkedIn & Professional")
            reason = "Detected professional networking"

        # 2. Media & Entertainment
        elif any(sig in combined for sig in ["youtube", "netflix", "twitch", "vlc", "mpv", "prime video"]):
            category = "Media & Entertainment"
            subcategory = pick_sub("Media & Entertainment", "Video & Streaming")
            reason = "Detected streaming video or media playback"
        elif any(sig in combined for sig in ["spotify", "music", "podcast", "soundcloud"]):
            category = "Media & Entertainment"
            subcategory = pick_sub("Media & Entertainment", "Music & Audio")
            reason = "Detected audio or music streaming"
        elif any(sig in combined for sig in ["steam", "lutris", "heroic", "game", "retroarch"]):
            category = "Media & Entertainment"
            subcategory = pick_sub("Media & Entertainment", "Gaming")
            reason = "Detected gaming platform or application"

        # 3. Communication
        elif any(sig in combined for sig in ["mail", "thunderbird", "outlook", "gmail", "inbox"]):
            category = "Communication"
            subcategory = pick_sub("Communication", "Email")
            reason = "Detected email client or inbox"
        elif any(sig in combined for sig in ["zoom", "meet", "teams meeting", "call"]):
            category = "Communication"
            subcategory = pick_sub("Communication", "Calls & Meetings")
            reason = "Detected audio/video conference"
        elif any(sig in combined for sig in ["slack", "discord", "telegram", "signal", "whatsapp", "teams"]):
            category = "Communication"
            subcategory = pick_sub("Communication", "Team Chat")
            reason = "Detected team or personal instant messaging"

        # 4. Development
        elif any(sig in combined for sig in ["terminal", "bash", "zsh", "foot", "alacritty", "kitty", "tmux", "sshfs", "ssh"]):
            category = "Development"
            subcategory = pick_sub("Development", "DevOps & Terminal")
            reason = "Detected shell terminal, SSH session, or command-line environment"
        elif any(sig in combined for sig in ["git", "github", "gitlab", "pull request", "commit", "diff", "bitbucket"]):
            category = "Development"
            subcategory = pick_sub("Development", "Git & Code Review")
            reason = "Detected version control or repository code review"
        elif any(sig in combined for sig in ["stackoverflow", "devdocs", "api reference", "documentation", "man "]):
            category = "Development"
            subcategory = pick_sub("Development", "Documentation & API")
            reason = "Detected programming documentation or StackOverflow"
        elif any(sig in combined for sig in ["code", "vscode", "vscodium", "vim", "nvim", "pycharm", "rust", "cargo", "python", "serpextractor"]):
            category = "Development"
            subcategory = pick_sub("Development", "Coding & IDE")
            reason = "Detected source code editor or software engineering"

        # 5. Productivity & Planning
        elif any(sig in combined for sig in ["notion", "obsidian", "logseq", "wiki"]):
            category = "Productivity & Planning"
            subcategory = pick_sub("Productivity & Planning", "Notes & Knowledge Base")
            reason = "Detected personal knowledge base or note-taking app"
        elif any(sig in combined for sig in ["todoist", "jira", "trello", "linear", "task"]):
            category = "Productivity & Planning"
            subcategory = pick_sub("Productivity & Planning", "Task & Project Management")
            reason = "Detected project management or issue tracker"
        elif any(sig in combined for sig in ["sheets", "calc", "excel", "docs", "writer", "libreoffice", "word"]):
            category = "Productivity & Planning"
            subcategory = pick_sub("Productivity & Planning", "Docs & Spreadsheets")
            reason = "Detected office document or spreadsheet"

        # 6. System & Admin
        elif any(sig in combined for sig in ["settings", "control", "gnome-control", "yast", "preferences"]):
            category = "System & Admin"
            subcategory = pick_sub("System & Admin", "OS & Desktop Settings")
            reason = "Detected system settings or preferences"
        elif any(sig in combined for sig in ["btop", "htop", "system monitor", "task manager"]):
            category = "System & Admin"
            subcategory = pick_sub("System & Admin", "Hardware & System Monitor")
            reason = "Detected system resource monitor"
        elif any(sig in combined for sig in ["zypper", "pacman", "apt", "dnf", "package"]):
            category = "System & Admin"
            subcategory = pick_sub("System & Admin", "Package Management")
            reason = "Detected software package manager"

        # 7. Browsing & Research
        elif url or "browser" in app.lower():
            if any(sig in combined for sig in ["arxiv", "paper", "research", "rfc", "whitepaper"]):
                category = "Browsing & Research"
                subcategory = pick_sub("Browsing & Research", "Technical Articles & Papers")
                reason = "Detected research paper or technical publication"
            elif any(sig in combined for sig in ["amazon", "ebay", "cart", "checkout", "store", "shop"]):
                category = "Browsing & Research"
                subcategory = pick_sub("Browsing & Research", "Shopping & E-Commerce")
                reason = "Detected online shopping or retail website"
            else:
                category = "Browsing & Research"
                subcategory = pick_sub("Browsing & Research", "Search & Exploration")
                reason = "General web search and page exploration"
        else:
            category = "Uncategorized"
            subcategory = pick_sub("Uncategorized", "General Activity")
            reason = "General desktop window activity"

        full_cat = f"{category} > {subcategory}"
        return {
            "category": category,
            "subcategory": subcategory,
            "full_category": full_cat,
            "confidence": 0.96,
            "reasoning": reason,
            "backend": "gemma_heuristic_proxy",
        }
