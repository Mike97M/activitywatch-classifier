"""Gemma E2B Oracle: High-intelligence active classifier with hierarchical subcategories."""

import concurrent.futures
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import httpx

from config import (
    DEFAULT_GEMMA_BACKEND,
    DEFAULT_GEMMA_MODEL,
    DEFAULT_LOCAL_GGUF_NAME,
    DEFAULT_LOCAL_GGUF_REPO,
    DEFAULT_TAXONOMY,
    LIGHT_LOCAL_GGUF_NAME,
    LLM_DEFAULT_CONFIDENCE,
    LOCAL_GGUF_PATH,
    MODELS_DIR,
    OPENAI_API_KEY,
    OPENAI_COMPATIBLE_URL,
)
from app_knowledge import app_knowledge

logger = logging.getLogger("gemma_oracle")

# Backend tag on every heuristic result (explicit backend or fallback after an LLM failure).
# Its confidence is hard-coded, so the pipeline never learns from it.
HEURISTIC_BACKEND = "gemma_heuristic_proxy"
BACKENDS = ("llama_server", "openai_compatible")


def _validate_backend(backend: str) -> str:
    name = backend.lower().replace("-", "_")
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Supported: {', '.join(BACKENDS)}")
    return name


def label_confidence_from_logprobs(tokens: Optional[List[Dict[str, Any]]]) -> Optional[float]:
    """P(category value) * P(subcategory value) from OpenAI-style token logprobs.

    Only tokens overlapping the two JSON string values count; the JSON scaffolding is forced
    by the grammar and says nothing about how sure the model is. None if unavailable.
    """
    if not tokens:
        return None
    text, spans = "", []
    for t in tokens:
        tok = t.get("token", "")
        spans.append((len(text), len(text) + len(tok), float(t.get("logprob", 0.0))))
        text += tok
    total = 0.0
    for key in ("category", "subcategory"):
        m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
        if not m:
            return None
        start, end = m.span(1)
        total += sum(lp for a, b, lp in spans if a < end and b > start)
    return math.exp(total)


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
        openai_url: str = OPENAI_COMPATIBLE_URL,
        openai_key: str = OPENAI_API_KEY,
        local_gguf_path: Optional[str] = None,
    ):
        self.backend = _validate_backend(backend)
        self.model_name = model_name
        self.taxonomy = taxonomy or dict(DEFAULT_TAXONOMY)
        self.categories = categories or list(self.taxonomy.keys())
        self.openai_url = openai_url
        self.openai_key = openai_key
        self.local_gguf_path = local_gguf_path or LOCAL_GGUF_PATH or None

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
            }
        return {
            "available": False,
            "path": None,
            "filename": None,
            "size_gb": 0.0,
        }

    def update_settings(self, **kwargs):
        """Update oracle settings at runtime (an unknown backend raises before anything changes)."""
        if kwargs.get("backend") is not None:
            kwargs["backend"] = _validate_backend(kwargs["backend"])
        for k, v in kwargs.items():
            if hasattr(self, k) and v is not None:
                setattr(self, k, v)

    def classify_activity(
        self,
        app: str,
        title: str,
        url: Optional[str] = None,
        taxonomy: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Classify a single activity into Category and Subcategory."""
        tax = taxonomy or self.taxonomy
        res = self.classify_batch([{"app": app, "title": title, "url": url}], tax, max_streams=1)
        return res[0] if res else self._smart_heuristic_classify(app, title, url, tax)

    def classify_batch(
        self,
        activities: List[Dict[str, Any]],
        taxonomy: Optional[Dict[str, List[str]]] = None,
        max_streams: int = 8,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Classify activities in parallel against llama-server or an OpenAI-compatible endpoint.

        Any failure (server down, timeout, invalid label) falls back to the built-in heuristic,
        tagged HEURISTIC_BACKEND so the pipeline never learns from it.
        """
        if not activities:
            return []
        tax = taxonomy or self.taxonomy
        workers = max(1, max_streams)
        base_url = self.openai_url

        if self.backend == "llama_server":
            from llama_server_manager import llama_server

            workers = min(workers, llama_server.n_parallel)
            try:
                model_path = self.find_local_model_path()
                if model_path or llama_server.external_url:
                    base_url = llama_server.ensure_server(model_path)
            except Exception as e:
                logger.error(f"Failed to ensure llama-server: {e}")

        def _eval_one(idx, a):
            app, title, url = a.get("app", ""), a.get("title", ""), a.get("url")
            res = self._call_openai_compatible(self._build_prompt(app, title, url, tax), tax, base_url)
            if not res or res.get("category") not in tax:
                res = self._smart_heuristic_classify(app, title, url, tax)
                res["reasoning"] = f"{res.get('reasoning', '')} (Fallback from {self.backend})"
            else:
                res["backend"] = self.backend
            return idx, res

        results: List[Optional[Dict[str, Any]]] = [None] * len(activities)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_eval_one, i, a) for i, a in enumerate(activities)]
            for done_count, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                i, res = fut.result()
                results[i] = res
                if progress_callback:
                    progress_callback(done_count, len(activities))
        return results

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

        parsed_domain = ""
        if url:
            try:
                from urllib.parse import urlparse
                test_url = url if "://" in url else f"https://{url}"
                p = urlparse(test_url)
                parsed_domain = p.netloc.lower().split(":")[0]
            except Exception:
                parsed_domain = ""

        return (
            f"You are Gemma E2B, an expert activity classifier for ActivityWatch.\n"
            f"Task: Categorize the following user activity into exactly ONE category and a matching subcategory from this taxonomy:\n\n"
            f"{taxonomy_text}\n\n"
            f"Classification Rules:\n"
            f"1. Terminal emulators (foot, kitty, alacritty, ptyxis, konsole, gnome-terminal, xterm, wezterm, etc.) and shell prompts/sessions ('<user>@<host>', '~', bash, fish, zsh, tmux, ssh, cli commands) MUST be categorized as category 'Development', subcategory 'DevOps & Terminal'. The only exception: system tools running in a terminal (htop, btop, top, and package managers such as zypper, apt, dnf, pacman, flatpak) follow rule 2.\n"
            f"2. 'System & Admin' is reserved strictly for OS system settings dialogs, hardware monitors (htop, btop), local activity trackers (ActivityWatch web UI), and package management tools.\n"
            f"3. For web browser activities, do NOT lazily dump activities into 'Browsing & Research > Search & Exploration'. Categorize based on the website domain, URL path, and actual function:\n"
            f"   - Local dev servers (localhost, 127.0.0.1, 0.0.0.0, dev ports), admin consoles (Django site admin, scraper dashboards, Grafana), code repositories (GitHub, GitLab, 'user/repo' titles) MUST be categorized under 'Development' (DevOps & Terminal, Coding & IDE, or Git & Code Review).\n"
            f"   - AI assistants and LLM chat sites (Claude, ChatGPT, Gemini, Google AI Studio, Perplexity) belong to 'Browsing & Research > AI & Research'.\n"
            f"   - Local activity monitors (ActivityWatch web UI) belong to 'System & Admin > Hardware & System Monitor'.\n"
            f"   - Webmail services and email inboxes accessed in a browser (Gmail, Proton Mail, Mailbox.org, Outlook/Office365, Yahoo Mail, Fastmail, webmail) MUST ALWAYS be categorized as 'Communication > Email'. NEVER categorize web email as 'Browsing & Research'.\n"
            f"   - Project management (Jira, Linear), notes (Notion), and office/spreadsheets (Google Docs/Sheets, Grist) belong to 'Productivity & Planning'.\n"
            f"   - Tech blogs, benchmarks (Artificial Analysis), engineering discussions (Hacker News), financial market charts, and research papers belong to 'Browsing & Research > Technical Articles & Papers'.\n"
            f"   - Online shopping, retail, and store sites belong to 'Browsing & Research > Shopping & E-Commerce'.\n"
            f"   - Blank new browser tabs ('New Tab', 'chrome://newtab', 'about:blank') belong to 'Uncategorized > General Activity'.\n"
            f"   - ONLY actual web search engines (Google Search, DuckDuckGo, Brave Search) or exploratory general browsing belong to 'Browsing & Research > Search & Exploration'.\n\n"
            f"Activity Details:\n"
            f"- Application: {app or 'browser'}{f' ({app_knowledge.get_description(app)})' if app and app_knowledge.get_description(app) else ''}\n"
            f"- Window / Tab Title: {title or 'unknown'}\n"
            f"- Website Domain: {parsed_domain or 'none'}\n"
            f"- URL / Link: {url or 'none'}\n\n"
            f"Respond with ONLY a valid JSON object in this exact schema, with no markdown fences and no extra fields:\n"
            f'{{"category": "<Category>", "subcategory": "<Subcategory>"}}'
        )

    # ------------------------------------------------------------------
    # Constrained decoding
    # ------------------------------------------------------------------
    @staticmethod
    def _gbnf_string(value: str) -> str:
        """GBNF literal matching a JSON string token (including its quotes)."""
        return json.dumps(f'"{value}"')

    def _build_grammar(self, taxonomy: Dict[str, List[str]]) -> str:
        """GBNF grammar that forces valid JSON with a valid category/subcategory pair.

        Removes the need for markdown/preamble tokens and guarantees one of the taxonomy
        pairs, which is both faster and more accurate than free-form generation.
        """
        q_cat = self._gbnf_string("category")
        q_sub = self._gbnf_string("subcategory")

        pairs = []
        for cat, subs in taxonomy.items():
            for sub in subs:
                pairs.append(
                    f'{q_cat} ws ":" ws {self._gbnf_string(cat)} '
                    f'ws "," ws {q_sub} ws ":" ws {self._gbnf_string(sub)}'
                )
        if not pairs:
            return ""

        rules = [
            'ws ::= [ \\t\\n]*',
            "pair ::= " + " | ".join(f"pair{i}" for i in range(len(pairs))),
        ]
        rules.extend(f"pair{i} ::= {pair}" for i, pair in enumerate(pairs))

        rules.append('root ::= "{" ws pair ws "}"')

        return "\n".join(rules)

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
                "confidence": LLM_DEFAULT_CONFIDENCE,
                "reasoning": data.get("reasoning", "Classified by Gemma E2B"),
                "backend": self.backend,
            }
        except Exception as e:
            logger.warning(f"Error parsing LLM JSON: {e}. Raw was: {raw_text[:200]}")
        return None

    def _call_openai_compatible(
        self, prompt: str, taxonomy: Dict[str, List[str]], base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Query any OpenAI-compatible server (llama-server, vLLM, LMStudio, LocalAI)."""
        try:
            base = (base_url or self.openai_url or "http://127.0.0.1:8088/v1").rstrip("/")
            url = f"{base}/chat/completions"
            headers = {}
            if self.openai_key:
                headers["Authorization"] = f"Bearer {self.openai_key}"
            payload = {
                "model": self.model_name or "local",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 48,
                "logprobs": True,
            }
            if self.backend == "llama_server":
                # llama-server accepts GBNF directly; this also avoids its json_schema
                # sampler conflict with --reasoning-budget 0.
                payload["grammar"] = self._build_grammar(taxonomy)
            else:
                payload["response_format"] = {"type": "json_object"}
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = (msg.get("content") or msg.get("reasoning_content") or "").strip()
                parsed = self._parse_llm_json(content, taxonomy)
                if parsed:
                    conf = label_confidence_from_logprobs(
                        (data["choices"][0].get("logprobs") or {}).get("content")
                    )
                    if conf is not None:
                        parsed["confidence"] = round(conf, 4)
                        parsed["confidence_source"] = "logprobs"
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
        """Granular heuristic classifier outputting Category and Subcategory with domain awareness."""
        parsed_domain = ""
        root_domain = ""
        url_clean = (url or "").strip().lower()
        title_clean = (title or "").strip().lower()
        app_clean = (app or "").strip().lower()

        if url_clean:
            try:
                from urllib.parse import urlparse
                test_url = url_clean if "://" in url_clean else f"https://{url_clean}"
                p = urlparse(test_url)
                netloc = p.netloc.lower().split(":")[0]
                parsed_domain = netloc
                parts = netloc.split(".")
                if len(parts) >= 2:
                    root_domain = ".".join(parts[-2:])
                else:
                    root_domain = netloc
            except Exception:
                parsed_domain = ""
                root_domain = ""

        combined = f"{app_clean} {title_clean} {url_clean} {parsed_domain} {root_domain}".lower()

        def pick_sub(cat: str, target_sub: str) -> str:
            subs = taxonomy.get(cat, ["General"])
            for s in subs:
                if target_sub.lower() in s.lower() or s.lower() in target_sub.lower():
                    return s
            return subs[0] if subs else "General"

        # -1. Recognized mobile and desktop apps from app knowledge base
        hint = app_knowledge.get_category_hint(app_clean)
        if hint:
            cat, sub = hint
            picked_sub = pick_sub(cat, sub)
            return {
                "category": cat,
                "subcategory": picked_sub,
                "full_category": f"{cat} > {picked_sub}",
                "confidence": 0.98,
                "reasoning": f"Identified application '{app_clean}' from application knowledge base",
                "backend": HEURISTIC_BACKEND,
            }

        # -1. Blank or New Browser Tab
        if (
            title_clean in ("new tab", "new tab - brave", "new tab — mozilla firefox", "about:blank", "about:newtab", "chrome://newtab", "")
            or title_clean.startswith("new tab - ")
            or title_clean.startswith("new tab — ")
        ) and (not url_clean or url_clean in ("about:blank", "about:newtab", "chrome://newtab/", "chrome://newtab")):
            sub = pick_sub("Uncategorized", "General Activity")
            return {
                "category": "Uncategorized",
                "subcategory": sub,
                "full_category": f"Uncategorized > {sub}",
                "confidence": 0.95,
                "reasoning": "Blank new browser tab or empty page",
                "backend": HEURISTIC_BACKEND,
            }

        # 0. Terminal Emulators & Command-line Shells
        terminal_apps = {"foot", "kitty", "alacritty", "ptyxis", "gnome-terminal", "konsole", "xterm", "wezterm", "urxvt", "terminator"}
        is_terminal = app_clean in terminal_apps or any(term in combined for term in ["bash", "zsh", "fish", "tmux", "sshfs"])
        if is_terminal or re.search(r"@[a-zA-Z0-9_.-]+:[~/\w.-]*", title or "") or title_clean in ("~", "foot"):
            if any(mon in combined for mon in ["btop", "htop", "nvtop", "system-monitor", "sensors"]):
                category = "System & Admin"
                subcategory = pick_sub("System & Admin", "Hardware & System Monitor")
                reason = "Detected hardware system monitor running inside terminal"
            elif any(pkg in combined for pkg in ["zypper", "dnf", "pacman", "apt install", "apt-get"]):
                category = "System & Admin"
                subcategory = pick_sub("System & Admin", "Package Management")
                reason = "Detected package manager command inside terminal"
            else:
                category = "Development"
                subcategory = pick_sub("Development", "DevOps & Terminal")
                reason = "Detected command line shell or terminal emulator activity"
            return {
                "category": category,
                "subcategory": subcategory,
                "full_category": f"{category} > {subcategory}",
                "confidence": 0.96,
                "reasoning": reason,
                "backend": HEURISTIC_BACKEND,
            }

        # 0.5. Local Development Servers, Localhost & ActivityWatch
        is_localhost = any(h in parsed_domain or h in url_clean for h in ["127.0.0.1", "0.0.0.0", "localhost", "lima.internal", ":5000", ":5600", ":8000", ":3000", ":5173", ":8080"])
        if is_localhost:
            if "classifier" in combined:
                category = "Development"
                subcategory = pick_sub("Development", "Coding & IDE")
                reason = "ActivityWatch Classifier active learning UI"
            elif "activitywatch" in combined:
                category = "System & Admin"
                subcategory = pick_sub("System & Admin", "Hardware & System Monitor")
                reason = "ActivityWatch local system activity monitor"
            elif any(adm in combined for adm in ["django", "admin", "phpmyadmin", "grafana", "kibana", "prometheus"]):
                category = "Development"
                subcategory = pick_sub("Development", "DevOps & Terminal")
                reason = "Local administrative interface or dev server dashboard"
            else:
                category = "Development"
                subcategory = pick_sub("Development", "DevOps & Terminal")
                reason = "Local development server or test endpoint"
            return {
                "category": category,
                "subcategory": subcategory,
                "full_category": f"{category} > {subcategory}",
                "confidence": 0.96,
                "reasoning": reason,
                "backend": HEURISTIC_BACKEND,
            }

        # 0.6. Remote Server Dashboards, Scrapers & Infrastructure
        if any(s in combined for s in ["your-server.de", "scraper/dashboard", "hetzner", "cpanel", "portainer", "grafana", "kibana", "prometheus", "sentry.io", "datadog", "digitalocean", "cloudflare"]):
            category = "Development"
            subcategory = pick_sub("Development", "DevOps & Terminal")
            reason = "Server management console, cloud infrastructure, or scraping dashboard"
            return {
                "category": category,
                "subcategory": subcategory,
                "full_category": f"{category} > {subcategory}",
                "confidence": 0.96,
                "reasoning": reason,
                "backend": HEURISTIC_BACKEND,
            }
        elif "django site admin" in combined or "/admin" in url_clean:
            category = "Development"
            subcategory = pick_sub("Development", "DevOps & Terminal")
            reason = "Django web application administrative interface"
            return {
                "category": category,
                "subcategory": subcategory,
                "full_category": f"{category} > {subcategory}",
                "confidence": 0.96,
                "reasoning": reason,
                "backend": HEURISTIC_BACKEND,
            }

        # 0.7. Code Hosting & Git Repositories
        is_git = (
            any(s in combined for s in ["github.com", "gitlab.com", "bitbucket.org", "gitea", "codeberg"])
            or bool(re.match(r"^[a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+(?:\s*[-—|:]\s*.*)?$", title_clean))
        )
        if is_git:
            category = "Development"
            subcategory = pick_sub("Development", "Git & Code Review")
            reason = "Version control repository or code review"
            return {
                "category": category,
                "subcategory": subcategory,
                "full_category": f"{category} > {subcategory}",
                "confidence": 0.96,
                "reasoning": reason,
                "backend": HEURISTIC_BACKEND,
            }

        # 0.8a. AI assistants / LLM chat sites (same as the rules and the LLM prompt)
        if any(s in combined for s in [
            "claude.ai", "chatgpt", "chat.openai.com", "aistudio.google.com", "gemini.google.com",
            "perplexity.ai", "poe.com", "copilot.microsoft.com", "chat.mistral.ai", "chat.deepseek.com",
        ]):
            category = "Browsing & Research"
            subcategory = pick_sub("Browsing & Research", "AI & Research")
            return {
                "category": category,
                "subcategory": subcategory,
                "full_category": f"{category} > {subcategory}",
                "confidence": 0.96,
                "reasoning": "AI assistant or LLM chat site",
                "backend": HEURISTIC_BACKEND,
            }

        # 0.8b. AI developer platforms & notebooks
        is_ai_coding = any(s in combined for s in [
            "platform.openai.com", "huggingface.co", "openrouter.ai", "colab.research.google.com",
            "kaggle.com", "jupyter", "modal.com", "replicate.com", "v0.dev"
        ])
        if is_ai_coding:
            category = "Development"
            subcategory = pick_sub("Development", "Coding & IDE")
            reason = "AI developer assistant, LLM workspace, or interactive notebook"
            return {
                "category": category,
                "subcategory": subcategory,
                "full_category": f"{category} > {subcategory}",
                "confidence": 0.96,
                "reasoning": reason,
                "backend": HEURISTIC_BACKEND,
            }

        # 0.9. Programming Docs & APIs
        is_dev_docs = any(s in combined for s in [
            "stackoverflow.com", "devdocs.io", "developer.mozilla.org", "developer.android.com",
            "readthedocs.io", "crates.io", "pypi.org", "pkg.go.dev", "npmjs.com"
        ]) or any(k in combined for k in ["api reference", "documentation"])
        if is_dev_docs:
            category = "Development"
            subcategory = pick_sub("Development", "Documentation & API")
            reason = "Programming documentation or developer API reference"
            return {
                "category": category,
                "subcategory": subcategory,
                "full_category": f"{category} > {subcategory}",
                "confidence": 0.96,
                "reasoning": reason,
                "backend": HEURISTIC_BACKEND,
            }

        # 0.10. Web Scraping & Crawling Tools
        is_scraper = any(s in combined for s in ["serpextractor.com", "apify.com", "scrape.ly"])
        if is_scraper:
            category = "Development"
            subcategory = pick_sub("Development", "Coding & IDE")
            reason = "Web scraping or data extraction API service"
            return {
                "category": category,
                "subcategory": subcategory,
                "full_category": f"{category} > {subcategory}",
                "confidence": 0.96,
                "reasoning": reason,
                "backend": HEURISTIC_BACKEND,
            }

        # 1. Social Media
        if any(sig in combined for sig in ["x.com", "twitter.com", "t.co"]):
            category = "Social Media"
            subcategory = pick_sub("Social Media", "X / Twitter")
            reason = "Detected X / Twitter microblogging"
        elif any(sig in combined for sig in ["reddit.com", "old.reddit.com"]):
            category = "Social Media"
            subcategory = pick_sub("Social Media", "Reddit & Communities")
            reason = "Detected Reddit community browsing"
        elif any(sig in combined for sig in ["linkedin.com", "xing.com"]):
            category = "Social Media"
            subcategory = pick_sub("Social Media", "LinkedIn & Professional")
            reason = "Detected professional networking"

        # 2. Media & Entertainment
        elif any(sig in combined for sig in ["youtube", "youtu.be", "netflix", "twitch", "vlc", "mpv", "prime video"]):
            category = "Media & Entertainment"
            subcategory = pick_sub("Media & Entertainment", "Video & Streaming")
            reason = "Detected streaming video or media playback"
        elif any(sig in combined for sig in ["spotify", "music.youtube", "soundcloud", "deezer", "apple music", "tidal"]):
            category = "Media & Entertainment"
            subcategory = pick_sub("Media & Entertainment", "Music & Audio")
            reason = "Detected audio or music streaming"
        elif any(sig in combined for sig in ["steam", "lutris", "heroic", "game", "retroarch", "sudoku", "chess", "puzzle"]):
            category = "Media & Entertainment"
            subcategory = pick_sub("Media & Entertainment", "Gaming")
            reason = "Detected gaming platform or application"

        # 3. Communication
        elif any(sig in combined for sig in [
            "gmail", "mail.google", "mailbox.org", "app.mailbox.org", "account.mailbox.org",
            "outlook", "office365", "proton.me", "protonmail", "fastmail", "webmail", "roundcube",
            "thunderbird", "k9mail", "evolution", "kmail", "yahoo mail", "mail.yahoo", "tutanota", "tuta.com", "mail"
        ]):
            category = "Communication"
            subcategory = pick_sub("Communication", "Email")
            reason = "Detected email client or inbox"
        elif any(sig in combined for sig in ["zoom.us", "meet.google", "teams meeting", "calendly.com", "cal.com"]):
            category = "Communication"
            subcategory = pick_sub("Communication", "Calls & Meetings")
            reason = "Detected audio/video conference"
        elif any(sig in combined for sig in ["slack", "discord", "telegram", "signal", "whatsapp", "teams", "matrix", "element"]):
            category = "Communication"
            subcategory = pick_sub("Communication", "Team Chat")
            reason = "Detected team or personal instant messaging"

        # 4. Development
        elif any(sig in combined for sig in ["terminal", "bash", "zsh", "foot", "alacritty", "kitty", "tmux", "sshfs", "ssh"]):
            category = "Development"
            subcategory = pick_sub("Development", "DevOps & Terminal")
            reason = "Detected shell terminal, SSH session, or command-line environment"
        elif any(sig in combined for sig in ["code", "vscode", "vscodium", "vim", "nvim", "pycharm", "rust", "cargo", "python"]):
            category = "Development"
            subcategory = pick_sub("Development", "Coding & IDE")
            reason = "Detected source code editor, AI developer tooling, or programming workspace"

        # 5. Productivity & Planning
        elif any(sig in combined for sig in ["notion.so", "obsidian.md", "logseq.com", "confluence", "wiki", "duolingo"]):
            category = "Productivity & Planning"
            subcategory = pick_sub("Productivity & Planning", "Notes & Knowledge Base")
            reason = "Detected personal knowledge base, learning, or note-taking app"
        elif any(sig in combined for sig in ["jira", "linear.app", "trello.com", "asana.com", "todoist.com"]):
            category = "Productivity & Planning"
            subcategory = pick_sub("Productivity & Planning", "Task & Project Management")
            reason = "Detected project management or issue tracker"
        elif any(sig in combined for sig in ["docs.google.com", "sheets.google.com", "getgrist.com", "airtable.com", "calc", "excel", "docs", "writer", "libreoffice", "word"]):
            category = "Productivity & Planning"
            subcategory = pick_sub("Productivity & Planning", "Docs & Spreadsheets")
            reason = "Detected office document or spreadsheet"

        # 6. System & Admin
        elif any(sig in combined for sig in ["settings", "control", "gnome-control", "yast", "preferences", "launcher", "system launcher", "clock", "alarm", "wellbeing", "shelter"]):
            category = "System & Admin"
            subcategory = pick_sub("System & Admin", "OS & Desktop Settings")
            reason = "Detected system settings, desktop utility, or launcher"
        elif any(sig in combined for sig in ["btop", "htop", "system monitor", "task manager", "activitywatch"]):
            category = "System & Admin"
            subcategory = pick_sub("System & Admin", "Hardware & System Monitor")
            reason = "Detected system resource or activity monitor"
        elif any(sig in combined for sig in ["zypper", "pacman", "apt", "dnf", "package"]):
            category = "System & Admin"
            subcategory = pick_sub("System & Admin", "Package Management")
            reason = "Detected software package manager"

        # 7. Browsing & Research
        elif url_clean or "browser" in app_clean:
            is_tech_article = (
                any(sig in combined for sig in [
                    "arxiv", "paper", "research", "rfc", "whitepaper",
                    "artificialanalysis.ai", "news.ycombinator.com", "hacker news", "lobste.rs",
                    "medium.com", "substack.com", "dev.to", "hashnode", "benchmark", "leaderboard",
                    "coding-agents", "evaluation dataset", "tokenbench",
                    "tradingview.com", "price chart", "stock market", "financial analysis"
                ])
                or any(p in url_clean for p in ["/blog/", "/work/", "/posts/", "/article/", "/tutorial/"])
            )
            is_shopping = any(sig in combined for sig in ["amazon.", "ebay.", "aliexpress.", "finn.no", "nettbutikk", "cart", "checkout", "store", "shop"])
            is_search = (
                (any(s in parsed_domain for s in ["google.", "duckduckgo.", "brave.com", "bing.com", "kagi.com"]) and any(p in url_clean for p in ["/search", "?q="]))
                or any(s in combined for s in ["- google search", "- brave search", "- duckduckgo search", "wikipedia.org", "wikimedia.org"])
            )

            if is_tech_article:
                category = "Browsing & Research"
                subcategory = pick_sub("Browsing & Research", "Technical Articles & Papers")
                reason = "Technical article, research publication, benchmark, or market analysis"
            elif is_shopping:
                category = "Browsing & Research"
                subcategory = pick_sub("Browsing & Research", "Shopping & E-Commerce")
                reason = "Online retail or shopping"
            elif is_search:
                category = "Browsing & Research"
                subcategory = pick_sub("Browsing & Research", "Search & Exploration")
                reason = "Web search engine query or encyclopedia exploration"
            else:
                category = "Browsing & Research"
                subcategory = pick_sub("Browsing & Research", "Search & Exploration")
                reason = "General web browsing and page exploration"
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
            "backend": HEURISTIC_BACKEND,
        }
