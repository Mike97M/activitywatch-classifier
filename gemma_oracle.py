"""Gemma E2B Oracle: High-intelligence active classifier with hierarchical subcategories."""

import json
import logging
import os
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
from app_knowledge import app_knowledge

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
        """Classify a list of activities, using multi-stream parallel decoding with 1 memory pass for local Gemma."""
        if not activities:
            return []
        tax = taxonomy or self.taxonomy
        if self.backend in ["local", "llama_cpp"]:
            chunk_size = min(max(1, max_streams), 12)
            results = []
            for i in range(0, len(activities), chunk_size):
                chunk = activities[i : i + chunk_size]
                if len(chunk) == 1:
                    a = chunk[0]
                    prompt = self._build_prompt(a.get("app", ""), a.get("title", ""), a.get("url"), tax)
                    res = self._call_llama_cpp(prompt, tax)
                    if not res or res.get("category") not in tax:
                        res = self._smart_heuristic_classify(a.get("app", ""), a.get("title", ""), a.get("url"), tax)
                        res["reasoning"] = f"{res.get('reasoning', '')} (Fallback from local Gemma)"
                    results.append(res)
                else:
                    batch_prompt = self._build_batch_prompt(chunk, tax)
                    chunk_res_map = {}
                    try:
                        model_path = self.find_local_model_path()
                        if not model_path:
                            raise FileNotFoundError("Local Gemma model not found")
                        if self._llama_model is None:
                            from llama_cpp import Llama
                            logger.info(f"Loading local Gemma model in-process from: {model_path}...")
                            self._llama_model = Llama(
                                model_path=str(model_path),
                                n_ctx=4096,
                                n_threads=4,
                                verbose=False,
                                flash_attn=True,
                            )
                        out = self._llama_model.create_chat_completion(
                            messages=[
                                {"role": "system", "content": "You are Gemma E2B, an expert activity classifier."},
                                {"role": "user", "content": batch_prompt},
                            ],
                            temperature=0.1,
                            max_tokens=len(chunk) * 75 + 100,
                        )
                        text = out["choices"][0]["message"]["content"]
                        parsed_items = self._parse_llm_json_batch(text, tax)
                        for item in parsed_items:
                            item_id = item.get("id")
                            if item_id is not None:
                                item["backend"] = f"local_gguf_batch ({model_path.name})"
                                chunk_res_map[item_id] = item
                    except Exception as e:
                        logger.error(f"Batch Gemma execution error: {e}")

                    for c_idx, a in enumerate(chunk):
                        act_id = a.get("id", c_idx)
                        if act_id in chunk_res_map:
                            results.append(chunk_res_map[act_id])
                        elif c_idx in chunk_res_map:
                            results.append(chunk_res_map[c_idx])
                        else:
                            h = self._smart_heuristic_classify(a.get("app", ""), a.get("title", ""), a.get("url"), tax)
                            h["reasoning"] = f"{h.get('reasoning', '')} (Fallback from batch Gemma)"
                            results.append(h)

                if progress_callback:
                    progress_callback(len(results), len(activities))
            return results

        if self.backend in ["llama_server", "llama-server"]:
            try:
                from llama_server_manager import llama_server
                model_path = self.find_local_model_path()
                if model_path:
                    self.openai_url = llama_server.ensure_server(model_path)
            except Exception as e:
                logger.error(f"Failed to ensure llama-server: {e}")

            import concurrent.futures
            from llama_server_manager import llama_server
            workers = min(max(1, max_streams), llama_server.n_parallel)
            results = [None] * len(activities)

            def _eval_one(idx, a):
                prompt = self._build_prompt(a.get("app", ""), a.get("title", ""), a.get("url"), tax)
                res = self._call_openai_compatible(prompt, tax)
                if not res or res.get("category") not in tax:
                    res = self._smart_heuristic_classify(a.get("app", ""), a.get("title", ""), a.get("url"), tax)
                    res["reasoning"] = f"{res.get('reasoning', '')} (Fallback from llama-server)"
                else:
                    res["backend"] = "llama_server"
                return idx, res

            done_count = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_eval_one, i, a) for i, a in enumerate(activities)]
                for fut in concurrent.futures.as_completed(futures):
                    i, res = fut.result()
                    results[i] = res
                    done_count += 1
                    if progress_callback:
                        progress_callback(done_count, len(activities))
            return results

        results = []
        for a in activities:
            prompt = self._build_prompt(a.get("app", ""), a.get("title", ""), a.get("url"), tax)
            if self.backend == "ollama":
                res = self._call_ollama(prompt, tax)
            elif self.backend == "gemini_api":
                res = self._call_gemini_api(prompt, tax)
            elif self.backend == "openai_compatible":
                res = self._call_openai_compatible(prompt, tax)
            else:
                res = self._smart_heuristic_classify(a.get("app", ""), a.get("title", ""), a.get("url"), tax)

            if not res or res.get("category") not in tax:
                res = self._smart_heuristic_classify(a.get("app", ""), a.get("title", ""), a.get("url"), tax)
                res["reasoning"] = f"{res.get('reasoning', '')} (Fallback from {self.backend})"
            results.append(res)
        return results

    def classify_parallel_streams(
        self,
        activities: List[Dict[str, Any]],
        taxonomy: Optional[Dict[str, List[str]]] = None,
        max_streams: int = 8,
    ) -> List[Dict[str, Any]]:
        """Decode multiple sequences simultaneously using llama.cpp parallel sequences (1 memory pass across all streams)."""
        if not activities:
            return []

        model_path = self.find_local_model_path()
        tax = taxonomy or self.taxonomy
        if not model_path:
            return [
                self._smart_heuristic_classify(
                    a.get("app", ""), a.get("title", ""), a.get("url"), tax
                )
                for a in activities
            ]

        try:
            from llama_cpp import Llama, _internals, llama_cpp
            import numpy as np

            if not self._llama_model:
                logger.info(f"Loading local Gemma model in-process from: {model_path}...")
                self._llama_model = Llama(
                    model_path=str(model_path),
                    n_ctx=4096,
                    n_threads=4,
                    verbose=False,
                    flash_attn=True,
                )
                logger.info("Local Gemma model loaded successfully.")

            vocab = llama_cpp.llama_model_get_vocab(self._llama_model._model.model)
            n_vocab = llama_cpp.llama_vocab_n_tokens(vocab)
            eos_token = llama_cpp.llama_vocab_eos(vocab)

            chunk_size = min(max_streams, len(activities))
            cparams = llama_cpp.llama_context_default_params()
            cparams.n_ctx = 4096
            cparams.n_batch = 1024
            cparams.n_ubatch = 512
            cparams.n_seq_max = chunk_size
            cparams.flash_attn_type = 1
            cparams.n_threads = 4
            cparams.kv_unified = True

            lctx = _internals.LlamaContext(model=self._llama_model._model, params=cparams, verbose=False)
            batch = llama_cpp.llama_batch_init(2048, 0, 1)

            results: List[Dict[str, Any]] = []

            for chunk_start in range(0, len(activities), chunk_size):
                chunk = activities[chunk_start : chunk_start + chunk_size]
                prompts = [
                    f"<start_of_turn>user\n{self._build_prompt(a.get('app', ''), a.get('title', ''), a.get('url'), tax)}<end_of_turn>\n<start_of_turn>model\n"
                    for a in chunk
                ]
                tokenized = [self._llama_model.tokenize(p.encode("utf-8")) for p in prompts]

                # Clear KV cache for new chunk
                lctx.kv_cache_clear()

                # Phase 1: Pack prompt tokens for all parallel streams into a single batch
                idx = 0
                stream_token_idx = {}
                for seq_id, tokens in enumerate(tokenized):
                    for pos, tok in enumerate(tokens):
                        batch.token[idx] = tok
                        batch.pos[idx] = pos
                        batch.n_seq_id[idx] = 1
                        batch.seq_id[idx][0] = seq_id
                        is_last = (pos == len(tokens) - 1)
                        batch.logits[idx] = 1 if is_last else 0
                        if is_last:
                            stream_token_idx[seq_id] = idx
                        idx += 1
                batch.n_tokens = idx

                ret = llama_cpp.llama_decode(lctx.ctx, batch)
                if ret != 0:
                    logger.warning(f"llama_decode failed on prompt batch (code {ret}), falling back.")
                    for a in chunk:
                        results.append(
                            self._smart_heuristic_classify(
                                a.get("app", ""), a.get("title", ""), a.get("url"), tax
                            )
                        )
                    continue

                streams = {}
                for seq_id in range(len(chunk)):
                    out_idx = stream_token_idx[seq_id]
                    logits_ptr = lctx.get_logits_ith(out_idx)
                    logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,))
                    next_tok = int(np.argmax(logits))
                    is_done = next_tok in (eos_token, 1, 107)
                    streams[seq_id] = {
                        "pos": len(tokenized[seq_id]),
                        "tokens": [next_tok] if not is_done else [],
                        "done": is_done,
                    }

                # Phase 2: Parallel token generation across all active streams (1 memory pass per token step)
                for step in range(80):
                    active_seqs = [s for s, data in streams.items() if not data["done"]]
                    if not active_seqs:
                        break

                    batch.n_tokens = len(active_seqs)
                    for i, seq_id in enumerate(active_seqs):
                        last_tok = streams[seq_id]["tokens"][-1] if streams[seq_id]["tokens"] else eos_token
                        batch.token[i] = last_tok
                        batch.pos[i] = streams[seq_id]["pos"]
                        batch.n_seq_id[i] = 1
                        batch.seq_id[i][0] = seq_id
                        batch.logits[i] = 1
                        streams[seq_id]["pos"] += 1

                    ret = llama_cpp.llama_decode(lctx.ctx, batch)
                    if ret != 0:
                        break

                    for i, seq_id in enumerate(active_seqs):
                        logits_ptr = lctx.get_logits_ith(i)
                        logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,))
                        next_tok = int(np.argmax(logits))
                        streams[seq_id]["tokens"].append(next_tok)
                        tok_str = self._llama_model.detokenize([next_tok]).decode(
                            "utf-8", errors="ignore"
                        )
                        if next_tok in (eos_token, 1, 107) or "}" in tok_str:
                            streams[seq_id]["done"] = True

                # Phase 3: Detokenize and parse outputs
                for seq_id, act in enumerate(chunk):
                    text = self._llama_model.detokenize(streams[seq_id]["tokens"]).decode(
                        "utf-8", errors="ignore"
                    )
                    parsed = self._parse_llm_json(text, tax)
                    if parsed:
                        parsed["backend"] = f"local_gguf_parallel ({model_path.name})"
                        results.append(parsed)
                    else:
                        h = self._smart_heuristic_classify(
                            act.get("app", ""), act.get("title", ""), act.get("url"), tax
                        )
                        h["reasoning"] = f"{h.get('reasoning', '')} (Fallback from parallel stream parse)"
                        results.append(h)

            llama_cpp.llama_batch_free(batch)
            lctx.close()
            return results
        except Exception as e:
            logger.error(f"classify_parallel_streams failed: {e}. Falling back to heuristic.")
            return [
                self._smart_heuristic_classify(
                    a.get("app", ""), a.get("title", ""), a.get("url"), tax
                )
                for a in activities
            ]

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
            f"1. Terminal emulators (foot, kitty, alacritty, ptyxis, konsole, gnome-terminal, xterm, wezterm, etc.) and shell prompts/sessions ('<user>@<host>', '~', bash, fish, zsh, tmux, ssh, cli commands) MUST ALWAYS be categorized as category 'Development', subcategory 'DevOps & Terminal'. NEVER categorize shell/terminal activities as 'System & Admin'.\n"
            f"2. 'System & Admin' is reserved strictly for OS system settings dialogs, hardware monitors (htop, btop), local activity trackers (ActivityWatch web UI), and package management tools.\n"
            f"3. For web browser activities, do NOT lazily dump activities into 'Browsing & Research > Search & Exploration'. Categorize based on the website domain, URL path, and actual function:\n"
            f"   - Local dev servers (localhost, 127.0.0.1, 0.0.0.0, dev ports), admin consoles (Django site admin, scraper dashboards, Grafana), code repositories (GitHub, GitLab, 'user/repo' titles), and AI assistants/LLMs (Claude, ChatGPT, Gemini, AI Studio) MUST be categorized under 'Development' (DevOps & Terminal, Coding & IDE, or Git & Code Review).\n"
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
            f"Respond with ONLY a valid JSON object in this exact schema, with no markdown fences:\n"
            f'{{"category": "<Category>", "subcategory": "<Subcategory>", "confidence": 0.95, "reasoning": "<1-sentence reasoning>"}}'
        )

    def _build_batch_prompt(
        self,
        activities: List[Dict[str, Any]],
        taxonomy: Dict[str, List[str]],
    ) -> str:
        tax_lines = []
        for cat, subcats in taxonomy.items():
            sub_str = ", ".join(subcats)
            tax_lines.append(f"- {cat}: [{sub_str}]")
        taxonomy_text = "\n".join(tax_lines)

        items_text = []
        for i, act in enumerate(activities):
            act_id = act.get("id", i)
            app = act.get("app") or "browser"
            title = act.get("title") or "unknown"
            url = act.get("url") or "none"
            parsed_domain = ""
            if url and url != "none":
                try:
                    from urllib.parse import urlparse
                    test_url = url if "://" in url else f"https://{url}"
                    p = urlparse(test_url)
                    parsed_domain = p.netloc.lower().split(":")[0]
                except Exception:
                    parsed_domain = ""
            items_text.append(
                f"[Activity {i+1}] ID: {act_id} | Application: {app} | Window Title: {title} | Domain: {parsed_domain or 'none'} | URL: {url}"
            )
        activities_formatted = "\n".join(items_text)

        return (
            f"You are Gemma E2B, an expert activity classifier for ActivityWatch.\n"
            f"Task: Categorize each of the following user activities into exactly ONE category and a matching subcategory from this taxonomy:\n\n"
            f"{taxonomy_text}\n\n"
            f"Classification Rules:\n"
            f"1. Terminal emulators (foot, kitty, alacritty, ptyxis, konsole, gnome-terminal, xterm, wezterm, etc.) and shell prompts/sessions ('<user>@<host>', '~', bash, fish, zsh, tmux, ssh) MUST ALWAYS be categorized as category 'Development', subcategory 'DevOps & Terminal'. NEVER categorize terminal shells as 'System & Admin'.\n"
            f"2. 'System & Admin' is reserved strictly for OS system settings dialogs, hardware monitors (htop, btop), local activity trackers (ActivityWatch web UI), and package management tools.\n"
            f"3. For web browser activities, do NOT lazily dump activities into 'Browsing & Research > Search & Exploration'. Categorize based on the website domain, URL path, and actual function:\n"
            f"   - Local dev servers (localhost, 127.0.0.1, 0.0.0.0, dev ports), admin consoles (Django site admin, scraper dashboards, Grafana), code repositories (GitHub, GitLab, 'user/repo' titles), and AI assistants/LLMs (Claude, ChatGPT, Gemini, AI Studio) MUST be categorized under 'Development' (DevOps & Terminal, Coding & IDE, or Git & Code Review).\n"
            f"   - Local activity monitors (ActivityWatch web UI) belong to 'System & Admin > Hardware & System Monitor'.\n"
            f"   - Webmail services and email inboxes accessed in a browser (Gmail, Proton Mail, Mailbox.org, Outlook/Office365, Yahoo Mail, Fastmail, webmail) MUST ALWAYS be categorized as 'Communication > Email'. NEVER categorize web email as 'Browsing & Research'.\n"
            f"   - Project management (Jira, Linear), notes (Notion), and office/spreadsheets (Google Docs/Sheets, Grist) belong to 'Productivity & Planning'.\n"
            f"   - Tech blogs, benchmarks (Artificial Analysis), engineering discussions (Hacker News), financial market charts, and research papers belong to 'Browsing & Research > Technical Articles & Papers'.\n"
            f"   - Online shopping, retail, and store sites belong to 'Browsing & Research > Shopping & E-Commerce'.\n"
            f"   - Blank new browser tabs ('New Tab', 'chrome://newtab', 'about:blank') belong to 'Uncategorized > General Activity'.\n"
            f"   - ONLY actual web search engines (Google Search, DuckDuckGo, Brave Search) or exploratory general browsing belong to 'Browsing & Research > Search & Exploration'.\n\n"
            f"Activities to categorize:\n"
            f"{activities_formatted}\n\n"
            f"Respond with ONLY a valid JSON array of objects, one for each activity in the exact same order:\n"
            f'[\n'
            f'  {{"id": <ID>, "category": "<Category>", "subcategory": "<Subcategory>", "confidence": 0.95, "reasoning": "<1-sentence reasoning>"}}\n'
            f']'
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

    def _parse_llm_json_batch(
        self, raw_text: str, taxonomy: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """Clean and extract a list of json objects from batch LLM generation."""
        results = []
        try:
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)

            items = []
            array_match = re.search(r"\[\s*\{.*?\}\s*\]", cleaned, re.DOTALL)
            if array_match:
                try:
                    items = json.loads(array_match.group(0))
                except Exception:
                    pass

            if not items:
                for obj_match in re.finditer(r"\{[^{}]*\}", cleaned, re.DOTALL):
                    try:
                        items.append(json.loads(obj_match.group(0)))
                    except Exception:
                        continue

            for item in items:
                if not isinstance(item, dict):
                    continue
                item_id = item.get("id")
                category_raw = str(item.get("category", "")).strip()
                subcategory_raw = str(item.get("subcategory", "")).strip()

                matched_cat = next(
                    (c for c in taxonomy.keys() if c.lower() == category_raw.lower()), None
                )
                if not matched_cat:
                    continue
                valid_subs = taxonomy[matched_cat]
                matched_sub = next(
                    (s for s in valid_subs if s.lower() == subcategory_raw.lower()), None
                )
                if not matched_sub:
                    matched_sub = next(
                        (s for s in valid_subs if subcategory_raw.lower() in s.lower()),
                        valid_subs[0] if valid_subs else "General",
                    )
                results.append({
                    "id": item_id,
                    "category": matched_cat,
                    "subcategory": matched_sub,
                    "full_category": f"{matched_cat} > {matched_sub}",
                    "confidence": float(item.get("confidence", 0.95)),
                    "reasoning": item.get("reasoning", "Classified by Gemma E2B"),
                    "backend": self.backend,
                })
        except Exception as e:
            logger.warning(f"Error parsing batch LLM JSON: {e}")
        return results

    def _call_llama_cpp(self, prompt: str, taxonomy: Dict[str, List[str]], max_tokens: int = 150) -> Dict[str, Any]:
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
                    flash_attn=True,
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
                max_tokens=max_tokens,
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
        """Query any OpenAI-compatible server (llama-server, vLLM, LMStudio, LocalAI)."""
        try:
            base = (self.openai_url or "http://127.0.0.1:8088/v1").rstrip("/")
            url = f"{base}/chat/completions"
            headers = {}
            if self.openai_key:
                headers["Authorization"] = f"Bearer {self.openai_key}"
            payload = {
                "model": self.model_name or "local",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 150,
            }
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = (msg.get("content") or msg.get("reasoning_content") or "").strip()
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
                "backend": "gemma_heuristic_proxy",
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
                "backend": "gemma_heuristic_proxy",
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
                "backend": "gemma_heuristic_proxy",
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
                "backend": "gemma_heuristic_proxy",
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
                "backend": "gemma_heuristic_proxy",
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
                "backend": "gemma_heuristic_proxy",
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
                "backend": "gemma_heuristic_proxy",
            }

        # 0.8. AI Developer Tools & LLM Workspaces
        is_ai_coding = any(s in combined for s in [
            "claude.ai", "chatgpt", "openai.com", "aistudio.google.com", "gemini.google.com",
            "huggingface.co", "openrouter.ai", "colab.research.google.com", "kaggle.com", "jupyter",
            "modal.com", "replicate.com", "v0.dev"
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
                "backend": "gemma_heuristic_proxy",
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
                "backend": "gemma_heuristic_proxy",
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
                "backend": "gemma_heuristic_proxy",
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
            "backend": "gemma_heuristic_proxy",
        }
