# ActivityWatch Gemma Classifier 🧠⚡

An intelligent, non-destructive activity classification and active learning pipeline for [ActivityWatch](https://activitywatch.net/) powered by a calibrated fast classifier and Google's **Gemma E2B** oracle.

---

## 🌟 Key Features

- **Non-Destructive Operations**: Source ActivityWatch buckets (e.g. `aw-watcher-window_*`, `aw-watcher-web-*`) are **never modified in-place**. Events are cloned, enriched with classification metadata (`category`, `confidence`, `classified_by`, `reasoning`), and saved to a brand-new target bucket (e.g. `aw-classifier-categorized`).
- **Two-Tier Architecture with Active Learning**:
  - **Tier 1 (Fast Classifier)**: High-speed, calibrated text classifier (`SGDClassifier` with TF-IDF character/word n-grams) that predicts activity categories in milliseconds with probability confidence scores.
  - **Tier 2 (Gemma E2B Oracle)**: When the fast classifier's confidence score drops below the confidence threshold (default `0.60`), **Gemma E2B** is automatically summoned to analyze the app, window title, and URL context.
  - **Self-Improving Online Retraining**: Every activity classified by Gemma is added to the training set, immediately retraining and improving the fast model.
- **Human-in-the-Loop Feedback**: Interactive UI allows correcting any activity's category with a single click, instantly feeding the correction back into the model.
- **Automatic PII & Personal Data Scrubbing**: Before training samples or classified activities are saved or committed to Git, an active anonymizer removes usernames, shell prompt identities (`user@host`), private home paths (`/home/username/...`), emails, IP addresses, contact names, and sensitive URL tokens/credentials.
- **Multiple Gemma Backends**:
  - `llama_server` (default): Native `llama.cpp` server started and managed automatically, with parallel slots and token logprobs for label confidence.
  - `local`: Direct in-process Gemma 4 E2B execution via `llama-cpp-python` (CPU/GPU).
  - `heuristic`: Built-in zero-latency smart oracle for instant offline use.
  - `ollama`: Connects to local/remote Ollama (`gemma-4-e2b`).
  - `gemini_api`: Google Gemini/Gemma API endpoint.
  - `openai_compatible`: Any OpenAI-compatible server (vLLM, LM Studio, LiteRT).
- **Modern Responsive Web UI**: Dashboard with live bucket discovery, category distribution chart, confidence tuning slider, and real-time activity stream.

---

## 🏗️ Architecture

```
                                    ┌────────────────────────┐
                                    │ ActivityWatch Watchers │
                                    │ (Window / Web Buckets) │
                                    └───────────┬────────────┘
                                                │ Read (Non-destructive)
                                                ▼
                                    ┌────────────────────────┐
                                    │ Feature Extractor      │
                                    │ (App, Title, Domain)   │
                                    └───────────┬────────────┘
                                                │
                                                ▼
                                    ┌────────────────────────┐
                                    │ Fast Classifier Model  │
                                    │ (TF-IDF + Log Loss)    │
                                    └───────────┬────────────┘
                                                │
                                  Confidence >= Threshold?
                                       /         \
                             YES      /           \      NO (Uncertain)
                                     /             \
                                    ▼               ▼
                        ┌──────────────────┐  ┌────────────────────────┐
                        │ Fast Classifier  │  │   Gemma E2B Oracle     │
                        │ Category Output  │  │  (Deep Understanding)  │
                        └─────────┬────────┘  └───────────┬────────────┘
                                  │                       │
                                  │                       ├─► [Add to Training Set]
                                  │                       │          │
                                  │                       │          ▼
                                  │                       │  [Retrain Fast Model]
                                  │                       │
                                  └───────────┬───────────┘
                                              │
                                              ▼
                                 ┌──────────────────────────┐
                                 │ Enriched Activity Event  │
                                 │ (Category, Score, Tag)   │
                                 └────────────┬─────────────┘
                                              │
                                              ▼
                                 ┌──────────────────────────┐
                                 │ New Target AW Bucket     │
                                 │ aw-classifier-categorized│
                                 └──────────────────────────┘
```

---

## ⚡ How Gemma Runs

Gemma runs **locally** — nothing leaves your machine with the default backends:
- **`llama_server` (default)**: the app downloads a pinned, SHA256-verified `llama-server` binary from the official llama.cpp releases into `bin/` and manages the process itself (4 parallel slots). Label confidence comes from token log-probabilities.
- **`local`**: weights (`.gguf`) are loaded in-process via `llama-cpp-python` (CPU, or GPU if available).
- No Docker, Ollama, or third-party servers needed.
- **Direct download from the script**:
  ```bash
  # Download Gemma 4 E2B Instruct GGUF weights directly into models/ (Standard Q4_K_M ~3.1 GB)
  uv run python main.py download-gemma --variant standard

  # Or download the lightweight Gemma 4 E2B variant (Q3_K_M ~2.5 GB)
  uv run python main.py download-gemma --variant light
  ```
  *(You can also click the download button inside the Web UI Settings panel).*
- **Pluggable Backends**: If preferred, you can also seamlessly switch backend to `ollama`, `gemini_api` (Google Cloud), `openai_compatible` (vLLM/LMStudio), or the built-in offline `heuristic` proxy.

---

## 🚀 Quickstart with `uv`

### 1. Requirements & Setup

Make sure `uv` is installed and an ActivityWatch instance is running.

```bash
cd path/to/activitywatch-classifier

# uv automatically locks and installs dependencies when running
uv sync
```

### 2. Download Gemma Weights Locally

```bash
uv run python main.py download-gemma
```

### 3. Launch the Web UI

```bash
uv run python main.py serve
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser.

The server binds to `127.0.0.1` only. To reach it from another device, set `WEB_HOST=0.0.0.0` and add the name/IP you'll use to `WEB_ALLOWED_HOSTS` (e.g. `WEB_ALLOWED_HOSTS=localhost,127.0.0.1,192.168.1.20`). There is no login, so only do this on a network you trust.

- Select source buckets to clone from (e.g. `aw-watcher-window_<hostname>`).
- Adjust the **Confidence Threshold** slider (default: `0.60`).
- Click **"Start Non-Destructive Cloning & Gemma Classification"**.
- View category distribution graphs, confidence metrics, and inspect the new bucket!

### 4. CLI Usage

You can also run classification or check status directly from the terminal with `uv`:

```bash
# Check connection to ActivityWatch, classifier state, and local Gemma model
uv run python main.py status

# Run non-destructive cloning & classification
uv run python main.py run \
  --source aw-watcher-window_<hostname> aw-watcher-web-brave_<hostname> \
  --target aw-classifier-categorized \
  --limit 200 \
  --threshold 0.60 \
  --backend llama_server
```

### 5. Automated Tests

```bash
uv run pytest tests/ -v
```

---

## 🔒 Safety & Non-Destructive Guarantee

ActivityWatch stores your personal productivity data. This application strictly guarantees:
1. **Zero in-place modification**: It only issues `GET` requests to source buckets.
2. **Dedicated target bucket**: All classified data is created in a separate bucket (`POST /api/0/buckets/<new_id>`).
3. You can safely inspect, re-run, or delete the target bucket at any time without impacting your raw ActivityWatch logs.
4. **One exception — ActivityWatch categories**: after each run the classifier's taxonomy replaces the `classes`, `category_sets` and `active_set_ids` settings in ActivityWatch. Before the first overwrite your original settings are saved to `data/aw_categories_backup.json`; restore them by POSTing each key back to `/api/0/settings/<key>`.

---

## 📜 Third-Party Licenses

This project uses, downloads, or talks to the following tools. Each keeps its own license:

| Component | How it's used | License |
|---|---|---|
| [ActivityWatch](https://activitywatch.net/) | Accessed through its local REST API (no code bundled) | MPL-2.0 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`) | Prebuilt binary downloaded into `bin/` | MIT |
| [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | In-process `local` backend | MIT |
| [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) weights (GGUF by Unsloth) | Downloaded into `models/` | Apache-2.0 |
| FastAPI, Pydantic, pytest | Python dependencies | MIT |
| Uvicorn, scikit-learn, NumPy, SciPy, HTTPX, joblib | Python dependencies | BSD-3-Clause |
| Requests, python-multipart, huggingface-hub | Python dependencies | Apache-2.0 |
| Tailwind CSS, Chart.js | Web UI, loaded from CDN | MIT |
| Font Awesome Free | Web UI icons, loaded from CDN | Icons CC BY 4.0, fonts SIL OFL 1.1, code MIT |

Model weights and binaries are downloaded on demand and are not redistributed in this repository. If you switch to another model or backend (Ollama, Gemini API, other GGUFs), its license and terms apply instead.
