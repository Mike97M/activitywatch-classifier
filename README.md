# ActivityWatch Gemma Classifier 🧠⚡

An intelligent, non-destructive activity classification and active learning pipeline for [ActivityWatch](https://activitywatch.net/) powered by a calibrated fast classifier and Google's **Gemma E2B** oracle.

---

## 🌟 Key Features

- **Non-Destructive Operations**: Source ActivityWatch buckets (e.g. `aw-watcher-window_*`, `aw-watcher-web-*`) are **never modified in-place**. Events are cloned, enriched with classification metadata (`category`, `confidence`, `classified_by`, `reasoning`), and saved to a brand-new target bucket (e.g. `aw-classifier-categorized`).
- **Two-Tier Architecture with Active Learning**:
  - **Tier 1 (Fast Classifier)**: High-speed, calibrated text classifier (`SGDClassifier` with TF-IDF character/word n-grams) that predicts activity categories in milliseconds with probability confidence scores.
  - **Tier 2 (Gemma E2B Oracle)**: When the fast classifier's confidence score drops below the confidence threshold (e.g. `< 0.75`), **Gemma E2B** is automatically summoned to analyze the app, window title, and URL context.
  - **Self-Improving Online Retraining**: Every activity classified by Gemma is added to the training set, immediately retraining and improving the fast model.
- **Human-in-the-Loop Feedback**: Interactive UI allows correcting any activity's category with a single click, instantly feeding the correction back into the model.
- **Automatic PII & Personal Data Scrubbing**: Before training samples or classified activities are saved or committed to Git, an active anonymizer removes usernames, shell prompt identities (`user@host`), private home paths (`/home/username/...`), emails, IP addresses, contact names, and sensitive URL tokens/credentials.
- **Multiple Gemma Backends**:
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

Gemma is executed **locally and directly in-process** inside the Python script via `llama-cpp-python`:
- **No external server required**: No need to install or run Docker, Ollama, or third-party servers.
- **In-process inference**: Model weights (`.gguf`) are loaded directly into RAM/CPU (or GPU if available) with multi-threaded inference (`n_threads=4`).
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

- Select source buckets to clone from (e.g. `aw-watcher-window_laptop`).
- Adjust the **Confidence Threshold** slider (default: `0.75`).
- Click **"Start Non-Destructive Cloning & Gemma Classification"**.
- View category distribution graphs, confidence metrics, and inspect the new bucket!

### 4. CLI Usage

You can also run classification or check status directly from the terminal with `uv`:

```bash
# Check connection to ActivityWatch, classifier state, and local Gemma model
uv run python main.py status

# Run non-destructive cloning & classification
uv run python main.py run \
  --source aw-watcher-window_laptop aw-watcher-web-brave_laptop \
  --target aw-classifier-categorized \
  --limit 200 \
  --threshold 0.75 \
  --backend local
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
