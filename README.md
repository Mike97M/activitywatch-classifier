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
- **Multiple Gemma Backends**:
  - `heuristic`: Built-in zero-latency smart oracle for instant offline use.
  - `ollama`: Connects to local/remote Ollama (`gemma-4-e2b`, `gemma:2b`).
  - `gemini_api`: Google Gemini/Gemma API endpoint.
  - `openai_compatible`: Any OpenAI-compatible server (vLLM, LM Studio, LiteRT).
  - `llama_cpp`: Local GGUF models on CPU/GPU via `llama-cpp-python`.
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

## 🚀 Quickstart

### 1. Requirements & Setup

Make sure Python 3.10+ and an ActivityWatch instance are running.

```bash
# Clone and enter the repository
cd path/to/activitywatch-classifier

# Create virtual environment
python3 -m venv --copies .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web UI

```bash
python main.py serve
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser.

- Select the source buckets you want to clone from (e.g. `aw-watcher-window_laptop`).
- Adjust the **Confidence Threshold** slider (default: `0.75`).
- Click **"Start Non-Destructive Cloning & Gemma Classification"**.
- View category distribution graphs, confidence metrics, and inspect the new bucket!

### 3. CLI Usage

You can also run classification or check status directly from the terminal:

```bash
# Check connection to ActivityWatch and classifier state
python main.py status

# Run non-destructive cloning and classification
python main.py run \
  --source aw-watcher-window_laptop aw-watcher-web-brave_laptop \
  --target aw-classifier-categorized \
  --limit 200 \
  --threshold 0.75 \
  --backend heuristic
```

---

## ⚙️ Configuration & Gemma Backends

Settings can be customized via the Web UI Settings modal, CLI arguments, or environment variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| `AW_SERVER_URL` | Auto-detected | ActivityWatch server endpoint (`http://localhost:5600`) |
| `CONFIDENCE_THRESHOLD` | `0.75` | Probability cutoff below which Gemma is called |
| `GEMMA_BACKEND` | `heuristic` | Engine: `heuristic`, `ollama`, `gemini_api`, `openai_compatible`, `llama_cpp` |
| `GEMMA_MODEL` | `google/gemma-4-E2B-it` | Model identifier or tag |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama service address |
| `GEMINI_API_KEY` | `""` | Google Gemini API key if using cloud backend |
| `WEB_PORT` | `5000` | Port for the web dashboard |

### Default Category Taxonomy
- **Development**: IDEs, terminals, git, StackOverflow, documentation, debugging.
- **Communication**: Slack, Discord, Telegram, Teams, Zoom, email.
- **Browsing & Research**: Articles, Wikipedia, search engines.
- **Productivity & Planning**: Obsidian, Notion, Todoist, spreadsheets, docs.
- **Media & Entertainment**: YouTube, Spotify, VLC, Netflix, games.
- **Social Media**: X / Twitter, Reddit, LinkedIn, Mastodon.
- **System & Admin**: OS settings, btop, package management.
- **Uncategorized**: General desktop activity.

*(Categories can be dynamically modified in the UI or configuration).*

---

## 🧪 Testing

Run the automated test suite with pytest:

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -v
```

---

## 🔒 Safety & Non-Destructive Guarantee

ActivityWatch stores your personal productivity data. This application strictly guarantees:
1. **Zero in-place modification**: It only issues `GET` requests to source buckets.
2. **Dedicated target bucket**: All classified data is created in a separate bucket (`POST /api/0/buckets/<new_id>`).
3. You can safely inspect, re-run, or delete the target bucket at any time without impacting your raw ActivityWatch logs.
