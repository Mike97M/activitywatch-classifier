/* ActivityWatch Classifier Frontend Application */

let currentCategories = [];
let categoryChart = null;
let lastEvents = [];

document.addEventListener("DOMContentLoaded", () => {
  initApp();
  setupEventListeners();
});

async function initApp() {
  await fetchStatus();
  await fetchBuckets();
  initChart();
  await loadTargetEvents();
}

// -------------------------------------------------------------
// Data Fetching & Status Updates
// -------------------------------------------------------------
async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();

    // ActivityWatch indicator
    const awIndicator = document.getElementById("aw-indicator");
    const awText = document.getElementById("aw-text");
    if (data.activitywatch.connected) {
      awIndicator.className = "w-2 h-2 rounded-full bg-emerald-400";
      awText.textContent = `AW: ${data.activitywatch.info.hostname || "Connected"}`;
    } else {
      awIndicator.className = "w-2 h-2 rounded-full bg-rose-500";
      awText.textContent = "AW: Offline";
    }

    // Classifier indicator
    const modelText = document.getElementById("model-text");
    const isTrained = data.classifier.is_trained;
    const samples = data.classifier.total_training_samples || 0;
    modelText.textContent = isTrained
      ? `Classifier: Online (${samples} samples)`
      : `Classifier: Untrained (${samples} samples)`;

    // Cards
    document.getElementById("stat-training-samples").textContent = samples;
    document.getElementById("stat-threshold-val").textContent = data.confidence_threshold.toFixed(2);
    document.getElementById("slider-threshold-label").textContent = data.confidence_threshold.toFixed(2);
    document.getElementById("confidence-threshold-slider").value = data.confidence_threshold;

    currentCategories = data.classifier.all_configured_categories || [];

    // Last run stats if available
    if (data.last_run_stats && data.last_run_stats.total_events_processed) {
      document.getElementById("stat-fast-calls").textContent = data.last_run_stats.fast_classifier_calls || 0;
      document.getElementById("stat-gemma-calls").textContent = data.last_run_stats.gemma_oracle_calls || 0;
      updateChart(data.last_run_stats.category_distribution || {});
    }

    // Pre-fill settings modal
    document.getElementById("setting-gemma-backend").value = data.gemma_oracle.backend || "local";
    document.getElementById("setting-ollama-host").value = data.gemma_oracle.ollama_host || "http://127.0.0.1:11434";
    document.getElementById("setting-gemma-model").value = data.gemma_oracle.model_name || "google/gemma-4-E2B-it";
    document.getElementById("setting-categories").value = currentCategories.join(", ");

    // Local model status badge
    const localInfo = data.gemma_oracle.local_model || {};
    const badge = document.getElementById("local-model-status-badge");
    const details = document.getElementById("local-model-details");
    if (badge && details) {
      if (localInfo.available) {
        badge.className = "px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-950 text-emerald-300 border border-emerald-800/60";
        badge.textContent = "Weights Ready";
        details.textContent = `${localInfo.filename} (${localInfo.size_gb} GB) in models/`;
      } else {
        badge.className = "px-2 py-0.5 rounded text-[10px] font-mono bg-amber-950 text-amber-300 border border-amber-800/60";
        badge.textContent = "No weights found";
        details.textContent = "No local GGUF in models/. Click download below or run: uv run python main.py download-gemma";
      }
    }
  } catch (err) {
    console.error("Status fetch failed:", err);
  }
}

async function fetchBuckets() {
  const container = document.getElementById("source-buckets-list");
  try {
    const res = await fetch("/api/buckets");
    const data = await res.json();
    const buckets = data.buckets || {};

    if (Object.keys(buckets).length === 0) {
      container.innerHTML = `<div class="text-xs text-slate-500 text-center py-4">No buckets found in ActivityWatch.</div>`;
      return;
    }

    container.innerHTML = "";
    for (const [bid, binfo] of Object.entries(buckets)) {
      // Don't show current target bucket as source candidate to avoid recursive self-cloning
      const targetId = document.getElementById("target-bucket-id").value;
      if (bid === targetId) continue;

      const isWindowOrWeb = bid.includes("window") || bid.includes("web");
      const client = binfo.client || "watcher";
      const type = binfo.type || "unknown";

      const item = document.createElement("label");
      item.className = "flex items-center justify-between p-2 rounded-lg bg-slate-900/80 hover:bg-slate-800/80 border border-slate-800 cursor-pointer transition";
      item.innerHTML = `
        <div class="flex items-center space-x-3 overflow-hidden">
          <input type="checkbox" name="source-bucket" value="${bid}" ${isWindowOrWeb ? "checked" : ""} 
            class="accent-indigo-500 rounded border-slate-700 bg-slate-950 focus:ring-0" />
          <div class="truncate">
            <div class="text-xs font-mono text-slate-200 truncate">${bid}</div>
            <div class="text-[10px] text-slate-500">${client} &bull; ${type}</div>
          </div>
        </div>
        <span class="text-[10px] px-2 py-0.5 rounded font-mono bg-slate-800 text-slate-400 ml-2">
          ${isWindowOrWeb ? "Active Watcher" : "Bucket"}
        </span>
      `;
      container.appendChild(item);
    }
  } catch (err) {
    container.innerHTML = `<div class="text-xs text-rose-400 text-center py-4">Failed to fetch buckets: ${err.message}</div>`;
  }
}

// -------------------------------------------------------------
// Pipeline Execution
// -------------------------------------------------------------
async function runPipeline() {
  const checkboxes = document.querySelectorAll('input[name="source-bucket"]:checked');
  const sourceBucketIds = Array.from(checkboxes).map((cb) => cb.value);
  const targetBucketId = document.getElementById("target-bucket-id").value.trim() || "aw-classifier-categorized";
  const limit = parseInt(document.getElementById("limit-per-bucket").value, 10);
  const threshold = parseFloat(document.getElementById("confidence-threshold-slider").value);

  if (sourceBucketIds.length === 0) {
    alert("Please select at least one source bucket to clone activities from.");
    return;
  }

  const btn = document.getElementById("btn-run-pipeline");
  const btnText = document.getElementById("btn-run-text");
  const progressContainer = document.getElementById("progress-container");
  const progressFill = document.getElementById("progress-bar-fill");
  const progressPercent = document.getElementById("progress-percent");

  btn.disabled = true;
  btnText.textContent = "Cloning & Gemma Classifying...";
  progressContainer.classList.remove("hidden");
  progressFill.style.width = "40%";
  progressPercent.textContent = "Processing...";

  try {
    const res = await fetch("/api/run-pipeline", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_bucket_ids: sourceBucketIds,
        target_bucket_id: targetBucketId,
        limit_per_bucket: limit,
        confidence_threshold: threshold,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Pipeline run failed");
    }

    const data = await res.json();
    const results = data.results;

    progressFill.style.width = "100%";
    progressPercent.textContent = "100% Done";

    // Update stats
    document.getElementById("stat-fast-calls").textContent = results.fast_classifier_calls || 0;
    document.getElementById("stat-gemma-calls").textContent = results.gemma_oracle_calls || 0;
    document.getElementById("stat-training-samples").textContent = results.classifier_stats?.total_training_samples || 0;

    // Update chart
    updateChart(results.category_distribution || {});

    // Render preview events
    if (results.recent_preview) {
      renderEventsTable(results.recent_preview);
    } else {
      await loadTargetEvents(targetBucketId);
    }

    await fetchStatus();
  } catch (err) {
    alert("Error executing pipeline: " + err.message);
  } finally {
    setTimeout(() => {
      btn.disabled = false;
      btnText.textContent = "Start Non-Destructive Cloning & Gemma Classification";
      progressContainer.classList.add("hidden");
    }, 1200);
  }
}

// -------------------------------------------------------------
// Live Events & Human In The Loop Feedback
// -------------------------------------------------------------
async function loadTargetEvents(targetBucketId = null) {
  const bid = targetBucketId || document.getElementById("target-bucket-id").value;
  try {
    const res = await fetch(`/api/events/target?bucket_id=${encodeURIComponent(bid)}&limit=100`);
    const data = await res.json();
    const events = data.events || [];
    lastEvents = events.map((e) => ({
      timestamp: e.timestamp,
      duration: e.duration,
      app: e.data?.app || "",
      title: e.data?.title || "",
      url: e.data?.url || "",
      category: e.data?.category || "Uncategorized",
      confidence: e.data?.confidence || 0.0,
      classified_by: e.data?.classified_by || "Unknown",
      reasoning: e.data?.reasoning || "",
      source_bucket: e.data?.source_bucket || "",
    }));

    if (lastEvents.length > 0) {
      renderEventsTable(lastEvents);
      document.getElementById("target-bucket-info").textContent = `${lastEvents.length} events categorized in '${bid}'`;
    }
  } catch (err) {
    console.error("Failed loading target events:", err);
  }
}

function renderEventsTable(events) {
  const tbody = document.getElementById("events-tbody");
  if (!events || events.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="px-4 py-8 text-center text-slate-500">No events found in target bucket.</td></tr>`;
    return;
  }

  tbody.innerHTML = "";
  events.forEach((ev, idx) => {
    const tr = document.createElement("tr");
    tr.className = "hover:bg-slate-900/60 transition group";

    const dateStr = ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : "--:--";
    const durationStr = formatDuration(ev.duration);
    const catClass = getCategoryBadgeClass(ev.category);
    const isGemma = (ev.classified_by || "").includes("gemma");

    tr.innerHTML = `
      <td class="px-4 py-3 whitespace-nowrap text-slate-400 font-mono text-[11px]">
        <div>${dateStr}</div>
        <div class="text-[10px] text-slate-500">${durationStr}</div>
      </td>
      <td class="px-4 py-3 font-semibold text-slate-200 whitespace-nowrap">
        <div class="flex items-center space-x-1.5">
          <i class="fa-solid fa-window-maximize text-slate-500 text-[10px]"></i>
          <span>${escapeHtml(ev.app || "Browser/App")}</span>
        </div>
      </td>
      <td class="px-4 py-3 max-w-xs truncate text-slate-300" title="${escapeHtml(ev.title || ev.url || "")}">
        <div class="truncate font-medium">${escapeHtml(ev.title || "No Title")}</div>
        ${ev.url ? `<div class="truncate text-[10px] text-indigo-400 font-mono">${escapeHtml(ev.url)}</div>` : ""}
      </td>
      <td class="px-4 py-3 whitespace-nowrap">
        <span class="category-tag ${catClass}">${escapeHtml(ev.category)}</span>
      </td>
      <td class="px-4 py-3 whitespace-nowrap font-mono">
        <div class="flex items-center space-x-2">
          <span class="${ev.confidence >= 0.75 ? 'text-emerald-400' : 'text-amber-400'} font-bold">
            ${(ev.confidence * 100).toFixed(0)}%
          </span>
          <div class="w-12 bg-slate-800 rounded-full h-1.5">
            <div class="${ev.confidence >= 0.75 ? 'bg-emerald-400' : 'bg-amber-400'} h-1.5 rounded-full" 
              style="width: ${Math.min(100, ev.confidence * 100)}%"></div>
          </div>
        </div>
      </td>
      <td class="px-4 py-3 whitespace-nowrap">
        <span class="px-2 py-0.5 rounded text-[10px] font-mono ${
          isGemma ? 'bg-purple-950 text-purple-300 border border-purple-800/60' : 'bg-emerald-950 text-emerald-300 border border-emerald-800/60'
        }">
          <i class="fa-solid ${isGemma ? 'fa-wand-magic-sparkles' : 'fa-bolt'} mr-1"></i>
          ${isGemma ? 'Gemma E2B' : 'Fast Classifier'}
        </span>
      </td>
      <td class="px-4 py-3 whitespace-nowrap text-right">
        <button onclick="openCorrectionDialog(${idx})" class="text-xs px-2.5 py-1 rounded bg-slate-800 hover:bg-indigo-600 text-slate-300 hover:text-white transition opacity-80 group-hover:opacity-100">
          <i class="fa-solid fa-pencil mr-1"></i> Override
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function openCorrectionDialog(idx) {
  const ev = lastEvents[idx];
  if (!ev) return;

  const currentCat = ev.category;
  const promptOptions = currentCategories.map((c, i) => `${i + 1}. ${c}`).join("\n");
  const choice = prompt(
    `Reclassify Activity:\nApp: ${ev.app}\nTitle: ${ev.title}\n\nCurrent: ${currentCat}\n\nEnter category name or number:\n${promptOptions}`,
    currentCat
  );

  if (!choice || choice.trim() === "") return;

  let chosenCat = choice.trim();
  const num = parseInt(chosenCat, 10);
  if (!isNaN(num) && num >= 1 && num <= currentCategories.length) {
    chosenCat = currentCategories[num - 1];
  }

  const featureText = `app:${ev.app} title:${ev.title} url:${ev.url}`;
  submitCorrection(featureText, chosenCat, idx);
}

async function submitCorrection(featureText, correctCategory, idx) {
  try {
    const res = await fetch("/api/correct-event", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        feature_text: featureText,
        correct_category: correctCategory,
      }),
    });

    if (res.ok) {
      if (lastEvents[idx]) {
        lastEvents[idx].category = correctCategory;
        lastEvents[idx].confidence = 1.0;
        lastEvents[idx].classified_by = "Human Feedback (Retrained)";
        renderEventsTable(lastEvents);
      }
      await fetchStatus();
    }
  } catch (err) {
    alert("Correction failed: " + err.message);
  }
}

// -------------------------------------------------------------
// Chart.js Visualization
// -------------------------------------------------------------
function initChart() {
  const ctx = document.getElementById("categoryChart").getContext("2d");
  categoryChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: [],
      datasets: [
        {
          data: [],
          backgroundColor: [
            "#6366f1",
            "#38bdf8",
            "#f43f5e",
            "#eab308",
            "#10b981",
            "#a855f7",
            "#94a3b8",
            "#64748b",
          ],
          borderColor: "#0f172a",
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "right",
          labels: { color: "#cbd5e1", boxWidth: 12, font: { size: 11 } },
        },
      },
      cutout: "68%",
    },
  });
}

function updateChart(dist) {
  const labels = Object.keys(dist);
  const data = Object.values(dist);
  const total = data.reduce((a, b) => a + b, 0);

  document.getElementById("total-chart-events").textContent = `${total} events`;
  const emptyMsg = document.getElementById("chart-empty-msg");

  if (total === 0) {
    emptyMsg.classList.remove("hidden");
    categoryChart.data.labels = [];
    categoryChart.data.datasets[0].data = [];
  } else {
    emptyMsg.classList.add("hidden");
    categoryChart.data.labels = labels;
    categoryChart.data.datasets[0].data = data;
  }
  categoryChart.update();
}

// -------------------------------------------------------------
// UI Event Handlers & Helpers
// -------------------------------------------------------------
function setupEventListeners() {
  document.getElementById("btn-run-pipeline").addEventListener("click", runPipeline);
  document.getElementById("btn-refresh-buckets").addEventListener("click", fetchBuckets);
  document.getElementById("btn-refresh-events").addEventListener("click", () => loadTargetEvents());

  // Slider change
  const slider = document.getElementById("confidence-threshold-slider");
  slider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value).toFixed(2);
    document.getElementById("slider-threshold-label").textContent = val;
    document.getElementById("stat-threshold-val").textContent = val;
  });
  slider.addEventListener("change", async (e) => {
    const val = parseFloat(e.target.value);
    await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confidence_threshold: val }),
    });
  });

  // Filter events
  document.getElementById("filter-events-input").addEventListener("input", (e) => {
    const query = e.target.value.toLowerCase().trim();
    if (!query) {
      renderEventsTable(lastEvents);
      return;
    }
    const filtered = lastEvents.filter(
      (ev) =>
        ev.app.toLowerCase().includes(query) ||
        ev.title.toLowerCase().includes(query) ||
        ev.category.toLowerCase().includes(query) ||
        ev.url.toLowerCase().includes(query)
    );
    renderEventsTable(filtered);
  });

  // Settings modal
  const settingsModal = document.getElementById("settings-modal");
  document.getElementById("btn-open-settings").addEventListener("click", () => {
    settingsModal.classList.remove("hidden");
  });
  document.getElementById("btn-close-settings").addEventListener("click", () => {
    settingsModal.classList.add("hidden");
  });
  document.getElementById("btn-cancel-settings").addEventListener("click", () => {
    settingsModal.classList.add("hidden");
  });

  document.getElementById("setting-gemma-backend").addEventListener("change", (e) => {
    const val = e.target.value;
    document.getElementById("local-options").classList.toggle("hidden", val !== "local");
    document.getElementById("ollama-options").classList.toggle("hidden", val !== "ollama");
    document.getElementById("gemini-options").classList.toggle("hidden", val !== "gemini_api");
  });

  document.getElementById("btn-download-gemma-ui").addEventListener("click", () => {
    downloadGemmaFromUI("gemma-4-e2b");
  });
  document.getElementById("btn-download-gemma2-ui").addEventListener("click", () => {
    downloadGemmaFromUI("gemma-2-2b");
  });

  document.getElementById("btn-save-settings").addEventListener("click", async () => {
    const backend = document.getElementById("setting-gemma-backend").value;
    const ollamaHost = document.getElementById("setting-ollama-host").value.trim();
    const gemmaModel = document.getElementById("setting-gemma-model").value.trim();
    const geminiKey = document.getElementById("setting-gemini-key").value.trim();
    const rawCats = document.getElementById("setting-categories").value;
    const categories = rawCats.split(",").map((s) => s.trim()).filter(Boolean);

    await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        gemma_backend: backend,
        ollama_host: ollamaHost,
        gemma_model: gemmaModel,
        gemini_api_key: geminiKey || undefined,
        categories: categories.length > 0 ? categories : undefined,
      }),
    });
    settingsModal.classList.add("hidden");
    await fetchStatus();
  });

  // Inspect raw target bucket
  const rawModal = document.getElementById("raw-modal");
  document.getElementById("btn-inspect-target").addEventListener("click", async () => {
    const bid = document.getElementById("target-bucket-id").value;
    try {
      const res = await fetch(`/api/events/target?bucket_id=${encodeURIComponent(bid)}&limit=15`);
      const data = await res.json();
      document.getElementById("raw-json-viewer").textContent = JSON.stringify(data, null, 2);
      rawModal.classList.remove("hidden");
    } catch (err) {
      alert("Error inspecting target bucket: " + err.message);
    }
  });
  document.getElementById("btn-close-raw").addEventListener("click", () => {
    rawModal.classList.add("hidden");
  });
}

async function downloadGemmaFromUI(modelName) {
  const btn1 = document.getElementById("btn-download-gemma-ui");
  const btn2 = document.getElementById("btn-download-gemma2-ui");
  const details = document.getElementById("local-model-details");

  btn1.disabled = true;
  btn2.disabled = true;
  details.textContent = `Downloading ${modelName} weights directly from Hugging Face... This may take a few minutes depending on your internet connection.`;

  try {
    const res = await fetch("/api/download-gemma", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: modelName, quant: "Q4_K_M" }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Download failed");
    }

    const data = await res.json();
    alert(`Success! Downloaded Gemma weights to ${data.path}`);
    await fetchStatus();
  } catch (err) {
    alert("Download failed: " + err.message);
    details.textContent = "Download failed: " + err.message;
  } finally {
    btn1.disabled = false;
    btn2.disabled = false;
  }
}

function getCategoryBadgeClass(category) {
  const cat = (category || "").toLowerCase();
  if (cat.includes("dev") || cat.includes("code")) return "cat-dev";
  if (cat.includes("comm") || cat.includes("chat")) return "cat-comm";
  if (cat.includes("social")) return "cat-social";
  if (cat.includes("media") || cat.includes("entertainment")) return "cat-media";
  if (cat.includes("prod") || cat.includes("plan")) return "cat-prod";
  if (cat.includes("sys") || cat.includes("admin")) return "cat-sys";
  return "cat-default";
}

function formatDuration(seconds) {
  if (!seconds || seconds <= 0) return "0s";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}m ${rem}s`;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
