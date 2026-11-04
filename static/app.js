/* ActivityWatch Classifier Frontend Application */

let currentCategories = [];
let currentTaxonomy = {};
let currentCategoryDist = {};
let currentSubcategoryDist = {};
let chartMode = "main"; // 'main' or 'sub'
let categoryChart = null;
let lastEvents = [];

// Initialize Theme immediately to prevent theme flashing
initTheme();

document.addEventListener("DOMContentLoaded", () => {
  checkEmbedMode();
  initApp();
  setupEventListeners();
});

function initTheme() {
  const saved = localStorage.getItem("aw-theme");
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  const isDark = saved ? saved === "dark" : (prefersDark !== false);
  if (isDark) {
    document.documentElement.classList.add("dark");
  } else {
    document.documentElement.classList.remove("dark");
  }
}

function toggleTheme() {
  const isDark = document.documentElement.classList.toggle("dark");
  localStorage.setItem("aw-theme", isDark ? "dark" : "light");
  updateChartTheme();
}

function checkEmbedMode() {
  const params = new URLSearchParams(window.location.search);
  const isEmbedded = params.get("embed") === "true" || window.self !== window.top;
  if (isEmbedded) {
    document.body.classList.add("is-embedded");
  }
  const embedInput = document.getElementById("embed-url-input");
  if (embedInput) {
    embedInput.value = `${window.location.origin}/?embed=true`;
  }
}

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

    // ActivityWatch indicator & direct links
    const awIndicator = document.getElementById("aw-indicator");
    const awText = document.getElementById("aw-text");
    const serverUrl = data.activitywatch.server_url || "http://localhost:5600";

    const navActivity = document.getElementById("nav-aw-activity");
    const navTimeline = document.getElementById("nav-aw-timeline");
    const navRaw = document.getElementById("nav-aw-raw");
    const btnRawLink = document.getElementById("btn-aw-raw-data-link");
    const btnActivityLink = document.getElementById("btn-aw-activity-tab-link");

    if (navActivity) navActivity.href = `${serverUrl}/#/activity/classified/view/summary`;
    if (navTimeline) navTimeline.href = `${serverUrl}/#/timeline`;
    if (navRaw) navRaw.href = `${serverUrl}/#/raw-data`;
    if (btnRawLink) btnRawLink.href = `${serverUrl}/#/raw-data`;
    if (btnActivityLink) btnActivityLink.href = `${serverUrl}/#/activity/classified/view/summary`;

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

    // Git tracking status
    const gitBadge = document.getElementById("git-status-badge");
    const gitInfo = document.getElementById("git-last-commit-info");
    if (gitBadge && data.git_status) {
      const gs = data.git_status;
      if (gs.is_modified_or_untracked) {
        gitBadge.className = "px-1.5 py-0.5 rounded text-[9px] font-mono bg-amber-100 dark:bg-amber-950 text-amber-700 dark:text-amber-300 border border-amber-300 dark:border-amber-800";
        gitBadge.textContent = "Git: Uncommitted";
      } else {
        const shortHash = gs.commit_hash || "Clean";
        gitBadge.className = "px-1.5 py-0.5 rounded text-[9px] font-mono bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800";
        gitBadge.textContent = `Git: ${shortHash}`;
      }
      if (gitInfo && gs.last_commit) {
        gitInfo.textContent = gs.last_commit;
      }
    }

    currentCategories = data.classifier.all_configured_categories || [];
    currentTaxonomy = data.classifier.taxonomy || data.gemma_oracle.taxonomy || {};

    // Last run stats if available
    if (data.last_run_stats && data.last_run_stats.total_events_processed) {
      document.getElementById("stat-fast-calls").textContent = data.last_run_stats.fast_classifier_calls || 0;
      document.getElementById("stat-gemma-calls").textContent = data.last_run_stats.gemma_oracle_calls || 0;
      currentCategoryDist = data.last_run_stats.category_distribution || {};
      currentSubcategoryDist = data.last_run_stats.subcategory_distribution || {};
      refreshChart();
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
    const afkBuckets = data.afk_buckets || {};

    const afkDetails = document.getElementById("afk-guard-details");
    if (afkDetails) {
      const afkNames = Object.keys(afkBuckets);
      if (afkNames.length > 0) {
        afkDetails.textContent = `Active guard connected to: ${afkNames.join(", ")}`;
      } else {
        afkDetails.textContent = "Accounts for AFK bucket; only classifies activities when user wasn't AFK.";
      }
    }

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
      item.className = "flex items-center justify-between p-2 rounded-lg bg-white dark:bg-slate-900/80 hover:bg-slate-100 dark:hover:bg-slate-800/80 border border-slate-200 dark:border-slate-800 cursor-pointer transition";
      item.innerHTML = `
        <div class="flex items-center space-x-3 overflow-hidden">
          <input type="checkbox" name="source-bucket" value="${bid}" ${isWindowOrWeb ? "checked" : ""} 
            class="accent-[#4DBA87] rounded border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-950 focus:ring-0" />
          <div class="truncate">
            <div class="text-xs font-mono text-slate-800 dark:text-slate-200 truncate">${bid}</div>
            <div class="text-[10px] text-slate-500 dark:text-slate-400">${client} &bull; ${type}</div>
          </div>
        </div>
        <span class="text-[10px] px-2 py-0.5 rounded font-mono bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 ml-2">
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
  const timeRange = document.getElementById("select-time-range")
    ? document.getElementById("select-time-range").value
    : "count";
  const threshold = parseFloat(document.getElementById("confidence-threshold-slider").value);
  const filterAfk = document.getElementById("toggle-filter-afk")
    ? document.getElementById("toggle-filter-afk").checked
    : true;
  const collectTraining = document.getElementById("toggle-collect-training")
    ? document.getElementById("toggle-collect-training").checked
    : true;
  const autoCommit = document.getElementById("toggle-auto-commit")
    ? document.getElementById("toggle-auto-commit").checked
    : false;

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
        time_range: timeRange,
        confidence_threshold: threshold,
        filter_afk: filterAfk,
        collect_training_data: collectTraining,
        auto_commit_git: autoCommit,
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

    if (results.afk_filtering_enabled && results.afk_events_filtered_out > 0) {
      document.getElementById("target-bucket-info").textContent = 
        `${results.total_events_processed} active events categorized (filtered out ${results.afk_events_filtered_out} idle/AFK events)`;
    }

    // Update chart
    currentCategoryDist = results.category_distribution || {};
    currentSubcategoryDist = results.subcategory_distribution || {};
    refreshChart();

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
      subcategory: e.data?.subcategory || "General Activity",
      full_category: e.data?.full_category || `${e.data?.category || "Uncategorized"} > General`,
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
    tr.className = "hover:bg-slate-50 dark:hover:bg-slate-900/60 transition group";

    const dateStr = ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : "--:--";
    const durationStr = formatDuration(ev.duration);
    const catClass = getCategoryBadgeClass(ev.category);
    const isGemma = (ev.classified_by || "").includes("gemma");

    tr.innerHTML = `
      <td class="px-4 py-3 whitespace-nowrap text-slate-500 dark:text-slate-400 font-mono text-[11px]">
        <div>${dateStr}</div>
        <div class="text-[10px] text-slate-400 dark:text-slate-500">${durationStr}</div>
      </td>
      <td class="px-4 py-3 font-semibold text-slate-800 dark:text-slate-200 whitespace-nowrap">
        <div class="flex items-center space-x-1.5">
          <i class="fa-solid fa-window-maximize text-slate-400 dark:text-slate-500 text-[10px]"></i>
          <span>${escapeHtml(ev.app || "Browser/App")}</span>
        </div>
      </td>
      <td class="px-4 py-3 max-w-xs truncate text-slate-700 dark:text-slate-300" title="${escapeHtml(ev.title || ev.url || "")}">
        <div class="truncate font-medium">${escapeHtml(ev.title || "No Title")}</div>
        ${ev.url ? `<div class="truncate text-[10px] text-teal-600 dark:text-teal-400 font-mono">${escapeHtml(ev.url)}</div>` : ""}
      </td>
      <td class="px-4 py-3 whitespace-nowrap">
        <div class="flex flex-col space-y-1">
          <span class="category-tag ${catClass}">${escapeHtml(ev.category)}</span>
          <span class="text-[10px] text-slate-500 dark:text-slate-400 font-medium flex items-center pl-1">
            <i class="fa-solid fa-angle-right text-[8px] text-slate-400 mr-1"></i>
            ${escapeHtml(ev.subcategory || "General")}
          </span>
        </div>
      </td>
      <td class="px-4 py-3 whitespace-nowrap font-mono">
        <div class="flex items-center space-x-2">
          <span class="${ev.confidence >= 0.75 ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'} font-bold">
            ${(ev.confidence * 100).toFixed(0)}%
          </span>
          <div class="w-12 bg-slate-200 dark:bg-slate-800 rounded-full h-1.5">
            <div class="${ev.confidence >= 0.75 ? 'bg-emerald-500' : 'bg-amber-500'} h-1.5 rounded-full" 
              style="width: ${Math.min(100, ev.confidence * 100)}%"></div>
          </div>
        </div>
      </td>
      <td class="px-4 py-3 whitespace-nowrap">
        <span class="px-2 py-0.5 rounded text-[10px] font-mono ${
          isGemma ? 'bg-purple-100 dark:bg-purple-950 text-purple-700 dark:text-purple-300 border border-purple-200 dark:border-purple-800/60' : 'bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800/60'
        }">
          <i class="fa-solid ${isGemma ? 'fa-wand-magic-sparkles' : 'fa-bolt'} mr-1"></i>
          ${isGemma ? 'Gemma 4 E2B' : 'Fast Classifier'}
        </span>
      </td>
      <td class="px-4 py-3 whitespace-nowrap text-right">
        <button onclick="openCorrectionDialog(${idx})" class="text-xs px-2.5 py-1 rounded bg-slate-100 hover:bg-[#4DBA87] hover:text-white dark:bg-slate-800 dark:hover:bg-[#4DBA87] text-slate-700 dark:text-slate-300 transition font-medium">
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

  const currentFull = ev.full_category || `${ev.category} > ${ev.subcategory || 'General'}`;

  // Build flattened taxonomy list
  let options = [];
  for (const [cat, subcats] of Object.entries(currentTaxonomy)) {
    for (const sub of subcats) {
      options.push(`${cat} > ${sub}`);
    }
  }
  if (options.length === 0) {
    options = currentCategories.map((c) => `${c} > General`);
  }

  const promptOptions = options.map((opt, i) => `${i + 1}. ${opt}`).join("\n");
  const choice = prompt(
    `Reclassify Activity:\nApp: ${ev.app}\nTitle: ${ev.title}\n\nCurrent: ${currentFull}\n\nEnter number or "Category > Subcategory":\n${promptOptions}`,
    currentFull
  );

  if (!choice || choice.trim() === "") return;

  let chosenPath = choice.trim();
  const num = parseInt(chosenPath, 10);
  if (!isNaN(num) && num >= 1 && num <= options.length) {
    chosenPath = options[num - 1];
  }

  let cat = chosenPath;
  let sub = "General";
  if (chosenPath.includes(" > ")) {
    const parts = chosenPath.split(" > ");
    cat = parts[0].trim();
    sub = parts[1].trim();
  }

  const featureText = `app:${ev.app} title:${ev.title} url:${ev.url}`;
  submitCorrection(featureText, cat, sub, idx);
}

async function submitCorrection(featureText, correctCategory, correctSubcategory, idx) {
  try {
    const res = await fetch("/api/correct-event", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        feature_text: featureText,
        correct_category: correctCategory,
        correct_subcategory: correctSubcategory,
      }),
    });

    if (res.ok) {
      if (lastEvents[idx]) {
        lastEvents[idx].category = correctCategory;
        lastEvents[idx].subcategory = correctSubcategory;
        lastEvents[idx].full_category = `${correctCategory} > ${correctSubcategory}`;
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
// Chart.js Visualization (ActivityWatch Theme)
// -------------------------------------------------------------
function initChart() {
  const ctx = document.getElementById("categoryChart").getContext("2d");
  const isDark = document.documentElement.classList.contains("dark");
  const legendColor = isDark ? "#cbd5e1" : "#475569";
  const borderColor = isDark ? "#21262d" : "#ffffff";

  categoryChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: [],
      datasets: [
        {
          data: [],
          backgroundColor: [
            "#4DBA87", // AW Primary Teal
            "#35495E", // AW Dark Slate
            "#38bdf8", // Sky
            "#f43f5e", // Rose
            "#eab308", // Amber
            "#a855f7", // Purple
            "#06b6d4", // Cyan
            "#94a3b8", // Slate
            "#10b981", // Emerald
            "#f97316", // Orange
          ],
          borderColor: borderColor,
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
          labels: { color: legendColor, boxWidth: 12, font: { size: 11, family: "'Varela Round', sans-serif" } },
        },
      },
      cutout: "68%",
    },
  });
}

function updateChartTheme() {
  if (!categoryChart) return;
  const isDark = document.documentElement.classList.contains("dark");
  const legendColor = isDark ? "#cbd5e1" : "#475569";
  const borderColor = isDark ? "#21262d" : "#ffffff";

  categoryChart.options.plugins.legend.labels.color = legendColor;
  if (categoryChart.data.datasets && categoryChart.data.datasets[0]) {
    categoryChart.data.datasets[0].borderColor = borderColor;
  }
  categoryChart.update();
}

function refreshChart() {
  const dist = chartMode === "main" ? currentCategoryDist : currentSubcategoryDist;
  updateChart(dist);
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

  // Scope / Time Range Change Handler
  const timeRangeSelect = document.getElementById("select-time-range");
  const limitSelect = document.getElementById("limit-per-bucket");
  const limitLabel = document.getElementById("limit-label");
  if (timeRangeSelect && limitSelect) {
    timeRangeSelect.addEventListener("change", (e) => {
      const val = e.target.value;
      if (val !== "count") {
        if (limitLabel) limitLabel.textContent = "Max Limit (0 = All in Period)";
        limitSelect.value = "0";
      } else {
        if (limitLabel) limitLabel.textContent = "Limit per Source";
        limitSelect.value = "150";
      }
    });
  }

  // Manual Git Commit Button
  const btnGitCommit = document.getElementById("btn-manual-git-commit");
  if (btnGitCommit) {
    btnGitCommit.addEventListener("click", async () => {
      btnGitCommit.disabled = true;
      const originalHtml = btnGitCommit.innerHTML;
      btnGitCommit.innerHTML = `<i class="fa-solid fa-spinner fa-spin text-xs"></i>`;
      try {
        const res = await fetch("/api/git/commit", { method: "POST" });
        const data = await res.json();
        if (data.success) {
          if (data.committed) {
            alert(`Git commit created: [${data.commit_hash}] ${data.message}`);
          } else {
            alert(data.message || "Training data is already up to date in Git.");
          }
          await fetchStatus();
        } else {
          alert("Git commit error: " + (data.error || data.message || "Unknown error"));
        }
      } catch (err) {
        alert("Git commit failed: " + err.message);
      } finally {
        btnGitCommit.disabled = false;
        btnGitCommit.innerHTML = originalHtml;
      }
    });
  }

  // Theme Toggle Button
  const btnTheme = document.getElementById("btn-theme-toggle");
  if (btnTheme) {
    btnTheme.addEventListener("click", toggleTheme);
  }

  // AFK Guard Toggle
  const toggleAfk = document.getElementById("toggle-filter-afk");
  const afkBadge = document.getElementById("afk-guard-badge");
  if (toggleAfk && afkBadge) {
    toggleAfk.addEventListener("change", (e) => {
      if (e.target.checked) {
        afkBadge.textContent = "Enabled";
        afkBadge.className = "ml-2 px-1.5 py-0.5 text-[9px] rounded font-mono bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800";
      } else {
        afkBadge.textContent = "Disabled";
        afkBadge.className = "ml-2 px-1.5 py-0.5 text-[9px] rounded font-mono bg-amber-100 dark:bg-amber-950 text-amber-700 dark:text-amber-300 border border-amber-300 dark:border-amber-800";
      }
    });
  }

  // Sync Categories to ActivityWatch Button
  const btnSyncAwCats = document.getElementById("btn-sync-aw-categories");
  if (btnSyncAwCats) {
    btnSyncAwCats.addEventListener("click", async () => {
      btnSyncAwCats.disabled = true;
      const originalHtml = btnSyncAwCats.innerHTML;
      btnSyncAwCats.innerHTML = `<i class="fa-solid fa-spinner fa-spin text-[9px] text-aw"></i><span>Syncing...</span>`;
      try {
        const res = await fetch("/api/sync-categories", { method: "POST" });
        const data = await res.json();
        if (data.status === "success") {
          btnSyncAwCats.innerHTML = `<i class="fa-solid fa-check text-[9px] text-emerald-500"></i><span>Synced ${data.result.classes_count} Rules!</span>`;
        } else {
          alert("Failed to sync categories: " + (data.detail || "Unknown error"));
          btnSyncAwCats.innerHTML = originalHtml;
        }
      } catch (err) {
        alert("Failed to sync categories: " + err.message);
        btnSyncAwCats.innerHTML = originalHtml;
      } finally {
        setTimeout(() => {
          btnSyncAwCats.disabled = false;
          btnSyncAwCats.innerHTML = originalHtml;
        }, 3000);
      }
    });
  }

  // Direct Embed Modal
  const embedModal = document.getElementById("embed-modal");
  const btnOpenEmbed = document.getElementById("btn-open-embed-modal");
  const btnCloseEmbed = document.getElementById("btn-close-embed");
  const btnDismissEmbed = document.getElementById("btn-dismiss-embed");
  const btnCopyEmbed = document.getElementById("btn-copy-embed-url");

  if (btnOpenEmbed && embedModal) {
    btnOpenEmbed.addEventListener("click", () => embedModal.classList.remove("hidden"));
    btnCloseEmbed?.addEventListener("click", () => embedModal.classList.add("hidden"));
    btnDismissEmbed?.addEventListener("click", () => embedModal.classList.add("hidden"));
    btnCopyEmbed?.addEventListener("click", () => {
      const input = document.getElementById("embed-url-input");
      if (input) {
        input.select();
        navigator.clipboard.writeText(input.value);
        btnCopyEmbed.textContent = "Copied!";
        setTimeout(() => { btnCopyEmbed.textContent = "Copy"; }, 2000);
      }
    });
  }

  // Chart Mode Toggle (Main Categories vs Subcategories)
  const btnMain = document.getElementById("btn-chart-main");
  const btnSub = document.getElementById("btn-chart-sub");
  if (btnMain && btnSub) {
    btnMain.addEventListener("click", () => {
      chartMode = "main";
      btnMain.className = "px-2 py-0.5 rounded bg-aw text-white font-semibold transition";
      btnSub.className = "px-2 py-0.5 rounded text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition";
      refreshChart();
    });
    btnSub.addEventListener("click", () => {
      chartMode = "sub";
      btnSub.className = "px-2 py-0.5 rounded bg-aw text-white font-semibold transition";
      btnMain.className = "px-2 py-0.5 rounded text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition";
      refreshChart();
    });
  }

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
    downloadGemmaFromUI("standard");
  });
  document.getElementById("btn-download-gemma2-ui").addEventListener("click", () => {
    downloadGemmaFromUI("light");
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

async function downloadGemmaFromUI(variant) {
  const btn1 = document.getElementById("btn-download-gemma-ui");
  const btn2 = document.getElementById("btn-download-gemma2-ui");
  const details = document.getElementById("local-model-details");

  btn1.disabled = true;
  btn2.disabled = true;
  const label = variant === "light" ? "Gemma 4 E2B Light (~2.5 GB)" : "Gemma 4 E2B Standard (~3.1 GB)";
  details.textContent = `Downloading ${label} weights directly from Hugging Face... This may take a few minutes depending on your internet connection.`;

  try {
    const res = await fetch("/api/download-gemma", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ variant: variant }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Download failed");
    }

    const data = await res.json();
    alert(`Success! Downloaded Gemma 4 E2B weights to ${data.path}`);
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
