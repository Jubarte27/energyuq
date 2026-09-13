(function() {
  "use strict";

  // Application State
  const state = {
    data: null,
    benchmarks: [],
    machines: [],
    flatRuns: [],
    currentBenchmark: null,
    currentMachine: null, // null indicates benchmark multi-plot view
    currentScope: "all",  // "all" or specific machine name in multi-plot view
    currentQoI: "energy_uj",
    currentPlotKey: null,
    searchQuery: "",
  };

  // DOM Elements
  const sidebarTree = document.getElementById("sidebar-tree");
  const searchFilter = document.getElementById("search-filter");
  const breadcrumbs = document.getElementById("breadcrumbs");
  const machinePillsBar = document.getElementById("machine-pills-bar");
  const viewContainer = document.getElementById("view-container");
  // const runCounterBadge = document.getElementById("run-counter-badge");
  // const reportDateBadge = document.getElementById("report-date-badge");
  // const btnPrevRun = document.getElementById("btn-prev-run");
  // const btnNextRun = document.getElementById("btn-next-run");
  const btnHelpKbd = document.getElementById("btn-help-kbd");
  const lightboxModal = document.getElementById("lightbox-modal");
  const lightboxImg = document.getElementById("lightbox-img");
  const lightboxCaption = document.getElementById("lightbox-caption");
  const lightboxCloseBtn = document.getElementById("lightbox-close-btn");

  // 1. Initialize Metadata from static JSON file
  async function initData() {
    try {
      const resp = await fetch("report_metadata.json");
      if (!resp.ok) {
        throw new Error("HTTP status " + resp.status + " (" + resp.statusText + ")");
      }
      state.data = await resp.json();
    } catch (err) {
      console.error("Failed to fetch report_metadata.json:", err);
      viewContainer.innerHTML = `
        <div class="plot-showcase" style="border-color: var(--accent-rose);">
          <h2 style="color: var(--accent-rose);">Metadata Loading Error</h2>
          <p style="color: var(--text-secondary); margin-top: 0.5rem;">
            Could not load <code>report_metadata.json</code> (${err.message}).
          </p>
          <p style="color: var(--text-muted); margin-top: 0.75rem; font-size: 0.85rem; line-height: 1.6;">
            If you are opening this file directly from the filesystem (<code>file://</code>), modern web browsers
            restrict local JavaScript <code>fetch()</code> calls due to CORS security policies.<br/><br/>
            To view the interactive dashboard, start a local HTTP server in this directory:<br/>
            <code style="display:inline-block; padding: 0.2rem 0.5rem; background: var(--bg-primary); border: 1px solid var(--border-color); border-radius: 4px; color: var(--accent-blue); margin: 0.4rem 0;">
              python3 -m http.server 8000
            </code><br/>
            then open <strong style="color: var(--accent-cyan);">http://localhost:8000</strong> in your browser.
          </p>
        </div>
      `;
      sidebarTree.innerHTML = `<p style="padding: 1rem; color: var(--text-muted); font-size: 0.85rem;">Metadata offline.</p>`;
      return;
    }

    // Populate state lists
    state.benchmarks = state.data.benchmarks || [];
    state.machines = state.data.machines || [];
    state.primaryQoI = state.data.primary_qoi || "energy_uj";
    state.currentQoI = state.primaryQoI;

    // Build flat runs index for Previous/Next navigation
    state.flatRuns = [];
    state.benchmarks.forEach(bench => {
      const bData = state.data.benchmark_sections[bench];
      if (bData && bData.runs) {
        Object.keys(bData.runs).forEach(mach => {
          state.flatRuns.push({
            benchmark: bench,
            machine: mach,
            tag: bData.runs[mach].tag,
          });
        });
      }
    });

    if (state.data.generated_at_human) {
      // reportDateBadge.textContent = state.data.generated_at_human;
    }

    // Parse initial URL hash or set default view
    parseHash();

    // Render index tree
    renderSidebar();

    // Render view
    renderCurrentView();

    // Setup event listeners
    setupEventListeners();
  }

  // 2. Hash Routing
  function parseHash() {
    const hash = window.location.hash.replace(/^#/, "");
    if (!hash) {
      if (state.benchmarks.length > 0) {
        state.currentBenchmark = state.benchmarks[0];
        state.currentMachine = null; // default to benchmark overview
        state.currentScope = "all";
        state.currentPlotKey = "dashboard";
      }
      return;
    }

    const params = new URLSearchParams(hash);
    const b = params.get("benchmark") || params.get("b");
    const m = params.get("machine") || params.get("m");
    const s = params.get("scope") || params.get("s");
    const p = params.get("plot") || params.get("p");
    const q = params.get("qoi") || params.get("q");

    if (b && (state.benchmarks.includes(b) || b === "Global")) {
      state.currentBenchmark = b;
    } else if (state.benchmarks.length > 0) {
      state.currentBenchmark = state.benchmarks[0];
    }

    state.currentMachine = (m && m !== "null" && m !== "all") ? m : null;
    state.currentScope = s || "all";
    if (p) state.currentPlotKey = p;
    if (q) state.currentQoI = q;
  }

  function updateHash() {
    const params = new URLSearchParams();
    if (state.currentBenchmark) params.set("b", state.currentBenchmark);
    if (state.currentMachine) params.set("m", state.currentMachine);
    if (state.currentScope && state.currentScope !== "all") params.set("s", state.currentScope);
    if (state.currentPlotKey) params.set("p", state.currentPlotKey);
    if (state.currentQoI) params.set("q", state.currentQoI);
    window.history.replaceState(null, "", "#" + params.toString());
  }

  // 3. Navigation Controls
  function navigateTo(benchmark, machine, plotKey, scope) {
    state.currentBenchmark = benchmark;
    state.currentMachine = machine;
    if (scope !== undefined && scope !== null) {
      state.currentScope = scope;
    }
    if (plotKey) {
      state.currentPlotKey = plotKey;
    } else {
      if (!machine) {
        state.currentPlotKey = "dashboard";
      } else {
        const run = getActiveRunData();
        if (run && run.plots && run.plots.length > 0) {
          state.currentPlotKey = run.plots[0].key;
        } else {
          state.currentPlotKey = null;
        }
      }
    }
    updateHash();
    renderSidebar();
    renderCurrentView();
  }

  function getActiveRunIndex() {
    if (!state.currentMachine) return -1;
    return state.flatRuns.findIndex(r => r.benchmark === state.currentBenchmark && r.machine === state.currentMachine);
  }

  function stepRun(delta) {
    const idx = getActiveRunIndex();
    if (idx === -1) {
      if (state.flatRuns.length > 0) {
        const target = delta > 0 ? state.flatRuns[0] : state.flatRuns[state.flatRuns.length - 1];
        navigateTo(target.benchmark, target.machine, null, null);
      }
      return;
    }
    let newIdx = idx + delta;
    if (newIdx < 0) newIdx = state.flatRuns.length - 1;
    if (newIdx >= state.flatRuns.length) newIdx = 0;
    const target = state.flatRuns[newIdx];
    navigateTo(target.benchmark, target.machine, null, null);
  }

  function getActiveRunData() {
    if (!state.currentBenchmark || !state.currentMachine) return null;
    const b = state.data.benchmark_sections[state.currentBenchmark];
    return b && b.runs ? b.runs[state.currentMachine] : null;
  }

  function getActiveBenchmarkData() {
    if (!state.currentBenchmark) return null;
    return state.data.benchmark_sections[state.currentBenchmark];
  }

  // 4. Sidebar Index Rendering
  function renderSidebar() {
    const q = state.searchQuery.toLowerCase().trim();
    let html = "";

    // Global Cross-Benchmark (if available)
    if (state.data.global_section && state.data.global_section.has_global) {
      const isAct = state.currentBenchmark === "Global";
      html += `
        <div class="tree-node">
          <div class="tree-bench-header ${isAct ? "active" : ""}" data-nav-bench="Global" data-nav-mach="">
            <div class="bench-name-wrap">
              <span>🌐 Global Cross-Benchmark</span>
            </div>
          </div>
        </div>
      `;
    }

    html += `<div class="tree-section-title"><span>Benchmarks (${state.benchmarks.length})</span></div>`;

    state.benchmarks.forEach(bench => {
      const bData = state.data.benchmark_sections[bench] || {};
      const machines = bData.machines || [];

      // Search filter match check
      const benchMatch = bench.toLowerCase().includes(q);
      const matchedMachines = machines.filter(m => m.toLowerCase().includes(q));
      if (q && !benchMatch && matchedMachines.length === 0) return;

      const isBenchActive = state.currentBenchmark === bench;
      const isBenchOverviewActive = isBenchActive && !state.currentMachine;

      html += `
        <div class="tree-node">
          <div class="tree-bench-header ${isBenchOverviewActive ? "active" : ""}" data-nav-bench="${bench}" data-nav-mach="">
            <div class="bench-name-wrap">
              <span>📊 ${bench}</span>
            </div>
            <span class="bench-badge">${machines.length} runs</span>
          </div>
          <div class="tree-children">
            <div class="tree-item ${isBenchOverviewActive ? "active" : ""}" data-nav-bench="${bench}" data-nav-mach="">
              <span>Comparative Overview</span>
            </div>
            ${machines.map(mach => {
              if (q && !benchMatch && !mach.toLowerCase().includes(q)) return "";
              const isMachActive = isBenchActive && state.currentMachine === mach;
              const runInfo = bData.runs ? bData.runs[mach] : null;
              const sampleCount = runInfo ? runInfo.sample_count : "";
              return `
                <div class="tree-item ${isMachActive ? "active" : ""}" data-nav-bench="${bench}" data-nav-mach="${mach}">
                  <span>💻 ${mach}</span>
                  ${sampleCount ? `<span class="bench-badge">${sampleCount} pts</span>` : ""}
                </div>
              `;
            }).join("")}
          </div>
        </div>
      `;
    });

    sidebarTree.innerHTML = html;

    // Attach click handlers to sidebar items
    sidebarTree.querySelectorAll("[data-nav-bench]").forEach(el => {
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        const b = el.getAttribute("data-nav-bench");
        const m = el.getAttribute("data-nav-mach") || null;
        navigateTo(b, m, null, null);
      });
    });
  }

  // 5. Main View Rendering
  function renderCurrentView() {
    // Update Run Counter
    const runIdx = getActiveRunIndex();
    if (runIdx !== -1) {
      // runCounterBadge.textContent = `Run ${runIdx + 1} of ${state.flatRuns.length}`;
      // btnPrevRun.disabled = false;
      // btnNextRun.disabled = false;
    } else {
      // runCounterBadge.textContent = `${state.benchmarks.length} Benchmarks`;
    }

    renderMachinePills();

    if (!state.currentMachine) {
      renderBenchmarkView();
    } else {
      renderIndividualRunView();
    }
  }

  function renderMachinePills() {
    const bData = getActiveBenchmarkData();
    if (!bData) {
      machinePillsBar.innerHTML = "";
      return;
    }
    const machines = bData.machines || [];
    let pillsHtml = `
      <button class="machine-pill ${!state.currentMachine ? "active" : ""}" data-nav-mach="">
        Multi-Run Overview
      </button>
    `;
    machines.forEach(mach => {
      const isAct = state.currentMachine === mach;
      pillsHtml += `
        <button class="machine-pill ${isAct ? "active" : ""}" data-nav-mach="${mach}">
          ${mach}
        </button>
      `;
    });
    machinePillsBar.innerHTML = pillsHtml;

    machinePillsBar.querySelectorAll("[data-nav-mach]").forEach(btn => {
      btn.addEventListener("click", () => {
        const m = btn.getAttribute("data-nav-mach") || null;
        navigateTo(state.currentBenchmark, m, null, null);
      });
    });
  }

  // 5.1 Benchmark Multi-Plot View with Scope Toggle / Swap
  function renderBenchmarkView() {
    const bench = state.currentBenchmark;
    breadcrumbs.innerHTML = `
      <span class="breadcrumb-item">Benchmarks</span>
      <span class="breadcrumb-sep">/</span>
      <span class="breadcrumb-item">${bench}</span>
      <span class="breadcrumb-sep">/</span>
      <span class="breadcrumb-active">Comparative Overview</span>
    `;

    const bData = getActiveBenchmarkData();
    if (!bData) return;

    const multiPlots = bData.multi_plots || {};
    const plotKeys = Object.keys(multiPlots);
    const machines = bData.machines || [];

    if (!state.currentPlotKey || !multiPlots[state.currentPlotKey]) {
      state.currentPlotKey = plotKeys[0] || "dashboard";
    }

    const activePlot = multiPlots[state.currentPlotKey] || {
      title: "Multi-Run Comparative Dashboard",
      tab_label: "Dashboard",
      description: "Overview of benchmark performance across evaluated machines.",
      scopes: {},
    };

    // Ensure valid current scope
    const availableScopes = activePlot.scopes ? Object.keys(activePlot.scopes) : ["all"];
    if (!availableScopes.includes(state.currentScope)) {
      state.currentScope = "all";
    }

    const activeScopeData = (activePlot.scopes && activePlot.scopes[state.currentScope])
      || (activePlot.scopes && activePlot.scopes["all"])
      || { file: activePlot.file || "", title: activePlot.title };

    const displayTitle = activeScopeData.title || activePlot.title;
    const displayFile = activeScopeData.file || activePlot.file || "";

    viewContainer.innerHTML = `
      <!-- Multi-Plot Tab Bar -->
      <div class="tabs-wrapper">
        <div class="tabs-header">
          <span class="tabs-title">Multi-Run Comparison Plots</span>
        </div>
        <div class="plot-pills">
          ${plotKeys.map(k => {
            const p = multiPlots[k];
            const isAct = state.currentPlotKey === k;
            return `<button class="plot-pill ${isAct ? "active" : ""}" data-plot-key="${k}">${p.tab_label || p.title}</button>`;
          }).join("")}
        </div>
      </div>

      <!-- Machine Scope Toggle Bar (Toggle swap between All Machines and specific machines) -->
      <div class="scope-toggle-bar">
        <div class="scope-left">
          <span class="scope-label">Machine Scope:</span>
          <div class="scope-pills">
            <button class="scope-pill ${state.currentScope === 'all' ? 'active' : ''}" data-scope="all">
              🌐 All Machines
            </button>
            ${machines.map(m => `
              <button class="scope-pill ${state.currentScope === m ? 'active' : ''}" data-scope="${m}">
                💻 ${m}
              </button>
            `).join('')}
          </div>
        </div>
        <button class="icon-btn" id="btn-swap-scope" title="Toggle swap between All Machines and individual machine (Shortcut: S)">
          ⇄ Swap Scope
        </button>
      </div>

      <!-- Active Plot Showcase Card -->
      <div class="plot-showcase">
        <div class="showcase-header">
          <div class="showcase-title-area">
            <span class="showcase-category">${activePlot.category || "Benchmark Multi-Run Analysis"}</span>
            <h3 class="showcase-title">${displayTitle}</h3>
          </div>
          <div class="showcase-actions">
            <a href="${displayFile}" target="_blank" class="icon-btn" title="Open full resolution in new tab">↗ Full Res</a>
            <button class="icon-btn" id="btn-zoom-lightbox" title="Enlarge in lightbox">🔍 Lightbox</button>
          </div>
        </div>
        <div class="image-stage" id="plot-stage">
          <img src="${displayFile}" alt="${displayTitle}" id="main-plot-img" />
        </div>
        <div class="plot-description-box">
          <strong>Visual Analysis:</strong> ${activePlot.description || "Comparison of execution behavior across distinct architectures."}
        </div>
      </div>

      <!-- KPI Summary Strip -->
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-title">Active Benchmark</div>
          <div class="kpi-value" style="color: var(--accent-cyan);">${bench}</div>
          <div class="kpi-subtext">${machines.length} Machines Evaluated</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">Active Scope</div>
          <div class="kpi-value" style="color: var(--accent-indigo);">
            ${state.currentScope === "all" ? "All Machines Combined" : `Machine: ${state.currentScope}`}
          </div>
          <div class="kpi-subtext">Toggle swap scope below</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">Evaluated Machines</div>
          <div class="kpi-value" style="font-size: 1rem; line-height: 1.6;">${machines.length ? machines.join(", ") : "None"}</div>
          <div class="kpi-subtext">Cross-Architecture Analysis</div>
        </div>
      </div>
    `;

    // Attach plot tab listeners
    viewContainer.querySelectorAll("[data-plot-key]").forEach(btn => {
      btn.addEventListener("click", () => {
        state.currentPlotKey = btn.getAttribute("data-plot-key");
        updateHash();
        renderBenchmarkView();
      });
    });

    // Attach machine scope listeners
    viewContainer.querySelectorAll("[data-scope]").forEach(btn => {
      btn.addEventListener("click", () => {
        state.currentScope = btn.getAttribute("data-scope");
        updateHash();
        renderBenchmarkView();
      });
    });

    // Attach swap scope button
    const btnSwap = document.getElementById("btn-swap-scope");
    if (btnSwap) {
      btnSwap.onclick = () => {
        if (state.currentScope === "all") {
          state.currentScope = machines.length > 0 ? machines[0] : "all";
        } else {
          // cycle to next machine or all
          const idx = machines.indexOf(state.currentScope);
          if (idx === machines.length - 1) {
            state.currentScope = "all";
          } else {
            state.currentScope = machines[idx + 1];
          }
        }
        updateHash();
        renderBenchmarkView();
      };
    }

    setupStageLightbox(displayFile, displayTitle);
  }

  // 5.2 Individual Run View
  function renderIndividualRunView() {
    const bench = state.currentBenchmark;
    const mach = state.currentMachine;
    const run = getActiveRunData();

    breadcrumbs.innerHTML = `
      <span class="breadcrumb-item">Benchmarks</span>
      <span class="breadcrumb-sep">/</span>
      <span class="breadcrumb-item">${bench}</span>
      <span class="breadcrumb-sep">/</span>
      <span class="breadcrumb-active">Machine: ${mach}</span>
    `;

    if (!run) {
      viewContainer.innerHTML = `<div class="plot-showcase"><h3>Run data for ${mach} not found.</h3></div>`;
      return;
    }

    const qois = Object.keys(run.qois_data || {});
    if (qois.length > 0 && !qois.includes(state.currentQoI)) {
      state.currentQoI = qois[0];
    }

    const activeQoIData = run.qois_data ? run.qois_data[state.currentQoI] : null;
    const stats = activeQoIData ? activeQoIData.sample_statistics : {};
    const bestEnergy = run.qois_data && run.qois_data.energy_uj ? run.qois_data.energy_uj.best_config_min : null;
    const bestTime = run.qois_data && run.qois_data.time ? run.qois_data.time.best_config_min : null;

    const allPlots = run.plots || [];
    const filteredPlots = allPlots.filter(p => !p.qoi || p.qoi === state.currentQoI);

    const plotChoices = [...filteredPlots, {
      key: "data_preview_table",
      title: "Evaluated Samples Data Table",
      tab_label: "Data Table",
      category: "Raw Data",
      is_table: true,
      description: "Tabular inspection of sample parameter evaluations and resulting QoI metrics."
    }];

    if (!state.currentPlotKey || !plotChoices.find(p => p.key === state.currentPlotKey)) {
      state.currentPlotKey = plotChoices[0] ? plotChoices[0].key : null;
    }

    const activePlot = plotChoices.find(p => p.key === state.currentPlotKey) || plotChoices[0];

    let optEnergyStr = "N/A";
    if (bestEnergy) {
      const parts = (run.input_params || []).filter(p => p in bestEnergy).map(p => `${p}=${bestEnergy[p]}`);
      optEnergyStr = parts.join(", ") || "Recorded";
    }

    let optTimeStr = "N/A";
    if (bestTime) {
      const parts = (run.input_params || []).filter(p => p in bestTime).map(p => `${p}=${bestTime[p]}`);
      optTimeStr = parts.join(", ") || "Recorded";
    }

    viewContainer.innerHTML = `
      <!-- KPI Cards Strip -->
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-title">Run Identification</div>
          <div class="kpi-value" style="font-size: 1.15rem; color: var(--accent-cyan);">${run.tag}</div>
          <div class="kpi-subtext">${run.sample_count} Samples | ${run.iterations} Iterations</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">Best Config (Min Energy)</div>
          <div class="kpi-value" style="font-size: 1.05rem; color: var(--accent-green);">${optEnergyStr}</div>
          <div class="kpi-subtext">${bestEnergy && bestEnergy.energy_uj ? (bestEnergy.energy_uj * 1e-6).toFixed(3) + " J" : "Evaluated Minimum"}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">Best Config (Min Time)</div>
          <div class="kpi-value" style="font-size: 1.05rem; color: var(--accent-amber);">${optTimeStr}</div>
          <div class="kpi-subtext">${bestTime && bestTime.time ? Number(bestTime.time).toFixed(4) + " s" : "Evaluated Minimum"}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">${_formatQoILabel(state.currentQoI)} Range</div>
          <div class="kpi-value" style="font-size: 1.05rem;">
            ${stats.min != null ? _formatMetricVal(stats.min, state.currentQoI) : "--"}
          </div>
          <div class="kpi-subtext">Mean: ${stats.mean != null ? _formatMetricVal(stats.mean, state.currentQoI) : "--"} | Max: ${stats.max != null ? _formatMetricVal(stats.max, state.currentQoI) : "--"}</div>
        </div>
      </div>

      <!-- Plot Selection Tabs & QoI Switcher -->
      <div class="tabs-wrapper">
        <div class="tabs-header">
          <span class="tabs-title">Diagnostic & Sensitivity Figures</span>
          <div class="qoi-selector">
            <span style="font-size: 0.78rem; color: var(--text-muted); font-weight: 600;">Metric:</span>
            ${qois.map(q => `
              <button class="qoi-btn ${state.currentQoI === q ? "active" : ""}" data-qoi-key="${q}">
                ${_formatQoILabel(q)}
              </button>
            `).join("")}
          </div>
        </div>
        <div class="plot-pills">
          ${plotChoices.map(p => {
            const isAct = state.currentPlotKey === p.key;
            return `
              <button class="plot-pill ${isAct ? "active" : ""}" data-plot-key="${p.key}">
                ${p.tab_label || p.title}
              </button>
            `;
          }).join("")}
        </div>
      </div>

      <!-- Focused Plot Showcase -->
      <div class="plot-showcase">
        <div class="showcase-header">
          <div class="showcase-title-area">
            <span class="showcase-category">${activePlot.category || "Evaluation Analysis"}</span>
            <h3 class="showcase-title">${activePlot.title}</h3>
          </div>
          <div class="showcase-actions">
            ${!activePlot.is_table ? `
              <a href="${activePlot.file}" target="_blank" class="icon-btn" title="Open in new tab">↗ Full Res</a>
              <button class="icon-btn" id="btn-zoom-lightbox" title="Enlarge in lightbox">🔍 Lightbox</button>
            ` : ""}
          </div>
        </div>

        ${!activePlot.is_table ? `
          <div class="image-stage" id="plot-stage">
            <img src="${activePlot.file}" alt="${activePlot.title}" id="main-plot-img" />
          </div>
        ` : `
          <div class="data-table-container">
            ${renderSamplesTable(run)}
          </div>
        `}

        <div class="plot-description-box">
          <strong>Analysis Description:</strong> ${activePlot.description}
        </div>
      </div>
    `;

    viewContainer.querySelectorAll("[data-qoi-key]").forEach(btn => {
      btn.addEventListener("click", () => {
        state.currentQoI = btn.getAttribute("data-qoi-key");
        updateHash();
        renderIndividualRunView();
      });
    });

    viewContainer.querySelectorAll("[data-plot-key]").forEach(btn => {
      btn.addEventListener("click", () => {
        state.currentPlotKey = btn.getAttribute("data-plot-key");
        updateHash();
        renderIndividualRunView();
      });
    });

    if (activePlot && !activePlot.is_table) {
      setupStageLightbox(activePlot.file, activePlot.title);
    }
  }

  function _formatCell(val) {
    if (val == null) return "";
    if (typeof val === "number") {
      return Number.isInteger(val) ? val : (val < 0.001 || val > 10000 ? val.toExponential(3) : val.toFixed(4));
    }
    return val;
  }

  function renderSamplesTable(run) {
    const samples = run.samples_preview || [];
    if (samples.length === 0) {
      return '<p style="padding: 1.5rem; color: var(--text-muted);">No evaluated sample records available for preview.</p>';
    }
    const cols = Object.keys(samples[0]);
    const ths = cols.map(c => '<th>' + c + '</th>').join('');
    const trs = samples.map(row => {
      const tds = cols.map(c => '<td>' + _formatCell(row[c]) + '</td>').join('');
      return '<tr>' + tds + '</tr>';
    }).join('');
    return '<table class="data-table"><thead><tr>' + ths + '</tr></thead><tbody>' + trs + '</tbody></table>';
  }

  // 6. Lightbox & Modals
  function setupStageLightbox(file, title) {
    const stage = document.getElementById("plot-stage");
    const btnZoom = document.getElementById("btn-zoom-lightbox");
    const open = () => {
      lightboxImg.src = file;
      lightboxCaption.textContent = title;
      lightboxModal.classList.add("active");
    };
    if (stage) stage.onclick = open;
    if (btnZoom) btnZoom.onclick = open;
  }

  function closeLightbox() {
    lightboxModal.classList.remove("active");
  }

  // 7. Event Listeners & Shortcuts
  function setupEventListeners() {
    // btnPrevRun.onclick = () => stepRun(-1);
    // btnNextRun.onclick = () => stepRun(1);

    searchFilter.oninput = (e) => {
      state.searchQuery = e.target.value;
      renderSidebar();
    };

    lightboxCloseBtn.onclick = closeLightbox;
    lightboxModal.onclick = (e) => {
      if (e.target === lightboxModal) closeLightbox();
    };

    btnHelpKbd.onclick = () => {
      alert(
        "EnergyUQ Navigation Shortcuts:\\n\\n" +
        "• ArrowLeft / ArrowRight : Navigate to Previous / Next Machine Run\\n" +
        "• ArrowUp / ArrowDown   : Switch between Diagnostic Plot Tabs\\n" +
        "• S                     : Toggle swap Machine Scope (All vs Machine)\\n" +
        "• Escape                : Close Lightbox Zoom\\n" +
        "• /                     : Focus Search Input"
      );
    };

    // Keyboard navigation
    window.addEventListener("keydown", (e) => {
      if (e.target === searchFilter) return;

      if (e.key === "ArrowLeft") {
        e.preventDefault();
        stepRun(-1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        stepRun(1);
      } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        const pills = Array.from(viewContainer.querySelectorAll(".plot-pill"));
        if (pills.length > 1) {
          e.preventDefault();
          const curIdx = pills.findIndex(p => p.classList.contains("active"));
          const nextIdx = e.key === "ArrowDown"
            ? (curIdx + 1) % pills.length
            : (curIdx - 1 + pills.length) % pills.length;
          pills[nextIdx].click();
        }
      } else if (e.key.toLowerCase() === "s") {
        const btnSwap = document.getElementById("btn-swap-scope");
        if (btnSwap) {
          e.preventDefault();
          btnSwap.click();
        }
      } else if (e.key === "Escape") {
        closeLightbox();
      } else if (e.key === "/") {
        e.preventDefault();
        searchFilter.focus();
      }
    });

    window.addEventListener("hashchange", () => {
      parseHash();
      renderSidebar();
      renderCurrentView();
    });
  }

  // Helpers
  function _formatQoILabel(qoi) {
    const map = {
      "energy_uj": "Energy (μJ)",
      "energy_j": "Energy (J)",
      "time": "Time (s)",
      "power_w": "Power (W)",
      "edp_j_s": "EDP (J·s)",
    };
    return map[qoi] || qoi;
  }

  function _formatMetricVal(val, qoi) {
    if (val == null) return "--";
    if (qoi === "energy_uj") {
      return (val * 1e-6).toFixed(3) + " J (" + Math.round(val) + " μJ)";
    }
    if (qoi === "time") {
      return Number(val).toFixed(4) + " s";
    }
    if (qoi === "power_w") {
      return Number(val).toFixed(2) + " W";
    }
    return typeof val === "number" ? val.toFixed(4) : val;
  }

  // Launch application
  initData();
})();

