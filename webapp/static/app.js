// Performance Modeling Studio — frontend.

const state = {
  meta: null,
  cfg: null,
  baseCfg: null,           // pristine, for reset
  selectedFile: null,
  evaluating: false,
  evalSeq: 0,              // monotonic, lets us drop stale responses
  charts: { tp: null, pp: null },
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function getDeep(obj, path) {
  return path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
}
function setDeep(obj, path, value) {
  const parts = path.split(".");
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    if (cur[parts[i]] == null || typeof cur[parts[i]] !== "object") cur[parts[i]] = {};
    cur = cur[parts[i]];
  }
  cur[parts[parts.length - 1]] = value;
}
function divisorsOf(n) {
  const out = [];
  for (let i = 1; i <= n; i++) if (n % i === 0) out.push(i);
  return out;
}
function fmtNumber(n, digits = 2) {
  if (n === null || n === undefined || isNaN(n) || !isFinite(n)) return "—";
  if (Math.abs(n) >= 1e12) return (n / 1e12).toFixed(digits) + "T";
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(digits) + "G";
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(digits) + "M";
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(digits) + "k";
  return n.toFixed(digits);
}

// ---------------------------- Setup ----------------------------

async function init() {
  state.meta = await fetch("/api/meta").then((r) => r.json());

  const sel = $("#config-select");
  state.meta.configs.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    sel.appendChild(opt);
  });
  const defaultCfg = state.meta.configs.includes("5p3b_p6_actual.yaml")
    ? "5p3b_p6_actual.yaml"
    : state.meta.configs[0];
  sel.value = defaultCfg;
  sel.addEventListener("change", () => loadConfig(sel.value));

  $("#reset-btn").addEventListener("click", () => loadConfig(state.selectedFile));

  // populate select-row controls that don't depend on config
  fillSelect("model.precision", state.meta.precisions);
  fillSelect("hardware.node_type", state.meta.node_types);
  fillSelect("optimizer.optimizer_type", state.meta.optimizers);
  fillSelect("performance.activation_checkpointing_type", state.meta.act_ckpt_types);

  $("#search-btn").addEventListener("click", runSearch);

  await loadConfig(defaultCfg);
}

async function loadConfig(name) {
  state.selectedFile = name;
  const cfg = await fetch("/api/config/" + encodeURIComponent(name)).then((r) => r.json());
  // ensure parallelism block exists
  if (!cfg.parallelism) {
    const numDevices = (cfg.search && cfg.search.num_devices) || 64;
    cfg.parallelism = {
      tp: 1, pp: 1, cp: 1, dp: numDevices, ep: 1, etp: 1,
      vpp: 1, sp: true, zero_level: 1, n_param_buckets: 5,
    };
  }
  if (!cfg.parallelism.etp) cfg.parallelism.etp = cfg.parallelism.tp || 1;
  if (cfg.parallelism.ep == null) cfg.parallelism.ep = 1;

  // Synthetic moe.enabled flag for UI
  cfg.model.moe = cfg.model.moe || null;
  if (cfg.model.moe) cfg.model.moe.enabled = true;

  state.cfg = cfg;
  state.baseCfg = JSON.parse(JSON.stringify(cfg));
  buildAllControls();
  refreshAll();
  await evaluate();
}

// ---------------------------- Controls ----------------------------

function buildAllControls() {
  $$(".slider-row").forEach((row) => buildSliderRow(row));
  $$(".select-row").forEach((row) => buildSelectRow(row));
  $$(".checkbox-row").forEach((row) => buildCheckboxRow(row));
  updateMoeVisibility();
}

function buildSliderRow(row) {
  const key = row.dataset.key;
  const min = parseFloat(row.dataset.min);
  const max = parseFloat(row.dataset.max);
  const step = parseFloat(row.dataset.step);

  // clear any existing controls (rebuild on config load)
  row.querySelectorAll("input,span.val").forEach((n) => n.remove());

  const range = document.createElement("input");
  range.type = "range";
  range.min = min; range.max = max; range.step = step;
  const num = document.createElement("input");
  num.type = "number";
  num.className = "num-input";
  num.min = min; num.max = max; num.step = step;
  const value = getDeep(state.cfg, key);
  range.value = num.value = value != null ? value : min;
  row.appendChild(range);
  row.appendChild(num);

  const isCheckedChange = step < 1;

  function commit(v) {
    let nv = isCheckedChange ? parseFloat(v) : Math.round(parseFloat(v));
    if (isNaN(nv)) return;
    setDeep(state.cfg, key, nv);
    range.value = nv; num.value = nv;
    onConfigChange(key);
  }
  range.addEventListener("input", (e) => commit(e.target.value));
  num.addEventListener("change", (e) => commit(e.target.value));
}

function buildSelectRow(row) {
  const key = row.dataset.key;
  // For parallelism slots, options come from divisors
  const isParSlot = ["parallelism.tp", "parallelism.pp", "parallelism.cp",
                     "parallelism.ep", "parallelism.etp"].includes(key);

  let select = row.querySelector("select");
  if (!select) {
    select = document.createElement("select");
    row.appendChild(select);
  }

  let options;
  if (isParSlot) {
    options = parallelismOptionsFor(key);
  } else if (key === "model.precision") {
    options = state.meta.precisions;
  } else if (key === "hardware.node_type") {
    options = state.meta.node_types;
  } else if (key === "optimizer.optimizer_type") {
    options = state.meta.optimizers;
  } else if (key === "performance.activation_checkpointing_type") {
    options = state.meta.act_ckpt_types;
  } else {
    options = [];
  }

  select.innerHTML = "";
  options.forEach((v) => {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = v;
    select.appendChild(o);
  });
  const cur = getDeep(state.cfg, key);
  if (cur != null && options.map(String).includes(String(cur))) {
    select.value = cur;
  } else if (options.length) {
    select.value = options[0];
    setDeep(state.cfg, key, parseValueLike(options[0]));
  }

  select.onchange = (e) => {
    const v = parseValueLike(e.target.value);
    setDeep(state.cfg, key, v);
    onConfigChange(key);
  };
}

function buildCheckboxRow(row) {
  const key = row.dataset.key;
  let cb = row.querySelector("input[type=checkbox]");
  if (!cb) {
    cb = document.createElement("input");
    cb.type = "checkbox";
    row.appendChild(cb);
  }
  let cur = getDeep(state.cfg, key);
  if (key === "model.moe.enabled") {
    cur = !!(state.cfg.model.moe && state.cfg.model.moe.enabled !== false);
  }
  cb.checked = !!cur;
  cb.onchange = () => {
    if (key === "model.moe.enabled") {
      if (cb.checked) {
        // Restore from baseCfg if there was MoE
        if (state.baseCfg.model.moe) {
          state.cfg.model.moe = JSON.parse(JSON.stringify(state.baseCfg.model.moe));
        } else {
          state.cfg.model.moe = {
            n_experts: 8, experts_per_token: 2,
            capacity_factor: 1, expert_inter_sz: state.cfg.model.inter_sz,
            moe_frequency: 1, expert_tp_degree: 1,
          };
        }
        state.cfg.model.moe.enabled = true;
      } else {
        state.cfg.model.moe = null;
      }
      buildAllControls();
      onConfigChange(key);
    } else {
      setDeep(state.cfg, key, cb.checked);
      onConfigChange(key);
    }
  };
}

function parseValueLike(v) {
  if (v === "true") return true;
  if (v === "false") return false;
  const n = Number(v);
  if (!isNaN(n) && v !== "" && !/^[a-zA-Z_]/.test(v)) return n;
  return v;
}

function fillSelect(key, options) {
  // Used pre-config-load only as a placeholder; real fill is in buildSelectRow.
}

function parallelismOptionsFor(key) {
  const numDevices = (state.cfg.search && state.cfg.search.num_devices) || 1;
  const nLayers = state.cfg.model.n_layers || 1;
  const nExperts = (state.cfg.model.moe && state.cfg.model.moe.n_experts) || 1;
  const tp = state.cfg.parallelism.tp || 1;
  const pp = state.cfg.parallelism.pp || 1;
  const cp = state.cfg.parallelism.cp || 1;
  const remaining = Math.floor(numDevices / Math.max(1, tp * pp * cp));

  if (key === "parallelism.tp") return divisorsOf(numDevices);
  if (key === "parallelism.pp")
    return divisorsOf(numDevices).filter((p) => nLayers % p === 0);
  if (key === "parallelism.cp") return divisorsOf(numDevices);
  if (key === "parallelism.ep") {
    const cap = Math.min(nExperts, Math.max(1, remaining * tp));
    return divisorsOf(nExperts).filter((e) => e <= cap);
  }
  if (key === "parallelism.etp") return divisorsOf(tp);
  return [];
}

function updateMoeVisibility() {
  const moeFields = document.querySelector(".moe-fields");
  if (!moeFields) return;
  const enabled = !!(state.cfg.model.moe && state.cfg.model.moe.enabled !== false);
  moeFields.classList.toggle("disabled", !enabled);
}

// Called after any user change. Recomputes derived fields and triggers eval.
function onConfigChange(key) {
  // adjust dp based on tp,pp,cp,num_devices
  recomputeDp();

  // when num_devices or n_layers changes, parallelism options may shift
  if (key === "search.num_devices" || key === "model.n_layers" ||
      key === "model.moe.n_experts" || key === "model.moe.enabled" ||
      key === "parallelism.tp" || key === "parallelism.pp" || key === "parallelism.cp") {
    refreshParallelismSelects();
  }

  if (key === "hardware.node_type") refreshHardwareInfo();
  if (key === "model.moe.enabled") updateMoeVisibility();

  // par-product display
  const pc = state.cfg.parallelism;
  const product = (pc.tp || 1) * (pc.pp || 1) * (pc.cp || 1) * (pc.dp || 1);
  $("#par-product").textContent =
    `tp(${pc.tp}) · pp(${pc.pp}) · cp(${pc.cp}) · dp(${pc.dp}) = ${product}`;

  scheduleEvaluate();
}

function recomputeDp() {
  const numDevices = (state.cfg.search && state.cfg.search.num_devices) || 1;
  const tp = state.cfg.parallelism.tp || 1;
  const pp = state.cfg.parallelism.pp || 1;
  const cp = state.cfg.parallelism.cp || 1;
  const denom = tp * pp * cp;
  let dp = denom > 0 ? Math.floor(numDevices / denom) : 1;
  if (dp < 1) dp = 1;
  state.cfg.parallelism.dp = dp;
  $("#dp-derived").textContent = `${dp}  (= ${numDevices} / (${tp}·${pp}·${cp}))`;
}

function refreshParallelismSelects() {
  ["parallelism.tp", "parallelism.pp", "parallelism.cp",
   "parallelism.ep", "parallelism.etp"].forEach((key) => {
    const row = document.querySelector(`.select-row[data-key="${key}"]`);
    if (row) buildSelectRow(row);
  });
  recomputeDp();
}

function refreshHardwareInfo() {
  const hw = state.meta.hardware[state.cfg.hardware.node_type];
  if (!hw) {
    $("#hw-info").textContent = "";
    return;
  }
  const dtype = state.cfg.model.precision;
  const flops = hw.peak_flops[dtype];
  const flopsStr = flops ? `${(flops / 1e12).toFixed(0)} TFLOPs` : "n/a";
  $("#hw-info").textContent =
    `${hw.n_devices_per_node} devices/node · ` +
    `${hw.device_memory_gb.toFixed(0)} GB HBM · ` +
    `peak ${dtype}=${flopsStr} · ` +
    `intra ${hw.intra_node_bw_gbps.toFixed(0)} GB/s · ` +
    `inter ${hw.inter_node_bw_gbps.toFixed(0)} GB/s`;
}

function refreshAll() {
  recomputeDp();
  refreshParallelismSelects();
  refreshHardwareInfo();
}

// ---------------------------- Evaluation ----------------------------

let evalTimer = null;
function scheduleEvaluate() {
  // debounce: rapid slider movement is common
  if (evalTimer) clearTimeout(evalTimer);
  evalTimer = setTimeout(evaluate, 120);
}

function buildPayloadFromState() {
  const cfg = JSON.parse(JSON.stringify(state.cfg));
  if (cfg.model.moe && cfg.model.moe.enabled === false) cfg.model.moe = null;
  if (cfg.model.moe) delete cfg.model.moe.enabled;
  cfg.parallelism.dp = (cfg.search.num_devices) /
    (cfg.parallelism.tp * cfg.parallelism.pp * cfg.parallelism.cp);
  return {
    model: cfg.model,
    data: cfg.data,
    hardware: cfg.hardware,
    performance: cfg.performance,
    optimizer: cfg.optimizer || { optimizer_type: "adam" },
    parallelism: cfg.parallelism,
  };
}

async function evaluate() {
  const seq = ++state.evalSeq;
  setStatus("busy", "evaluating");
  try {
    const resp = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayloadFromState()),
    }).then((r) => r.json());
    if (seq !== state.evalSeq) return; // stale
    if (!resp.ok) {
      showError(resp.error);
      paintMetrics(null);
      setStatus("err", "invalid");
      return;
    }
    hideError();
    paintMetrics(resp);
    setStatus("ok", "ready");
    refreshSweepCharts();
  } catch (e) {
    showError(String(e));
    setStatus("err", "error");
  }
}

function paintMetrics(r) {
  if (!r) {
    ["m-mfu", "m-iter", "m-tput", "m-mem"].forEach((id) => $("#" + id).textContent = "—");
    $("#m-mem-card").classList.remove("warn", "danger");
    $("#mem-fill").style.width = "0%";
    $("#mem-fill").classList.remove("warn", "danger");
    return;
  }
  $("#m-mfu").textContent = r.mfu.toFixed(2) + "%";
  $("#m-iter").textContent = r.iteration_time_s.toFixed(3) + " s";
  $("#m-tput").textContent = fmtNumber(r.throughput_tokens_per_sec);
  $("#m-tput-sub").textContent =
    fmtNumber(r.tokens_per_day) + " tokens/day · " +
    "ideal/iter · gbs=" + state.cfg.data.gbs;
  $("#m-mem").textContent = r.memory_per_device_gb.toFixed(2) + " GB";
  $("#m-mem-sub").textContent =
    `${r.memory_per_device_gb.toFixed(1)} of ${r.device_memory_gb.toFixed(0)} GB ` +
    `(${(r.memory_fraction * 100).toFixed(0)}%)`;

  const frac = r.memory_fraction;
  const card = $("#m-mem-card");
  const fill = $("#mem-fill");
  card.classList.remove("warn", "danger");
  fill.classList.remove("warn", "danger");
  fill.style.width = Math.min(100, frac * 100).toFixed(1) + "%";
  if (frac > 1.0) {
    card.classList.add("danger"); fill.classList.add("danger");
  } else if (frac > 0.85) {
    card.classList.add("warn"); fill.classList.add("warn");
  }
}

// ---------------------------- Sweep charts ----------------------------

let sweepInflight = false;
let sweepPending = false;
async function refreshSweepCharts() {
  if (sweepInflight) { sweepPending = true; return; }
  sweepInflight = true;
  try {
    const numDevices = state.cfg.search.num_devices;
    const nLayers = state.cfg.model.n_layers;
    const tps = divisorsOf(numDevices).filter((tp) => tp <= 64);
    const pps = divisorsOf(numDevices).filter((pp) => nLayers % pp === 0 && pp <= 32);

    const tpResults = await Promise.all(
      tps.map((tp) => evaluateOverride({ tp, pp: 1, cp: 1 }))
    );
    const ppResults = await Promise.all(
      pps.map((pp) => evaluateOverride({ tp: state.cfg.parallelism.tp, pp, cp: 1 }))
    );

    drawChart("chart-tp", "tp", `tp sweep (pp=1, cp=1)`, tps, tpResults);
    drawChart("chart-pp", "pp", `pp sweep (tp=${state.cfg.parallelism.tp}, cp=1)`, pps, ppResults);

    // Fire heatmaps in the background — they don't gate the main charts.
    refreshHeatmaps();
  } catch (e) {
    // best effort — sweep failures shouldn't break the UI
    console.warn("sweep failed", e);
  } finally {
    sweepInflight = false;
    if (sweepPending) {
      sweepPending = false;
      refreshSweepCharts();
    }
  }
}

async function evaluateOverride(override) {
  const payload = buildPayloadFromState();
  const numDevices = state.cfg.search.num_devices;
  const tp = override.tp ?? payload.parallelism.tp;
  const pp = override.pp ?? payload.parallelism.pp;
  const cp = override.cp ?? payload.parallelism.cp;
  const dp = numDevices / (tp * pp * cp);
  if (!Number.isInteger(dp) || dp < 1) return null;
  payload.parallelism = { ...payload.parallelism, tp, pp, cp, dp };
  // adjust ep upper bound
  if (payload.model.moe) {
    const nExp = payload.model.moe.n_experts;
    if (payload.parallelism.ep > nExp) payload.parallelism.ep = nExp;
  }
  // etp must divide tp
  if (payload.parallelism.etp > tp) payload.parallelism.etp = tp;
  while (payload.parallelism.etp > 1 && tp % payload.parallelism.etp !== 0) {
    payload.parallelism.etp = Math.max(1, payload.parallelism.etp - 1);
  }
  try {
    const r = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then((r) => r.json());
    return r;
  } catch (_) {
    return null;
  }
}

function drawChart(canvasId, label, title, xs, results) {
  const ctx = document.getElementById(canvasId);
  const mfus = results.map((r) => (r && r.ok ? r.mfu : null));
  const mems = results.map((r) => (r && r.ok ? r.memory_per_device_gb : null));
  const memFrac = results.map((r) => (r && r.ok ? r.memory_fraction : 0));
  const memColors = memFrac.map((f) =>
    f > 1.0 ? "#f85149" : (f > 0.85 ? "#d29922" : "#3fb950")
  );

  const data = {
    labels: xs.map(String),
    datasets: [
      {
        label: "MFU %",
        data: mfus,
        borderColor: "#58a6ff",
        backgroundColor: "rgba(88,166,255,0.2)",
        yAxisID: "y",
        tension: 0.2,
        pointRadius: 4,
      },
      {
        label: "Memory GB",
        data: mems,
        type: "bar",
        backgroundColor: memColors,
        yAxisID: "y1",
      },
    ],
  };
  const opts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: { display: true, text: title, color: "#e6edf3" },
      legend: { labels: { color: "#e6edf3" } },
    },
    scales: {
      x: { title: { display: true, text: label, color: "#8b949e" },
            ticks: { color: "#8b949e" }, grid: { color: "#30363d" } },
      y: { type: "linear", position: "left",
            title: { display: true, text: "MFU %", color: "#58a6ff" },
            ticks: { color: "#8b949e" }, grid: { color: "#30363d" } },
      y1: { type: "linear", position: "right",
            title: { display: true, text: "Memory GB", color: "#d29922" },
            ticks: { color: "#8b949e" }, grid: { drawOnChartArea: false } },
    },
  };
  const key = canvasId.endsWith("tp") ? "tp" : "pp";
  if (state.charts[key]) {
    state.charts[key].data = data;
    state.charts[key].options = opts;
    state.charts[key].update("none");
  } else {
    state.charts[key] = new Chart(ctx, { type: "line", data, options: opts });
  }
}

// ---------------------------- Heatmaps ----------------------------

let heatmapInflight = false;
let heatmapPending = false;
let heatmapSeq = 0;

async function refreshHeatmaps() {
  if (heatmapInflight) { heatmapPending = true; return; }
  heatmapInflight = true;
  const seq = ++heatmapSeq;
  try {
    const numDevices = state.cfg.search.num_devices;
    const nLayers = state.cfg.model.n_layers;
    const moe = state.cfg.model.moe;
    const moeOn = !!(moe && moe.enabled !== false);
    const nExperts = moeOn ? (moe.n_experts || 1) : 1;

    const tps = divisorsOf(numDevices).filter((tp) => tp <= 32);
    const pps = divisorsOf(numDevices).filter((pp) => nLayers % pp === 0 && pp <= 16);
    const eps = moeOn
      ? divisorsOf(nExperts).filter((e) => e <= Math.min(nExperts, 32))
      : [1];

    // tp × ep (pp=1, cp=1)
    const tpEpMatrix = await computeMatrix(tps, eps, (tp, ep) => ({
      override: { tp, pp: 1, cp: 1, ep },
    }));
    if (seq !== heatmapSeq) return;
    renderHeatmap("heatmap-tp-ep", "tp", "ep", tps, eps, tpEpMatrix);

    // pp × ep (cp=1, tp = current)
    const tpFixed = state.cfg.parallelism.tp;
    const ppEpMatrix = await computeMatrix(pps, eps, (pp, ep) => ({
      override: { tp: tpFixed, pp, cp: 1, ep },
    }));
    if (seq !== heatmapSeq) return;
    renderHeatmap("heatmap-pp-ep", "pp", `ep (tp=${tpFixed})`, pps, eps, ppEpMatrix);
  } catch (e) {
    console.warn("heatmap failed", e);
  } finally {
    heatmapInflight = false;
    if (heatmapPending) {
      heatmapPending = false;
      refreshHeatmaps();
    }
  }
}

async function computeMatrix(rows, cols, buildOverride) {
  // Returns rows × cols of {ok, mfu, memory_per_device_gb, memory_fraction}
  const tasks = [];
  for (const r of rows) {
    for (const c of cols) {
      tasks.push({ r, c, payload: buildOverride(r, c).override });
    }
  }
  const results = await Promise.all(tasks.map((t) => evaluateOverride(t.payload)));
  const matrix = {};
  results.forEach((res, i) => {
    const { r, c } = tasks[i];
    if (!matrix[r]) matrix[r] = {};
    matrix[r][c] = res;
  });
  return matrix;
}

function renderHeatmap(elId, rowLabel, colLabel, rowVals, colVals, matrix) {
  const el = document.getElementById(elId);
  if (!rowVals.length || !colVals.length) {
    el.innerHTML = '<div class="hm-empty">no valid grid</div>';
    return;
  }

  // Find max valid MFU for color scaling, plus best cell coords
  let maxMfu = 0;
  let bestKey = null;
  for (const r of rowVals) {
    for (const c of colVals) {
      const cell = matrix[r] && matrix[r][c];
      if (cell && cell.ok && cell.memory_fraction <= 1.0 && cell.mfu > maxMfu) {
        maxMfu = cell.mfu;
        bestKey = `${r}_${c}`;
      }
    }
  }

  const grid = document.createElement("div");
  grid.className = "heatmap-grid";
  grid.style.gridTemplateColumns = `auto repeat(${colVals.length}, minmax(36px, 1fr))`;

  // Header row: corner + col labels
  const corner = document.createElement("div");
  corner.className = "hm-corner";
  corner.textContent = `${rowLabel} \\ ${colLabel}`;
  grid.appendChild(corner);
  colVals.forEach((c) => {
    const h = document.createElement("div");
    h.className = "hm-axis-x";
    h.textContent = c;
    grid.appendChild(h);
  });

  // Each row
  rowVals.forEach((r) => {
    const ylab = document.createElement("div");
    ylab.className = "hm-axis-y";
    ylab.textContent = r;
    grid.appendChild(ylab);

    colVals.forEach((c) => {
      const cellEl = document.createElement("div");
      cellEl.className = "hm-cell";
      const cell = matrix[r] && matrix[r][c];
      if (!cell || !cell.ok) {
        cellEl.classList.add("invalid");
        cellEl.textContent = "—";
        cellEl.title = cell && cell.error ? cell.error : "invalid";
      } else if (cell.memory_fraction > 1.0) {
        cellEl.classList.add("oom");
        cellEl.textContent = "OOM";
        cellEl.title =
          `${rowLabel}=${r}, ${colLabel}=${c}\n` +
          `MFU: ${cell.mfu.toFixed(2)}%\n` +
          `Mem: ${cell.memory_per_device_gb.toFixed(1)} GB ` +
          `(${(cell.memory_fraction * 100).toFixed(0)}%) — over budget`;
      } else {
        const v = cell.mfu / Math.max(maxMfu, 1e-9);
        cellEl.style.background = mfuColor(v);
        cellEl.textContent = cell.mfu.toFixed(1);
        cellEl.title =
          `${rowLabel}=${r}, ${colLabel}=${c}\n` +
          `MFU: ${cell.mfu.toFixed(2)}%\n` +
          `Mem: ${cell.memory_per_device_gb.toFixed(1)} GB ` +
          `(${(cell.memory_fraction * 100).toFixed(0)}%)\n` +
          `Iter: ${cell.iteration_time_s.toFixed(3)} s`;
        if (`${r}_${c}` === bestKey) cellEl.classList.add("best");
        cellEl.style.cursor = "pointer";
        cellEl.addEventListener("click", () => applyHeatmapCell(rowLabel, colLabel, r, c));
      }
      grid.appendChild(cellEl);
    });
  });

  el.innerHTML = "";
  el.appendChild(grid);

  const legend = document.createElement("div");
  legend.className = "hm-legend";
  legend.innerHTML =
    `<span>0%</span><div class="hm-legend-bar"></div>` +
    `<span>${maxMfu > 0 ? maxMfu.toFixed(1) + "% (best)" : "—"}</span>`;
  el.appendChild(legend);
}

function mfuColor(v) {
  // v in [0,1] -> green ramp from dark to bright
  v = Math.max(0, Math.min(1, v));
  // Interpolate from #0d3a1e -> #3fb950 -> #7ee896
  const stops = [
    [13, 58, 30],
    [26, 107, 58],
    [63, 185, 80],
    [126, 232, 150],
  ];
  const seg = v * (stops.length - 1);
  const i = Math.floor(seg);
  const t = seg - i;
  const a = stops[i];
  const b = stops[Math.min(i + 1, stops.length - 1)];
  const r = Math.round(a[0] + (b[0] - a[0]) * t);
  const g = Math.round(a[1] + (b[1] - a[1]) * t);
  const bl = Math.round(a[2] + (b[2] - a[2]) * t);
  return `rgb(${r},${g},${bl})`;
}

function applyHeatmapCell(rowLabel, colLabel, rowVal, colVal) {
  // Apply selected (tp/pp, ep) onto the live config.
  if (rowLabel === "tp") {
    state.cfg.parallelism.tp = rowVal;
    state.cfg.parallelism.pp = 1;
    state.cfg.parallelism.cp = 1;
  } else if (rowLabel === "pp") {
    state.cfg.parallelism.pp = rowVal;
    state.cfg.parallelism.cp = 1;
  }
  state.cfg.parallelism.ep = colVal;
  if (state.cfg.parallelism.etp > state.cfg.parallelism.tp) {
    state.cfg.parallelism.etp = state.cfg.parallelism.tp;
  }
  buildAllControls();
  refreshAll();
  scheduleEvaluate();
}

// ---------------------------- Search ----------------------------

async function runSearch() {
  const btn = $("#search-btn");
  btn.disabled = true;
  $("#search-status").textContent = "running…";
  const tbody = document.querySelector("#search-table tbody");
  tbody.innerHTML = "";

  const cfg = JSON.parse(JSON.stringify(state.cfg));
  if (cfg.model.moe && cfg.model.moe.enabled === false) cfg.model.moe = null;
  if (cfg.model.moe) delete cfg.model.moe.enabled;
  delete cfg.parallelism;

  const body = {
    cfg,
    num_devices: state.cfg.search.num_devices,
    top_k: parseInt($("#search-top-k").value, 10) || 10,
    max_ep: parseInt($("#search-max-ep").value, 10) || 32,
    max_tp: $("#search-max-tp").value ? parseInt($("#search-max-tp").value, 10) : null,
    max_cp: $("#search-max-cp").value ? parseInt($("#search-max-cp").value, 10) : null,
    max_mbs: parseInt($("#search-max-mbs").value, 10) || 1,
    activation_checkpointing: $("#search-act-ckpt").value,
    memory_limit_fraction: parseFloat($("#search-mem-frac").value) || 0.9,
  };

  try {
    const resp = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json());
    if (!resp.ok) {
      $("#search-status").textContent = "error: " + resp.error;
      return;
    }
    $("#search-status").textContent =
      `${resp.results.length} top configs · ${resp.valid_count} valid / ${resp.total_evaluated} total`;
    resp.results.forEach((row, i) => {
      const tr = document.createElement("tr");
      if (i === 0) tr.classList.add("best");
      const c = row.config;
      const memCls = row.memory_fraction > 1.0 ? "danger" :
                     row.memory_fraction > 0.85 ? "warn" : "";
      tr.innerHTML = `
        <td>${i + 1}</td>
        <td>${c.tp}</td><td>${c.pp}</td><td>${c.cp}</td><td>${c.dp}</td>
        <td>${c.ep}</td><td>${c.etp ?? c.tp}</td>
        <td>${c.microbatch_sz ?? state.cfg.data.microbatch_sz}</td>
        <td>${c.activation_checkpointing ?? "—"}</td>
        <td><b>${row.mfu.toFixed(2)}</b></td>
        <td class="mem-cell ${memCls}">${row.memory_per_device_gb.toFixed(2)}</td>
        <td class="mem-cell ${memCls}">${(row.memory_fraction * 100).toFixed(0)}%</td>
        <td>${row.iteration_time_s.toFixed(3)}</td>
        <td>${fmtNumber(row.throughput_tokens_per_sec)}</td>
        <td><button class="apply-btn">Apply</button></td>
      `;
      tr.querySelector(".apply-btn").addEventListener("click", () => applySearchResult(c));
      tbody.appendChild(tr);
    });
  } catch (e) {
    $("#search-status").textContent = "error: " + e;
  } finally {
    btn.disabled = false;
  }
}

function applySearchResult(c) {
  state.cfg.parallelism.tp = c.tp;
  state.cfg.parallelism.pp = c.pp;
  state.cfg.parallelism.cp = c.cp;
  state.cfg.parallelism.dp = c.dp;
  state.cfg.parallelism.ep = c.ep;
  state.cfg.parallelism.etp = c.etp ?? c.tp;
  if ("microbatch_sz" in c) state.cfg.data.microbatch_sz = c.microbatch_sz;
  if ("activation_checkpointing" in c) {
    state.cfg.performance.activation_checkpointing_type = c.activation_checkpointing;
  }
  buildAllControls();
  refreshAll();
  scheduleEvaluate();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ---------------------------- Status / errors ----------------------------

function setStatus(kind, text) {
  const pill = $("#status-pill");
  pill.className = "pill " + kind;
  pill.textContent = text;
}
function showError(msg) {
  const box = $("#error-box");
  box.classList.remove("hidden");
  box.textContent = msg;
}
function hideError() {
  $("#error-box").classList.add("hidden");
}

document.addEventListener("DOMContentLoaded", init);
