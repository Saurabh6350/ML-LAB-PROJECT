/* ML Fraud Detection Project — main.js */

const API = "";  // same origin

// ── Static fallback presets ───────────────────────────────────────
const PRESETS = {
  legit: {
    Time:149362, Amount:9.99,
    V1:1.19,V2:0.27,V3:0.41,V4:0.32,V5:-0.08,V6:-0.01,
    V7:0.09,V8:0.07,V9:-0.24,V10:-0.17,V11:-0.08,V12:-0.14,
    V13:0.04,V14:0.28,V15:0.05,V16:-0.04,V17:-0.02,V18:-0.01,
    V19:0.01,V20:-0.03,V21:-0.02,V22:0.04,V23:-0.01,V24:0.00,
    V25:0.08,V26:0.04,V27:0.01,V28:0.01
  },
  fraud: {
    Time:43200, Amount:2125.87,
    V1:-3.04,V2:2.83,V3:-4.26,V4:3.14,V5:-2.08,V6:-0.63,
    V7:-3.35,V8:1.73,V9:-0.86,V10:-4.32,V11:3.56,V12:-4.81,
    V13:0.34,V14:-4.62,V15:-0.22,V16:-2.71,V17:-3.15,V18:-1.52,
    V19:0.72,V20:-0.27,V21:0.28,V22:-0.63,V23:0.07,V24:-0.12,
    V25:0.23,V26:-0.14,V27:0.34,V28:0.11
  },
  uncertain: {
    Time:86400, Amount:312.44,
    V1:-0.52,V2:0.38,V3:-0.81,V4:0.61,V5:-0.44,V6:0.21,
    V7:-0.36,V8:0.14,V9:-0.22,V10:-0.72,V11:0.48,V12:-0.59,
    V13:0.09,V14:-0.71,V15:0.02,V16:-0.41,V17:-0.38,V18:-0.19,
    V19:0.11,V20:-0.07,V21:0.06,V22:-0.09,V23:0.02,V24:-0.04,
    V25:0.08,V26:-0.03,V27:0.07,V28:0.02
  },
};

// ── Demo cases from backend ───────────────────────────────────────
let demoCases = [];

async function loadDemoCases() {
  try {
    const res  = await fetch(`${API}/api/demo_cases`);
    const data = await res.json();
    demoCases  = data.cases || [];
    renderDemoCaseButtons();
  } catch(e) {
    const row = document.getElementById("demoCaseRow");
    if (row) row.innerHTML = '<span class="preset-label">DEMO:</span><span class="dim-text">offline</span>';
  }
}

function renderDemoCaseButtons() {
  const row = document.getElementById("demoCaseRow");
  if (!row || !demoCases.length) return;
  const icons = {legitimate:"✅", suspicious:"⚠️", fraud:"🚨", edge:"🔍"};
  row.innerHTML = '<span class="preset-label">DEMO:</span>' +
    demoCases.map((c,i) =>
      `<button class="preset-btn preset-${c.category}" onclick="loadDemoCase(${i})"
               title="${c.description}">${icons[c.category]||"•"} ${c.label}</button>`
    ).join("");
}

function loadDemoCase(idx) {
  const c = demoCases[idx];
  if (!c) return;
  fillForm(c.data);
  log(`Demo loaded: ${c.label} (${c.category})`, "info");
}

function loadPreset(name) {
  if (name === "random") {
    const d = {};
    for (let i=1;i<=28;i++) d[`V${i}`] = +(Math.random()*6-3).toFixed(4);
    d.Time   = Math.floor(Math.random()*172792);
    d.Amount = +(Math.random()*3000).toFixed(2);
    fillForm(d);
  } else {
    fillForm(PRESETS[name]);
  }
  log(`Preset loaded: ${name}`, "info");
}

function fillForm(d) {
  document.getElementById("f-time").value   = d.Time   ?? 0;
  document.getElementById("f-amount").value = d.Amount ?? 0;
  for (let i=1;i<=28;i++) {
    const el = document.getElementById(`v${i}`);
    if (el) el.value = d[`V${i}`] ?? 0;
  }
}

function buildVGrid() {
  const g = document.getElementById("vGrid");
  for (let i=1;i<=28;i++) {
    g.innerHTML += `<div class="v-field">
      <span class="v-label">V${i}</span>
      <input class="v-input" id="v${i}" type="number" value="0" step="0.0001">
    </div>`;
  }
}

function readForm() {
  const d = {
    Time:   parseFloat(document.getElementById("f-time").value)   || 0,
    Amount: parseFloat(document.getElementById("f-amount").value) || 0,
  };
  for (let i=1;i<=28;i++) {
    const el = document.getElementById(`v${i}`);
    d[`V${i}`] = el ? (parseFloat(el.value) || 0) : 0;
  }
  return d;
}

// ── Prediction ────────────────────────────────────────────────────
async function runPredict() {
  const data = readForm();
  const btn  = document.querySelector(".analyse-btn");
  btn.innerHTML = "⏳  ANALYSING...";
  btn.disabled = true;

  try {
    const res  = await fetch(`${API}/api/predict`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(data)
    });
    const json = await res.json();
    if (json.error) {
      alert("Backend error: " + json.error + (json.trace ? "\n\n" + json.trace : ""));
      return;
    }
    renderResult(json);
    log(
      `${json.prediction} | ${json.probability_pct}% | ${json.risk_level} | $${data.Amount.toFixed(2)}`,
      json.prediction === "FRAUD" ? "fraud" : "legit"
    );
  } catch(e) {
    alert("Backend unreachable. Is app.py running?\n" + e.message);
    log("Backend error: " + e.message, "warn");
  } finally {
    btn.innerHTML = '<span class="btn-icon">⚡</span> ANALYSE TRANSACTION';
    btn.disabled = false;
  }
}

// ── Render result ─────────────────────────────────────────────────
function renderResult(r) {
  document.getElementById("resultPanel").style.display = "block";

  // Verdict
  const vEl = document.getElementById("resultVerdict");
  vEl.textContent = r.prediction;
  vEl.className   = "result-verdict verdict-" + r.prediction.toLowerCase();

  // Badges
  document.getElementById("resultBadges").innerHTML =
    `<span class="conf-badge conf-${r.confidence}">${r.confidence}</span>` +
    `<span class="risk-badge risk-${r.risk_level}">${r.risk_level}</span>`;

  // ── Probability bar ──────────────────────────────────────────────
  // r.probability      = 0.0 – 1.0  (raw)
  // r.probability_pct  = 0.0 – 100.0 (already multiplied)
  // Use probability_pct directly as CSS width % — DO NOT multiply again
  const pct   = r.probability_pct;     // e.g. 73.21
  const prob  = r.probability;         // e.g. 0.7321

  const color = prob >= 0.60 ? "#E8455A"
               : prob >= 0.35 ? "#F5A623"
               : "#2ECC71";

  document.getElementById("probValue").textContent = pct.toFixed(1) + "%";
  document.getElementById("probValue").style.color = color;

  const fill = document.getElementById("probBarFill");
  // Use sqrt scaling so intermediate probabilities show a meaningful bar
  // e.g. 1% raw → ~10% bar, 25% raw → ~50% bar, 100% raw → 100% bar
  const visualPct = prob <= 0 ? 0.5
                  : prob >= 1 ? 100
                  : Math.max(0.5, Math.sqrt(prob) * 100);
  fill.style.width      = visualPct.toFixed(2) + "%";
  fill.style.background = color;

  // Anomaly scores
  document.getElementById("isoValue").textContent  = r.iso_score;
  document.getElementById("txHour").textContent    =
    r.transaction_hour !== undefined ? r.transaction_hour + "h" : "—";
  document.getElementById("txAmount").textContent  =
    "$" + (r.amount !== undefined ? parseFloat(r.amount).toFixed(2) : "—");

  // Risk factors
  const rf = r.risk_factors || {};
  document.getElementById("riskFactors").innerHTML = [
    {label:"High amount  (>$500)",   flag: rf.high_amount},
    {label:"Off-hours  (10pm–6am)",  flag: rf.off_hours},
    {label:"ISO anomaly detected",   flag: rf.iso_anomalous},
  ].map(f => `
    <div class="rf-item">
      <div class="rf-dot ${f.flag ? 'warn' : 'ok'}"></div>
      <span class="rf-text ${f.flag ? 'flagged' : 'safe'}">${f.label}</span>
    </div>`).join("");

  // SHAP
  document.getElementById("shapList").innerHTML =
    (r.explanation || []).map(e => {
      const cls = e.includes("↑") ? "shap-up" : "shap-down";
      return `<div class="shap-item ${cls}">${e}</div>`;
    }).join("") || '<div class="dim-text" style="font-size:9px">Not available</div>';

  // Review banner
  document.getElementById("reviewBanner").style.display =
    r.needs_review ? "block" : "none";

  document.getElementById("resultPanel").scrollIntoView({behavior:"smooth", block:"nearest"});
}

// ── Tabs ──────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  });
});

// ── Metrics ───────────────────────────────────────────────────────
async function loadMetrics() {
  try {
    const res  = await fetch(`${API}/api/metrics`);
    const data = await res.json();
    const m    = data.metrics;

    document.getElementById("hd-auc").textContent = "AUC-ROC: " + m.roc_auc;
    document.getElementById("hd-f1").textContent  = "F1: "      + m.f1;

    document.getElementById("m-roc").textContent  = m.roc_auc;
    document.getElementById("m-pr").textContent   = m.avg_prec;
    document.getElementById("m-f1").textContent   = m.f1;
    document.getElementById("m-rec").textContent  = m.recall;
    document.getElementById("m-prec").textContent = m.precision;
    if (m.mcc !== undefined) {
      const mccEl = document.getElementById("m-mcc");
      if (mccEl) mccEl.textContent = m.mcc;
    }
    const thrEl = document.getElementById("m-thr");
    if (thrEl) thrEl.textContent = (data.threshold || m.threshold || "0.5");

    // Feature importance
    const fi  = data.feature_importance || [];
    const max = fi[0]?.importance || 1;
    document.getElementById("featureList").innerHTML = fi.slice(0, 12).map(f => `
      <div class="feat-item">
        <span class="feat-name">${f.feature}</span>
        <div class="feat-bar-wrap">
          <div class="feat-bar-bg">
            <div class="feat-bar-fill" style="width:${(f.importance/max*100).toFixed(1)}%"></div>
          </div>
          <span class="feat-pct">${(f.importance/max*100).toFixed(0)}%</span>
        </div>
      </div>`).join("");

    log("Model metrics loaded", "info");
  } catch(e) {
    log("Backend offline — start app.py", "warn");
    loadFallbackMetrics();
  }
}

function loadFallbackMetrics() {
  // Use known copy2 values as fallback
  const m = {roc_auc:"0.9792",avg_prec:"0.8552",f1:"0.6241",recall:"0.8980",precision:"0.4783"};
  document.getElementById("m-roc").textContent  = m.roc_auc;
  document.getElementById("m-pr").textContent   = m.avg_prec;
  document.getElementById("m-f1").textContent   = m.f1;
  document.getElementById("m-rec").textContent  = m.recall;
  document.getElementById("m-prec").textContent = m.precision;
  document.getElementById("hd-auc").textContent = "AUC-ROC: " + m.roc_auc;
  document.getElementById("hd-f1").textContent  = "F1: " + m.f1;
  const mccEl2 = document.getElementById("m-mcc");
  if (mccEl2) mccEl2.textContent = "—";
  const thrEl2 = document.getElementById("m-thr");
  if (thrEl2) thrEl2.textContent = "0.82";
  const fi   = ["V4","V14","iso_anomaly_score","time_bin","v14_v4_interact",
                 "V10","V8","V12","V3","Time","V18","v11_v12_interact"];
  const vals = [1.24,1.17,0.63,0.57,0.40,0.32,0.31,0.28,0.26,0.25,0.24,0.24];
  const max  = vals[0];
  document.getElementById("featureList").innerHTML = fi.map((f,i) => `
    <div class="feat-item">
      <span class="feat-name">${f}</span>
      <div class="feat-bar-wrap">
        <div class="feat-bar-bg">
          <div class="feat-bar-fill" style="width:${(vals[i]/max*100).toFixed(0)}%"></div>
        </div>
        <span class="feat-pct">${(vals[i]/max*100).toFixed(0)}%</span>
      </div>
    </div>`).join("");
}

// ── Batch ─────────────────────────────────────────────────────────
function handleDrop(e) {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) processBatch(file);
}
function handleBatchFile(e) { processBatch(e.target.files[0]); }

async function processBatch(file) {
  const prog = document.getElementById("batchProgress");
  const bar  = document.getElementById("bpBar");
  const lbl  = document.getElementById("bpLabel");
  prog.style.display = "block";
  document.getElementById("batchResult").style.display = "none";
  lbl.textContent = "Uploading..."; bar.style.width = "20%";
  await sleep(200);
  lbl.textContent = "Running predictions..."; bar.style.width = "60%";
  const fd = new FormData();
  fd.append("file", file);
  try {
    const res  = await fetch(`${API}/api/predict_batch`, {method:"POST", body:fd});
    const data = await res.json();
    bar.style.width = "100%"; lbl.textContent = "Done!";
    await sleep(300);
    prog.style.display = "none";
    renderBatch(data);
    log(`Batch: ${data.total} rows | ${data.fraud_detected} fraud | ${data.fraud_rate_pct}%`, "warn");
  } catch(e) {
    prog.style.display = "none";
    alert("Backend unreachable: " + e.message);
  }
}

function renderBatch(data) {
  document.getElementById("batchResult").style.display = "block";
  document.getElementById("batchStats").innerHTML = [
    {l:"TOTAL",      v: data.total,               c:"var(--text)"},
    {l:"FRAUD",      v: data.fraud_detected,       c:"var(--red)"},
    {l:"LEGIT",      v: data.legitimate,           c:"var(--green)"},
    {l:"UNCERTAIN",  v: data.uncertain,            c:"var(--blue)"},
    {l:"FRAUD RATE", v: data.fraud_rate_pct + "%", c:"var(--amber)"},
  ].map(s => `<div class="bs-card">
    <span class="bs-label">${s.l}</span>
    <span class="bs-val" style="color:${s.c}">${s.v}</span>
  </div>`).join("");

  document.getElementById("batchTbody").innerHTML =
    (data.results || []).map((r, i) => {
      const pc = r.prediction === "FRAUD" ? "var(--red)" : "var(--green)";
      return `<tr>
        <td style="color:var(--dim2)">${i+1}</td>
        <td>$${r.amount.toFixed(2)}</td>
        <td style="color:${pc};font-weight:600">${r.probability_pct}%</td>
        <td><span class="risk-badge risk-${r.risk_level}" style="font-size:9px">${r.prediction}</span></td>
        <td><span class="risk-badge risk-${r.risk_level}" style="font-size:9px">${r.risk_level}</span></td>
        <td><span class="conf-badge conf-${r.confidence}" style="font-size:9px">${r.confidence}</span></td>
      </tr>`;
    }).join("");
}

// ── Log ───────────────────────────────────────────────────────────
function log(msg, type="info") {
  const feed = document.getElementById("logFeed");
  const now  = new Date().toLocaleTimeString("en-GB");
  const el   = document.createElement("div");
  el.className   = `log-item log-${type}`;
  el.textContent = `[${now}] ${msg}`;
  feed.prepend(el);
  while (feed.children.length > 60) feed.lastChild.remove();
}
function clearLog() {
  document.getElementById("logFeed").innerHTML =
    '<div class="log-item log-info">Log cleared</div>';
}
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── INIT ─────────────────────────────────────────────────────────
buildVGrid();
loadPreset("legit");
loadMetrics();
loadDemoCases();
