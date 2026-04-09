/* ML Fraud Detection Project — main.js */
const API = "";

// ── Presets ───────────────────────────────────────────────────────
const PRESETS = {
  legit: {Time:149362,Amount:9.99,V1:1.19,V2:0.27,V3:0.41,V4:0.32,V5:-0.08,V6:-0.01,V7:0.09,V8:0.07,V9:-0.24,V10:-0.17,V11:-0.08,V12:-0.14,V13:0.04,V14:0.28,V15:0.05,V16:-0.04,V17:-0.02,V18:-0.01,V19:0.01,V20:-0.03,V21:-0.02,V22:0.04,V23:-0.01,V24:0.00,V25:0.08,V26:0.04,V27:0.01,V28:0.01},
  fraud: {Time:43200,Amount:2125.87,V1:-3.04,V2:2.83,V3:-4.26,V4:3.14,V5:-2.08,V6:-0.63,V7:-3.35,V8:1.73,V9:-0.86,V10:-4.32,V11:3.56,V12:-4.81,V13:0.34,V14:-4.62,V15:-0.22,V16:-2.71,V17:-3.15,V18:-1.52,V19:0.72,V20:-0.27,V21:0.28,V22:-0.63,V23:0.07,V24:-0.12,V25:0.23,V26:-0.14,V27:0.34,V28:0.11},
  uncertain: {Time:86400,Amount:312.44,V1:-0.52,V2:0.38,V3:-0.81,V4:0.61,V5:-0.44,V6:0.21,V7:-0.36,V8:0.14,V9:-0.22,V10:-0.72,V11:0.48,V12:-0.59,V13:0.09,V14:-0.71,V15:0.02,V16:-0.41,V17:-0.38,V18:-0.19,V19:0.11,V20:-0.07,V21:0.06,V22:-0.09,V23:0.02,V24:-0.04,V25:0.08,V26:-0.03,V27:0.07,V28:0.02},
};

let demoCases = [];
let _metricsData = null;  // cache for plots

// ── Demo cases ────────────────────────────────────────────────────
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
      `<button class="preset-btn preset-${c.category}" onclick="loadDemoCase(${i})" title="${c.description}">${icons[c.category]||"•"} ${c.label}</button>`
    ).join("");
}

function loadDemoCase(idx) {
  const c = demoCases[idx];
  if (!c) return;
  fillForm(c.data);
  log(`Demo: ${c.label} (${c.category})`, "info");
}

function loadPreset(name) {
  if (name === "random") {
    const d = {};
    for (let i=1;i<=28;i++) d[`V${i}`] = +(Math.random()*6-3).toFixed(4);
    d.Time   = Math.floor(Math.random()*172792);
    d.Amount = +(Math.random()*3000).toFixed(2);
    fillForm(d);
  } else { fillForm(PRESETS[name]); }
  log(`Preset: ${name}`, "info");
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
    g.innerHTML += `<div class="v-field"><span class="v-label">V${i}</span><input class="v-input" id="v${i}" type="number" value="0" step="0.0001"></div>`;
  }
}

function readForm() {
  const d = {Time:parseFloat(document.getElementById("f-time").value)||0,Amount:parseFloat(document.getElementById("f-amount").value)||0};
  for (let i=1;i<=28;i++) { const el=document.getElementById(`v${i}`); d[`V${i}`]=el?(parseFloat(el.value)||0):0; }
  return d;
}

// ── Predict ───────────────────────────────────────────────────────
async function runPredict() {
  const data = readForm();
  const btn  = document.querySelector(".analyse-btn");
  btn.innerHTML = "⏳  ANALYSING..."; btn.disabled = true;
  try {
    const res  = await fetch(`${API}/api/predict`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
    const json = await res.json();
    if (json.error) { alert("Backend error:\n" + json.error + (json.trace?"\n\n"+json.trace:"")); return; }
    renderResult(json);
    log(`${json.prediction} | ${json.probability_pct}% | ${json.risk_level} | $${data.Amount.toFixed(2)}`, json.prediction==="FRAUD"?"fraud":"legit");
  } catch(e) {
    alert("Backend unreachable. Is app.py running?\n" + e.message);
    log("Error: " + e.message, "warn");
  } finally {
    btn.innerHTML = '<span class="btn-icon">⚡</span> ANALYSE TRANSACTION'; btn.disabled=false;
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
    `<span class="conf-badge conf-${r.confidence}">${r.confidence}</span> ` +
    `<span class="risk-badge risk-${r.risk_level}">${r.risk_level}</span>`;

  // Probability bar
  // r.probability_pct is already 0–100 from the calibrated prob×100
  const pct   = parseFloat(r.probability_pct);   // e.g. 73.5
  const prob  = parseFloat(r.probability);        // e.g. 0.735  (calibrated)
  const color = prob >= 0.80 ? "#E8455A"
               : prob >= 0.55 ? "#F59E0B"
               : prob >= 0.30 ? "#F5A623"
               : "#2ECC71";

  document.getElementById("probValue").textContent = pct.toFixed(1) + "%";
  document.getElementById("probValue").style.color = color;
  const fill = document.getElementById("probBarFill");
  fill.style.width      = pct + "%";   // pct is 0–100, correct for CSS width
  fill.style.background = color;

  // Raw probability footnote
  if (r.raw_probability !== undefined) {
    document.getElementById("probRaw").textContent =
      `Raw model output: ${(r.raw_probability*100).toFixed(4)}%  |  Threshold: ${(r.threshold_used||0.5)*100}%`;
  }

  // Scores
  document.getElementById("isoValue").textContent  = r.iso_score;
  document.getElementById("txHour").textContent    = r.transaction_hour !== undefined ? r.transaction_hour + "h" : "—";
  document.getElementById("txAmount").textContent  = "$" + (r.amount !== undefined ? parseFloat(r.amount).toFixed(2) : "—");

  // Risk factors
  const rf = r.risk_factors || {};
  document.getElementById("riskFactors").innerHTML = [
    {label:"High amount (>$500)",  flag:rf.high_amount},
    {label:"Off-hours (10pm–6am)", flag:rf.off_hours},
    {label:"ISO anomaly detected", flag:rf.iso_anomalous},
  ].map(f=>`<div class="rf-item"><div class="rf-dot ${f.flag?'warn':'ok'}"></div><span class="rf-text ${f.flag?'flagged':'safe'}">${f.label}</span></div>`).join("");

  // SHAP
  document.getElementById("shapList").innerHTML =
    (r.explanation||[]).map(e=>{const cls=e.includes("↑")?"shap-up":"shap-down";return`<div class="shap-item ${cls}">${e}</div>`;}).join("") ||
    '<div class="dim-text" style="font-size:9px">Not available</div>';

  document.getElementById("reviewBanner").style.display = r.needs_review ? "block" : "none";
  document.getElementById("resultPanel").scrollIntoView({behavior:"smooth",block:"nearest"});
}

// ── Tabs ──────────────────────────────────────────────────────────
let plotsDrawn = false;
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b=>b.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(p=>p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-"+btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "plots" && !plotsDrawn && _metricsData) {
      drawAllPlots(_metricsData); plotsDrawn = true;
    }
  });
});

// ── Metrics ───────────────────────────────────────────────────────
async function loadMetrics() {
  try {
    const res  = await fetch(`${API}/api/metrics`);
    const m    = await res.json();   // flat structure from new /api/metrics
    _metricsData = m;

    // Header
    document.getElementById("hd-auc").textContent = "AUC-ROC: " + m.roc_auc;
    document.getElementById("hd-f1").textContent  = "F1: " + m.f1;

    // Sidebar metrics
    document.getElementById("m-roc").textContent  = m.roc_auc;
    document.getElementById("m-pr").textContent   = m.avg_prec;
    document.getElementById("m-f1").textContent   = m.f1;
    document.getElementById("m-rec").textContent  = m.recall;
    document.getElementById("m-prec").textContent = m.precision;
    document.getElementById("m-mcc").textContent  = m.mcc  !== undefined ? m.mcc  : "—";
    document.getElementById("m-thr").textContent  = m.threshold !== undefined ? m.threshold : "—";

    // Feature importance list
    const fi  = m.feature_importance || [];
    const max = fi[0]?.importance || 1;
    document.getElementById("featureList").innerHTML = fi.slice(0,12).map(f=>`
      <div class="feat-item">
        <span class="feat-name">${f.feature}</span>
        <div class="feat-bar-wrap">
          <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:${(f.importance/max*100).toFixed(1)}%"></div></div>
          <span class="feat-pct">${(f.importance/max*100).toFixed(0)}%</span>
        </div>
      </div>`).join("");

    log("Metrics loaded", "info");
  } catch(e) {
    log("Backend offline — start app.py", "warn");
    loadFallbackMetrics();
  }
}

function loadFallbackMetrics() {
  const m = {roc_auc:"0.9792",avg_prec:"0.8552",f1:"0.7468",recall:"0.6020",precision:"0.9833",mcc:"—",threshold:"0.82"};
  document.getElementById("hd-auc").textContent = "AUC-ROC: " + m.roc_auc;
  document.getElementById("hd-f1").textContent  = "F1: " + m.f1;
  Object.entries({roc:m.roc_auc,pr:m.avg_prec,f1:m.f1,rec:m.recall,prec:m.precision,mcc:m.mcc,thr:m.threshold})
    .forEach(([k,v])=>{ const el=document.getElementById("m-"+k); if(el) el.textContent=v; });
  // Fallback feature list
  const fi=["V4","V14","iso_anomaly_score","time_bin","v14_v4_interact","V10","V8","V12","V3","Time","V18","v11_v12_interact"];
  const vals=[1.24,1.17,0.63,0.57,0.40,0.32,0.31,0.28,0.26,0.25,0.24,0.24];
  const max=vals[0];
  document.getElementById("featureList").innerHTML=fi.map((f,i)=>`
    <div class="feat-item"><span class="feat-name">${f}</span>
      <div class="feat-bar-wrap">
        <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:${(vals[i]/max*100).toFixed(0)}%"></div></div>
        <span class="feat-pct">${(vals[i]/max*100).toFixed(0)}%</span>
      </div></div>`).join("");
}

// ── PLOTS ─────────────────────────────────────────────────────────
function drawAllPlots(m) {
  drawROC(m);
  drawPR(m);
  drawCM(m);
  drawClassReport(m);
  drawFI(m);
}

function getCtx(id) {
  const canvas = document.getElementById(id);
  if (!canvas) return null;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.offsetWidth  || 340;
  const H = canvas.offsetHeight || 260;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return {ctx, W, H};
}

function drawAxes(ctx, W, H, PAD, xlbl, ylbl) {
  const CW=W-PAD.l-PAD.r, CH=H-PAD.t-PAD.b;
  ctx.fillStyle="#fff"; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle="#e5e7eb"; ctx.lineWidth=1;
  for(let i=0;i<=5;i++){
    const x=PAD.l+i/5*CW, y=PAD.t+i/5*CH;
    ctx.beginPath(); ctx.moveTo(x,PAD.t); ctx.lineTo(x,PAD.t+CH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD.l,y); ctx.lineTo(PAD.l+CW,y); ctx.stroke();
    ctx.fillStyle="#9ca3af"; ctx.font="9px 'JetBrains Mono',monospace";
    ctx.textAlign="center"; ctx.fillText((i/5).toFixed(1), PAD.l+i/5*CW, PAD.t+CH+14);
    if(i>0){ctx.textAlign="right"; ctx.fillText((i/5).toFixed(1), PAD.l-5, PAD.t+(1-i/5)*CH+3);}
  }
  ctx.fillStyle="#374151"; ctx.font="bold 10px 'JetBrains Mono',monospace";
  ctx.textAlign="center"; ctx.fillText(xlbl, PAD.l+CW/2, H-2);
  ctx.save(); ctx.rotate(-Math.PI/2); ctx.fillText(ylbl, -(PAD.t+CH/2), 12); ctx.restore();
  return {CW,CH};
}

function drawROC(m) {
  const s = getCtx("rocCanvas"); if(!s) return;
  const {ctx,W,H}=s, PAD={l:42,r:14,t:16,b:32};
  const {CW,CH}=drawAxes(ctx,W,H,PAD,"FPR","TPR");
  // Diagonal
  ctx.strokeStyle="#d1d5db"; ctx.setLineDash([4,4]); ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(PAD.l,PAD.t+CH); ctx.lineTo(PAD.l+CW,PAD.t); ctx.stroke();
  ctx.setLineDash([]);
  const pts=m.roc_curve||[]; if(!pts.length) return;
  ctx.strokeStyle="#f59e0b"; ctx.lineWidth=2.2;
  // Fill area
  ctx.beginPath();
  ctx.moveTo(PAD.l, PAD.t+CH);
  pts.forEach(([fpr,tpr])=>ctx.lineTo(PAD.l+fpr*CW, PAD.t+(1-tpr)*CH));
  ctx.lineTo(PAD.l+CW, PAD.t+CH); ctx.closePath();
  ctx.fillStyle="rgba(245,158,11,.1)"; ctx.fill();
  // Line
  ctx.beginPath();
  pts.forEach(([fpr,tpr],i)=>{ const x=PAD.l+fpr*CW, y=PAD.t+(1-tpr)*CH; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
  ctx.stroke();
  ctx.fillStyle="#d97706"; ctx.font="bold 10px 'JetBrains Mono',monospace";
  ctx.textAlign="left"; ctx.fillText(`AUC = ${m.roc_auc}`, PAD.l+6, PAD.t+16);
}

function drawPR(m) {
  const s = getCtx("prCanvas"); if(!s) return;
  const {ctx,W,H}=s, PAD={l:42,r:14,t:16,b:32};
  const {CW,CH}=drawAxes(ctx,W,H,PAD,"Recall","Precision");
  const bl=(m.test_fraud||98)/(m.test_total||56962);
  ctx.strokeStyle="#d1d5db"; ctx.setLineDash([4,4]); ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(PAD.l,PAD.t+(1-bl)*CH); ctx.lineTo(PAD.l+CW,PAD.t+(1-bl)*CH); ctx.stroke();
  ctx.setLineDash([]);
  const pts=m.pr_curve||[]; if(!pts.length) return;
  ctx.strokeStyle="#3b82f6"; ctx.lineWidth=2.2;
  ctx.beginPath();
  pts.forEach(([rec,prec],i)=>{ const x=PAD.l+rec*CW, y=PAD.t+(1-prec)*CH; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
  ctx.stroke();
  ctx.fillStyle="#1d4ed8"; ctx.font="bold 10px 'JetBrains Mono',monospace";
  ctx.textAlign="left"; ctx.fillText(`AP = ${m.avg_prec}`, PAD.l+6, PAD.t+16);
}

function drawCM(m) {
  const s = getCtx("cmCanvas"); if(!s) return;
  const {ctx,W,H}=s;
  ctx.fillStyle="#fff"; ctx.fillRect(0,0,W,H);
  const tn=m.tn||0,fp=m.fp||0,fn=m.fn||0,tp=m.tp||0;
  const total=tn+fp+fn+tp||1;
  const cells=[
    {v:tn,lbl:"TN",sub:"True Neg",  bg:"#f0fdf4",c:"#16a34a",x:0,y:0},
    {v:fp,lbl:"FP",sub:"False Pos", bg:"#fef2f2",c:"#dc2626",x:1,y:0},
    {v:fn,lbl:"FN",sub:"False Neg", bg:"#eff6ff",c:"#2563eb",x:0,y:1},
    {v:tp,lbl:"TP",sub:"True Pos",  bg:"#fffbeb",c:"#d97706",x:1,y:1},
  ];
  const PAD=32, cw=(W-PAD*2)/2, ch=(H-PAD*2-20)/2;
  // Header labels
  ctx.fillStyle="#374151"; ctx.font="bold 10px 'JetBrains Mono',monospace";
  ctx.textAlign="center";
  ctx.fillText("Pred: Legit", PAD+cw/2, 18);
  ctx.fillText("Pred: Fraud", PAD+cw+cw/2, 18);
  ctx.save(); ctx.rotate(-Math.PI/2);
  ctx.fillText("Actual: Legit",  -(PAD+20+ch/2),  12);
  ctx.fillText("Actual: Fraud",  -(PAD+20+ch+ch/2), 12);
  ctx.restore();
  cells.forEach(({v,lbl,sub,bg,c,x,y})=>{
    const cx=PAD+x*cw, cy=PAD+20+y*ch;
    ctx.fillStyle=bg; ctx.fillRect(cx+2,cy+2,cw-4,ch-4);
    ctx.strokeStyle="#e5e7eb"; ctx.lineWidth=1; ctx.strokeRect(cx+2,cy+2,cw-4,ch-4);
    ctx.fillStyle=c; ctx.font=`bold 22px 'JetBrains Mono',monospace`;
    ctx.textAlign="center"; ctx.fillText(v.toLocaleString(), cx+cw/2, cy+ch/2-4);
    ctx.font=`bold 11px 'JetBrains Mono',monospace`;
    ctx.fillText(lbl, cx+cw/2, cy+ch/2+14);
    ctx.fillStyle="#6b7280"; ctx.font=`9px 'JetBrains Mono',monospace`;
    ctx.fillText(`${(v/total*100).toFixed(1)}%`, cx+cw/2, cy+ch/2+26);
  });
}

function drawClassReport(m) {
  const el = document.getElementById("classReport");
  if (!el) return;
  const thr   = m.threshold || 0.82;
  const tn=m.tn||0,fp=m.fp||0,fn=m.fn||0,tp=m.tp||0;
  const lP=tn/(tn+fn||1), lR=tn/(tn+fp||1), lF=2*lP*lR/(lP+lR||1);
  const fP=m.precision||0, fR=m.recall||0, fF=m.f1||0;
  const mP=((lP+fP)/2).toFixed(4), mR=((lR+fR)/2).toFixed(4), mF=((lF+fF)/2).toFixed(4);
  function clr(v){return v>=0.85?"cr-good":v>=0.60?"cr-warn":"cr-bad";}
  el.innerHTML=`<table class="cr-table">
    <thead><tr><th>CLASS</th><th>PRECISION</th><th>RECALL</th><th>F1</th><th>SUPPORT</th></tr></thead>
    <tbody>
      <tr><td>Legitimate</td><td class="${clr(lP)}">${lP.toFixed(4)}</td><td class="${clr(lR)}">${lR.toFixed(4)}</td><td class="${clr(lF)}">${lF.toFixed(4)}</td><td>${(tn+fp).toLocaleString()}</td></tr>
      <tr><td>Fraud</td><td class="${clr(fP)}">${Number(fP).toFixed(4)}</td><td class="${clr(fR)}">${Number(fR).toFixed(4)}</td><td class="${clr(fF)}">${Number(fF).toFixed(4)}</td><td>${(tp+fn).toLocaleString()}</td></tr>
      <tr><td>Macro Avg</td><td>${mP}</td><td>${mR}</td><td>${mF}</td><td>${(tn+fp+tp+fn).toLocaleString()}</td></tr>
    </tbody></table>
    <div style="margin-top:10px;font-size:9px;color:#6b7280;font-family:'JetBrains Mono',monospace">
      Decision threshold: ${thr} &nbsp;|&nbsp; AUC-ROC: ${m.roc_auc} &nbsp;|&nbsp; AUC-PR: ${m.avg_prec}
    </div>`;
}

function drawFI(m) {
  const s = getCtx("fiCanvas"); if(!s) return;
  const {ctx,W,H}=s;
  ctx.fillStyle="#fff"; ctx.fillRect(0,0,W,H);
  const fi=m.feature_importance||[];
  if(!fi.length) return;
  const top=fi.slice(0,12);
  const maxVal=top[0].importance;
  const PAD={l:140,r:60,t:16,b:20};
  const barH=Math.floor((H-PAD.t-PAD.b)/top.length)-3;
  const CW=W-PAD.l-PAD.r;
  top.forEach((f,i)=>{
    const y=PAD.t+i*(barH+3);
    const bw=(f.importance/maxVal)*CW;
    // Bar
    ctx.fillStyle="rgba(245,158,11,.15)"; ctx.fillRect(PAD.l,y,CW,barH);
    ctx.fillStyle="#f59e0b"; ctx.fillRect(PAD.l,y,bw,barH);
    // Label
    ctx.fillStyle="#374151"; ctx.font=`${Math.min(barH-1,11)}px 'JetBrains Mono',monospace`;
    ctx.textAlign="right"; ctx.fillText(f.feature, PAD.l-5, y+barH*0.72);
    // Value
    ctx.fillStyle="#6b7280"; ctx.textAlign="left";
    ctx.fillText(f.importance.toFixed(4), PAD.l+bw+5, y+barH*0.72);
  });
  ctx.fillStyle="#374151"; ctx.font="bold 9px 'JetBrains Mono',monospace";
  ctx.textAlign="center"; ctx.fillText("Mean |SHAP value|", PAD.l+CW/2, H-4);
}

// ── Batch ─────────────────────────────────────────────────────────
function handleDrop(e){e.preventDefault();const file=e.dataTransfer.files[0];if(file)processBatch(file);}
function handleBatchFile(e){processBatch(e.target.files[0]);}

async function processBatch(file) {
  const prog=document.getElementById("batchProgress"),bar=document.getElementById("bpBar"),lbl=document.getElementById("bpLabel");
  prog.style.display="block"; document.getElementById("batchResult").style.display="none";
  lbl.textContent="Uploading..."; bar.style.width="20%"; await sleep(200);
  lbl.textContent="Running predictions..."; bar.style.width="60%";
  const fd=new FormData(); fd.append("file",file);
  try {
    const res=await fetch(`${API}/api/predict_batch`,{method:"POST",body:fd});
    const data=await res.json();
    bar.style.width="100%"; lbl.textContent="Done!"; await sleep(300);
    prog.style.display="none"; renderBatch(data);
    log(`Batch: ${data.total} rows | ${data.fraud_detected} fraud | ${data.fraud_rate_pct}%`,"warn");
  } catch(e) { prog.style.display="none"; alert("Backend unreachable: "+e.message); }
}

function renderBatch(data) {
  document.getElementById("batchResult").style.display="block";
  document.getElementById("batchStats").innerHTML=[
    {l:"TOTAL",v:data.total,c:"var(--text)"},
    {l:"FRAUD",v:data.fraud_detected,c:"var(--red)"},
    {l:"LEGIT",v:data.legitimate,c:"var(--green)"},
    {l:"UNCERTAIN",v:data.uncertain,c:"var(--blue)"},
    {l:"FRAUD RATE",v:data.fraud_rate_pct+"%",c:"var(--amber)"},
  ].map(s=>`<div class="bs-card"><span class="bs-label">${s.l}</span><span class="bs-val" style="color:${s.c}">${s.v}</span></div>`).join("");
  document.getElementById("batchTbody").innerHTML=(data.results||[]).map((r,i)=>{
    const pc=r.prediction==="FRAUD"?"var(--red)":"var(--green)";
    return`<tr><td style="color:var(--dim2)">${i+1}</td><td>$${r.amount.toFixed(2)}</td><td style="color:${pc};font-weight:600">${r.probability_pct}%</td><td><span class="risk-badge risk-${r.risk_level}" style="font-size:9px">${r.prediction}</span></td><td><span class="risk-badge risk-${r.risk_level}" style="font-size:9px">${r.risk_level}</span></td><td><span class="conf-badge conf-${r.confidence}" style="font-size:9px">${r.confidence}</span></td></tr>`;
  }).join("");
}

// ── Log ───────────────────────────────────────────────────────────
function log(msg,type="info"){
  const feed=document.getElementById("logFeed"),now=new Date().toLocaleTimeString("en-GB");
  const el=document.createElement("div"); el.className=`log-item log-${type}`; el.textContent=`[${now}] ${msg}`;
  feed.prepend(el); while(feed.children.length>60) feed.lastChild.remove();
}
function clearLog(){document.getElementById("logFeed").innerHTML='<div class="log-item log-info">Log cleared</div>';}
function sleep(ms){return new Promise(r=>setTimeout(r,ms));}

// ── INIT ─────────────────────────────────────────────────────────
buildVGrid();
loadPreset("legit");
loadMetrics();
loadDemoCases();
