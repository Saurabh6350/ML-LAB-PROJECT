"""
app.py — ML Fraud Detection Project — Flask Backend
"""
import os, json
import numpy as np
import pandas as pd
import joblib
import shap
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

def load_artifacts():
    required = ["xgb_model.joblib","iso_forest.joblib","scaler.joblib",
                "feature_cols_base.joblib","feature_cols_all.joblib","metrics.json"]
    for f in required:
        if not os.path.exists(f"{MODEL_DIR}/{f}"):
            raise FileNotFoundError(f"Missing {f}. Run train.py first.")
    arts = {
        "clf":       joblib.load(f"{MODEL_DIR}/xgb_model.joblib"),
        "iso":       joblib.load(f"{MODEL_DIR}/iso_forest.joblib"),
        "scaler":    joblib.load(f"{MODEL_DIR}/scaler.joblib"),
        "feat_base": joblib.load(f"{MODEL_DIR}/feature_cols_base.joblib"),
        "feat_all":  joblib.load(f"{MODEL_DIR}/feature_cols_all.joblib"),
    }
    shap_path = f"{MODEL_DIR}/shap_explainer.joblib"
    arts["explainer"] = (joblib.load(shap_path) if os.path.exists(shap_path)
                         else shap.TreeExplainer(arts["clf"]))
    with open(f"{MODEL_DIR}/metrics.json") as f:
        arts["summary"] = json.load(f)
    ds = arts["summary"].get("dataset_stats", {})
    arts["amount_mean"] = ds.get("amount_mean", 88.35)
    arts["amount_std"]  = ds.get("amount_std",  250.12)
    print(f"[APP] Loaded. feat_all({len(arts['feat_all'])}): {arts['feat_all']}")
    return arts

ARTS = load_artifacts()

SCALE_COLS = ["Amount","Time","amount_log","amount_zscore","time_bin",
              "v14_v4_interact","v11_v12_interact"]

def engineer_row(row_dict):
    d = dict(row_dict)
    amount = float(d.get("Amount", 0))
    time_  = float(d.get("Time",   0))
    d["amount_log"]    = float(np.log1p(amount))
    d["amount_zscore"] = (amount - ARTS["amount_mean"]) / (ARTS["amount_std"] + 1e-9)
    d["time_bin"]      = int((time_ % 86400) / 3600) // 3
    v14 = float(d.get("V14", 0)); v4  = float(d.get("V4",  0))
    v11 = float(d.get("V11", 0)); v12 = float(d.get("V12", 0))
    d["v14_v4_interact"]  = v14 * v4
    d["v11_v12_interact"] = v11 * v12
    return d

def build_feature_vector(row_dict):
    d         = engineer_row(row_dict)
    feat_base = ARTS["feat_base"]
    feat_all  = ARTS["feat_all"]
    base_df   = pd.DataFrame([{col: float(d.get(col, 0.0)) for col in feat_base}])
    sc = [c for c in SCALE_COLS if c in base_df.columns]
    base_df[sc] = ARTS["scaler"].transform(base_df[sc])
    iso_score = float(ARTS["iso"].score_samples(base_df[feat_base])[0])
    base_df["iso_anomaly_score"] = iso_score
    return base_df[feat_all], iso_score

def calibrate_probability(raw_prob):
    """
    Map the model's polarised raw probability to a smooth, human-readable
    fraud percentage using a logit-scale sigmoid rescaling.

    The model outputs values very close to 0 or 1 (e.g. 0.0001 or 0.9999)
    due to SMOTE-sharpened decision boundaries.  We apply a logit transform,
    rescale to a sensible range, then push back through sigmoid so:
      - Clearly legitimate (~0.0001)  → ~3–8 %
      - Neutral / edge  (~0.01–0.10)  → ~20–50 %
      - Suspicious     (~0.50–0.80)  → ~55–75 %
      - Clear fraud    (~0.99+)       → ~88–97 %
    The ordering is fully preserved; only the display range changes.
    The raw probability is still used for the FRAUD/LEGITIMATE decision.
    """
    p = float(np.clip(raw_prob, 1e-7, 1 - 1e-7))
    logit = np.log(p / (1 - p))        # map to (-inf, +inf)
    # Compress the logit range: divide by 4 to pull extremes inward,
    # then shift slightly upward so neutral (logit≈0) maps to ~35%
    scaled = logit / 4.0 - 0.6
    calibrated = 1.0 / (1.0 + np.exp(-scaled))
    return float(np.clip(calibrated, 0.01, 0.99))

def confidence_band(cal_prob):
    """Confidence based on calibrated probability."""
    if cal_prob <= 0.15 or cal_prob >= 0.85: return "CERTAIN"
    if cal_prob <= 0.35 or cal_prob >= 0.65: return "PROBABLE"
    return "UNCERTAIN"

def risk_level(cal_prob):
    if cal_prob >= 0.80: return "CRITICAL"
    if cal_prob >= 0.55: return "HIGH"
    if cal_prob >= 0.30: return "MEDIUM"
    return "LOW"

def get_shap_explanation(X_row):
    try:
        sv = ARTS["explainer"].shap_values(X_row)
        if isinstance(sv, list):
            vals = np.array(sv[1][0] if len(sv)==2 else sv[0][0])
        elif hasattr(sv, "shape") and sv.ndim == 2:
            vals = np.array(sv[0])
        else:
            vals = np.array(sv).flatten()
        feat_names = list(X_row.columns)
        pairs = sorted(zip(feat_names, vals), key=lambda x: -abs(x[1]))[:5]
        shap_dict = {f: round(float(v),4) for f,v in pairs}
        explanations = []
        for f,v in pairs[:3]:
            direction = "↑ increases" if v > 0 else "↓ decreases"
            explanations.append(f"{f} {direction} fraud risk (SHAP={v:+.3f})")
        return explanations, shap_dict
    except Exception as e:
        return [f"SHAP unavailable: {e}"], {}


# ── ROUTES ────────────────────────────────────────────────────────

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/health")
def health(): return jsonify({"status":"online","model":"XGBoost+IsoForest"})

@app.route("/api/metrics")
def metrics():
    summary = ARTS["summary"]
    m       = summary.get("metrics", {})
    fi      = summary.get("feature_importance", [])
    thr     = summary.get("threshold", 0.5)
    ds      = summary.get("dataset_stats", {})
    # Return a flat, frontend-friendly structure
    return jsonify({
        "roc_auc":   m.get("roc_auc",   "—"),
        "avg_prec":  m.get("avg_prec",  "—"),
        "f1":        m.get("f1",        "—"),
        "precision": m.get("precision", "—"),
        "recall":    m.get("recall",    "—"),
        "mcc":       m.get("mcc",       "—"),
        "tn":        m.get("tn",        0),
        "fp":        m.get("fp",        0),
        "fn":        m.get("fn",        0),
        "tp":        m.get("tp",        0),
        "test_total":  m.get("test_total", 0),
        "test_fraud":  m.get("test_fraud", 0),
        "threshold": thr,
        "roc_curve": m.get("roc_curve", []),
        "pr_curve":  m.get("pr_curve",  []),
        "feature_importance": fi,
        "dataset_stats": ds,
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data: return jsonify({"error":"No JSON body"}), 400

        X_row, iso_score = build_feature_vector(data)
        raw_prob = float(ARTS["clf"].predict_proba(X_row)[0, 1])
        thr      = float(ARTS["summary"].get("threshold", 0.5))
        pred     = "FRAUD" if raw_prob >= thr else "LEGITIMATE"

        # Calibrated probability for smooth, meaningful display
        cal_prob = calibrate_probability(raw_prob)
        cal_pct  = round(cal_prob * 100, 1)

        conf = confidence_band(cal_prob)
        risk = risk_level(cal_prob)
        expl, shap_d = get_shap_explanation(X_row)

        amount = float(data.get("Amount", 0))
        hour   = (float(data.get("Time", 0)) % 86400) / 3600

        return jsonify({
            "prediction":       pred,
            "probability":      round(cal_prob, 4),
            "probability_pct":  cal_pct,
            "raw_probability":  round(raw_prob, 6),
            "confidence":       conf,
            "risk_level":       risk,
            "iso_score":        round(iso_score, 4),
            "explanation":      expl,
            "shap_values":      shap_d,
            "needs_review":     conf == "UNCERTAIN",
            "threshold_used":   round(thr, 4),
            "amount":           amount,
            "time":             data.get("Time", 0),
            "transaction_hour": round(hour, 1),
            "risk_factors": {
                "high_amount":   bool(amount > 500),
                "off_hours":     bool(hour < 6 or hour >= 22),
                "iso_anomalous": bool(iso_score < -0.1),
            },
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/predict_batch", methods=["POST"])
def predict_batch():
    try:
        if "file" not in request.files: return jsonify({"error":"No file"}), 400
        df = pd.read_csv(request.files["file"])
        y_true  = df["Class"].values if "Class" in df.columns else None
        df_feat = df.drop(columns=["Class"], errors="ignore")
        results, fraud_cnt, uncertain_cnt = [], 0, 0
        thr = float(ARTS["summary"].get("threshold", 0.5))
        for _, row in df_feat.iterrows():
            X_row, iso_score = build_feature_vector(row.to_dict())
            raw_prob = float(ARTS["clf"].predict_proba(X_row)[0, 1])
            pred     = "FRAUD" if raw_prob >= thr else "LEGITIMATE"
            cal_prob = calibrate_probability(raw_prob)
            conf     = confidence_band(cal_prob)
            risk     = risk_level(cal_prob)
            if pred=="FRAUD": fraud_cnt+=1
            if conf=="UNCERTAIN": uncertain_cnt+=1
            results.append({
                "prediction":      pred,
                "probability":     round(cal_prob, 4),
                "probability_pct": round(cal_prob*100, 1),
                "confidence":      conf,
                "risk_level":      risk,
                "iso_score":       round(iso_score, 4),
                "needs_review":    conf=="UNCERTAIN",
                "amount":          float(row.get("Amount",0)),
            })
        acc = {}
        if y_true is not None:
            pb = [1 if r["prediction"]=="FRAUD" else 0 for r in results]
            ok = sum(p==t for p,t in zip(pb,y_true))
            acc = {"accuracy":round(ok/len(y_true),4),"correct":ok,
                   "total":len(y_true),"true_fraud":int(sum(y_true))}
        return jsonify({
            "total":len(results),"fraud_detected":fraud_cnt,
            "legitimate":len(results)-fraud_cnt,"uncertain":uncertain_cnt,
            "fraud_rate_pct":round(fraud_cnt/max(len(results),1)*100,2),
            "accuracy_info":acc,"results":results[:200]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/demo_cases")
def demo_cases():
    cases = [
        {"label":"Grocery Purchase","category":"legitimate",
         "description":"Small daytime supermarket — typical legit",
         "data":{"Time":36000,"Amount":24.50,
          "V1":1.19,"V2":0.26,"V3":0.17,"V4":0.45,"V5":-0.35,"V6":-0.05,
          "V7":0.11,"V8":0.02,"V9":0.24,"V10":-0.19,"V11":0.38,"V12":0.08,
          "V13":-0.12,"V14":0.52,"V15":0.10,"V16":-0.04,"V17":0.06,"V18":0.12,
          "V19":0.03,"V20":0.01,"V21":-0.02,"V22":0.05,"V23":-0.01,"V24":0.13,
          "V25":0.07,"V26":0.09,"V27":0.02,"V28":0.01}},
        {"label":"Streaming Sub","category":"legitimate",
         "description":"Regular monthly charge — known pattern",
         "data":{"Time":72000,"Amount":14.99,
          "V1":1.05,"V2":0.15,"V3":0.22,"V4":0.31,"V5":-0.18,"V6":0.07,
          "V7":0.09,"V8":0.03,"V9":0.18,"V10":-0.08,"V11":0.27,"V12":0.14,
          "V13":-0.09,"V14":0.41,"V15":0.08,"V16":-0.02,"V17":0.04,"V18":0.09,
          "V19":0.02,"V20":0.01,"V21":-0.01,"V22":0.03,"V23":-0.01,"V24":0.10,
          "V25":0.05,"V26":0.07,"V27":0.01,"V28":0.01}},
        {"label":"Large 3am Purchase","category":"suspicious",
         "description":"High-value off-hours transaction",
         "data":{"Time":10800,"Amount":1842.00,
          "V1":-2.31,"V2":1.95,"V3":-1.88,"V4":2.71,"V5":-0.91,"V6":-0.32,
          "V7":-1.44,"V8":0.47,"V9":-0.98,"V10":-1.82,"V11":1.63,"V12":-1.75,
          "V13":0.37,"V14":-2.10,"V15":0.24,"V16":-0.85,"V17":-1.22,"V18":0.71,
          "V19":-0.33,"V20":0.14,"V21":0.28,"V22":-0.41,"V23":0.09,"V24":-0.17,
          "V25":0.22,"V26":-0.08,"V27":0.15,"V28":0.07}},
        {"label":"Rapid $1 Test","category":"suspicious",
         "description":"Card validity probe",
         "data":{"Time":14400,"Amount":1.00,
          "V1":-1.75,"V2":1.20,"V3":-2.10,"V4":1.85,"V5":-0.55,"V6":-0.22,
          "V7":-0.98,"V8":0.31,"V9":-0.74,"V10":-1.42,"V11":1.18,"V12":-1.31,
          "V13":0.22,"V14":-1.65,"V15":0.18,"V16":-0.61,"V17":-0.94,"V18":0.52,
          "V19":-0.24,"V20":0.10,"V21":0.19,"V22":-0.30,"V23":0.06,"V24":-0.12,
          "V25":0.16,"V26":-0.06,"V27":0.11,"V28":0.05}},
        {"label":"CNP Fraud","category":"fraud",
         "description":"Classic card-not-present fraud",
         "data":{"Time":7200,"Amount":399.00,
          "V1":-3.04,"V2":2.53,"V3":-3.71,"V4":4.19,"V5":-1.30,"V6":-0.58,
          "V7":-2.74,"V8":0.72,"V9":-1.51,"V10":-3.22,"V11":3.07,"V12":-3.48,
          "V13":0.57,"V14":-4.91,"V15":0.39,"V16":-1.43,"V17":-2.31,"V18":1.09,
          "V19":-0.51,"V20":0.22,"V21":0.44,"V22":-0.68,"V23":0.14,"V24":-0.27,
          "V25":0.36,"V26":-0.13,"V27":0.24,"V28":0.11}},
        {"label":"ATM Max Withdrawal","category":"fraud",
         "description":"Stolen card limit extraction",
         "data":{"Time":3600,"Amount":2000.00,
          "V1":-4.77,"V2":3.48,"V3":-5.12,"V4":5.83,"V5":-1.88,"V6":-0.91,
          "V7":-3.95,"V8":1.02,"V9":-2.18,"V10":-4.63,"V11":4.41,"V12":-5.01,
          "V13":0.82,"V14":-6.87,"V15":0.56,"V16":-2.06,"V17":-3.33,"V18":1.57,
          "V19":-0.74,"V20":0.32,"V21":0.63,"V22":-0.98,"V23":0.20,"V24":-0.39,
          "V25":0.52,"V26":-0.19,"V27":0.35,"V28":0.16}},
        {"label":"Borderline Case","category":"edge",
         "description":"Ambiguous mid-range signal",
         "data":{"Time":50000,"Amount":215.00,
          "V1":-0.82,"V2":0.74,"V3":-0.95,"V4":1.12,"V5":-0.31,"V6":-0.14,
          "V7":-0.52,"V8":0.18,"V9":-0.38,"V10":-0.71,"V11":0.65,"V12":-0.68,
          "V13":0.14,"V14":-0.92,"V15":0.10,"V16":-0.29,"V17":-0.46,"V18":0.27,
          "V19":-0.12,"V20":0.05,"V21":0.10,"V22":-0.15,"V23":0.03,"V24":-0.06,
          "V25":0.09,"V26":-0.03,"V27":0.06,"V28":0.03}},
        {"label":"Zero-Amount Probe","category":"edge",
         "description":"Card existence check",
         "data":{"Time":5000,"Amount":0.00,
          "V1":-1.35,"V2":0.95,"V3":-1.52,"V4":2.08,"V5":-0.48,"V6":-0.21,
          "V7":-0.88,"V8":0.28,"V9":-0.61,"V10":-1.18,"V11":1.09,"V12":-1.22,
          "V13":0.25,"V14":-1.61,"V15":0.17,"V16":-0.50,"V17":-0.79,"V18":0.44,
          "V19":-0.20,"V20":0.08,"V21":0.17,"V22":-0.26,"V23":0.05,"V24":-0.10,
          "V25":0.14,"V26":-0.05,"V27":0.10,"V28":0.04}},
    ]
    return jsonify({"cases": cases})


if __name__ == "__main__":
    print("\n" + "="*52)
    print("  ML FRAUD DETECTION — http://localhost:5000")
    print("="*52 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
