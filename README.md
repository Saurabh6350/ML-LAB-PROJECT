# FraudSentinel — Credit Card Fraud Detection Platform
## Full-Stack ML System with Novelty Hybrid Detection

---

## PROJECT STRUCTURE
```
fraudapp/
├── README.md
├── requirements.txt
├── train.py              ← STEP 1: Train model on your dataset
├── app.py                ← STEP 2: Run Flask backend
├── model/                ← Auto-created after training
│   ├── xgb_model.joblib
│   ├── iso_forest.joblib
│   ├── scaler.joblib
│   ├── feature_cols.joblib
│   └── metrics.json
├── static/
│   ├── css/style.css
│   └── js/main.js
└── templates/
    └── index.html
```

---

## SETUP

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train on YOUR dataset
```bash
# With Kaggle creditcard.csv:
python train.py --data creditcard.csv

# Without dataset (uses synthetic data to demo):
python train.py
```
Dataset format: CSV with columns `Time, V1-V28, Amount, Class`  
`Class`: 0 = legitimate, 1 = fraud

### 3. Run the platform
```bash
python app.py
```
Open: http://localhost:5000

---

## NOVELTY APPROACH

### 1. Hybrid Isolation Forest + XGBoost
Standard approaches only use supervised XGBoost. This system runs an **Isolation Forest** (unsupervised) in parallel — its anomaly score becomes a meta-feature injected into XGBoost. This means the model catches anomalies even when labelled fraud examples are scarce.

### 2. Velocity Feature Engineering
Raw V1-V28 features are PCA-transformed. On top, we engineer:
- `amount_zscore` — how extreme the amount is vs dataset distribution
- `time_bin` — hour-of-day bucket (fraud spikes at night)
- `amount_log` — log-scaled amount (handles skew)

### 3. Confidence-Gated Predictions
Binary fraud/legit is too crude. Every prediction has:
- **Fraud probability** (0–100%)
- **Confidence band**: CERTAIN / PROBABLE / UNCERTAIN
- Uncertain predictions (0.4–0.6 range) are flagged for human review

### 4. SHAP Explainability
Every single prediction comes with a natural-language explanation of which features drove the score up or down, making results auditable.

---

## API ENDPOINTS

| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/` | Frontend dashboard |
| GET  | `/api/metrics` | Model performance metrics |
| POST | `/api/predict` | Single transaction prediction |
| POST | `/api/predict_batch` | Batch CSV prediction |
| GET  | `/api/feature_importance` | Top feature importances |

### Single Prediction Request
```json
POST /api/predict
{
  "Time": 1234,
  "V1": -1.36, "V2": -0.07, ..., "V28": 0.02,
  "Amount": 149.62
}
```

### Response
```json
{
  "prediction": "FRAUD",
  "probability": 0.9341,
  "confidence": "CERTAIN",
  "iso_score": -0.142,
  "risk_level": "CRITICAL",
  "explanation": ["V14 strongly indicates fraud", "High amount relative to baseline"],
  "shap_values": {"V14": -0.82, "Amount": 0.41, ...}
}
```
