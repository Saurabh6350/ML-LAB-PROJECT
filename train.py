"""
train.py — ML Fraud Detection Project — Training Pipeline
==========================================================
Novelties:
  1. Isolation Forest anomaly score as supervised meta-feature
  2. V-feature interaction terms (V14xV4, V11xV12)
  3. SMOTE k=5 minority oversampling
  4. XGBoost with scale_pos_weight cost-sensitive learning
  5. Precision-Recall optimal threshold tuning (maximises F1)
  6. Isotonic probability calibration for reliable scores
  7. SHAP TreeExplainer per-prediction explainability
  8. Dataset stats saved for exact inference-time z-scoring
  9. Full diagnostic plots saved to plots/

Usage:
    python train.py --data creditcard.csv
    python train.py                        # synthetic fallback
"""

import argparse, json, os, time, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler
from sklearn.ensemble         import IsolationForest
from sklearn.calibration      import CalibratedClassifierCV, calibration_curve
from sklearn.metrics          import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report, matthews_corrcoef
)
from imblearn.over_sampling   import SMOTE
from xgboost                  import XGBClassifier
import shap

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor":"#080B11","axes.facecolor":"#0D1018",
    "axes.edgecolor":"#1A2030","axes.labelcolor":"#CDD5E0",
    "xtick.color":"#5A6880","ytick.color":"#5A6880",
    "text.color":"#CDD5E0","grid.color":"#1A2030","grid.linewidth":0.6,
    "font.family":"monospace","font.size":10,
    "axes.titlesize":12,"axes.titlecolor":"#F5A623","axes.titleweight":"bold",
    "figure.dpi":150,
})
AMBER="#F5A623"; RED="#E8455A"; GREEN="#2ECC71"; BLUE="#4A9EFF"; PURPLE="#A78BFA"; DIM="#3D4A60"
_T0 = time.time()
def _t(): return f"{time.time()-_T0:5.1f}s"
def _save(fig, name):
    p = f"{PLOTS_DIR}/{name}"
    fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); print(f"[PLOT] {name}")


# 1. DATA
def load_data(csv_path=None):
    if csv_path and os.path.exists(csv_path):
        print(f"[DATA] Loading: {csv_path}")
        df = pd.read_csv(csv_path)
        missing = {"Class","Amount","Time"} - set(df.columns)
        if missing: raise ValueError(f"CSV missing: {missing}")
        print(f"[DATA] {len(df):,} rows | fraud={df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
        return df
    print("[DATA] Generating synthetic data...")
    np.random.seed(42); N_L, N_F = 284315, 492
    def mc(n,lbl,a_mu,a_sig,sh):
        d={"Time":np.random.uniform(0,172792,n),
           "Amount":np.abs(np.random.lognormal(a_mu,a_sig,n)),"Class":lbl}
        for i in range(1,29): d[f"V{i}"]=np.random.normal(sh.get(i,0),1.0+0.5*(i%3),n)
        return pd.DataFrame(d)
    fs={4:-3.1,11:-4.2,12:-3.8,14:-4.5,17:-2.9,10:-2.5,3:3.2,7:2.8}
    df=pd.concat([mc(N_L,0,3.2,1.8,{}),mc(N_F,1,4.5,2.1,fs)]).sample(frac=1,random_state=42).reset_index(drop=True)
    print(f"[DATA] {len(df):,} rows | fraud={N_F} ({N_F/len(df)*100:.3f}%)")
    return df


# 2. FEATURES — mirrored exactly in app.py
def engineer_features(df, amount_mean=None, amount_std=None):
    df = df.copy()
    amt = df["Amount"]
    if amount_mean is None: amount_mean = float(amt.mean())
    if amount_std  is None: amount_std  = float(amt.std())
    df["amount_log"]       = np.log1p(amt)
    df["amount_zscore"]    = (amt - amount_mean) / (amount_std + 1e-9)
    df["time_bin"]         = ((df["Time"] % 86400) / 3600).astype(int) // 3
    df["v14_v4_interact"]  = df["V14"] * df["V4"]
    df["v11_v12_interact"] = df["V11"] * df["V12"]
    return df, amount_mean, amount_std

SCALE_COLS = ["Amount","Time","amount_log","amount_zscore","time_bin",
              "v14_v4_interact","v11_v12_interact"]

def preprocess(df):
    df_eng, amount_mean, amount_std = engineer_features(df)
    feat_cols = [c for c in df_eng.columns if c != "Class"]
    X = df_eng[feat_cols].copy(); y = df_eng["Class"].values
    sc = [c for c in SCALE_COLS if c in X.columns]
    scaler = StandardScaler()
    X[sc] = scaler.fit_transform(X[sc])
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    print(f"[SPLIT] Train={len(X_train):,} | Test={len(X_test):,} | "
          f"Train-fraud={y_train.sum()} | Test-fraud={y_test.sum()}")
    return X_train,X_test,y_train,y_test,scaler,feat_cols,amount_mean,amount_std


# 3. ISOLATION FOREST
def add_isolation_score(X_train, X_test, feature_cols):
    print(f"[{_t()}][ISO] Training Isolation Forest...")
    iso = IsolationForest(n_estimators=200,contamination=0.002,max_samples="auto",random_state=42,n_jobs=-1)
    iso.fit(X_train[feature_cols])
    X_train=X_train.copy(); X_test=X_test.copy()
    X_train["iso_anomaly_score"]=iso.score_samples(X_train[feature_cols])
    X_test["iso_anomaly_score"] =iso.score_samples(X_test[feature_cols])
    print(f"[{_t()}][ISO] Done."); return X_train,X_test,iso


# 4. TRAIN
def train_model(X_train, y_train):
    print(f"[{_t()}][SMOTE] Oversampling (k=5)...")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res,y_res = sm.fit_resample(X_train,y_train)
    print(f"[{_t()}][SMOTE] Balanced: {len(X_res):,} samples")
    spw = max(1, int((y_res==0).sum() / max((y_res==1).sum(),1)))
    print(f"[{_t()}][XGB] Training XGBoost (n=400, scale_pos_weight={spw})...")
    clf = XGBClassifier(
        n_estimators=400, learning_rate=0.04, max_depth=6,
        min_child_weight=3, subsample=0.85, colsample_bytree=0.80,
        gamma=0.1, reg_alpha=0.05, reg_lambda=1.2,
        scale_pos_weight=spw, eval_metric="aucpr",
        use_label_encoder=False, random_state=42, n_jobs=-1,
    )
    clf.fit(X_res,y_res,eval_set=[(X_train,y_train)],verbose=False)
    print(f"[{_t()}][XGB] Done.")
    print(f"[{_t()}][CAL] Calibrating (isotonic)...")
    cal = CalibratedClassifierCV(clf, cv="prefit", method="isotonic")
    cal.fit(X_train, y_train)
    print(f"[{_t()}][CAL] Done.")
    return cal, clf


# 5. THRESHOLD
def find_best_threshold(y_true, y_prob):
    prec,rec,thrs = precision_recall_curve(y_true, y_prob)
    f1s = 2*prec*rec/(prec+rec+1e-9)
    idx = int(np.argmax(f1s[:-1]))
    thr = float(thrs[idx])
    print(f"[{_t()}][THR] Optimal={thr:.4f}  F1={f1s[idx]:.4f}  P={prec[idx]:.4f}  R={rec[idx]:.4f}")
    return thr


# 6. EVALUATE
def evaluate(clf, X_test, y_test, threshold=0.5):
    y_prob = clf.predict_proba(X_test)[:,1]
    y_pred = (y_prob>=threshold).astype(int)
    tn,fp,fn,tp = confusion_matrix(y_test,y_pred).ravel()
    fpr,tpr,_ = roc_curve(y_test,y_prob)
    step=max(1,len(fpr)//150)
    roc_pts=[[round(float(a),4),round(float(b),4)] for a,b in zip(fpr[::step],tpr[::step])]
    pc,rc,_ = precision_recall_curve(y_test,y_prob)
    step2=max(1,len(pc)//150)
    pr_pts=[[round(float(a),4),round(float(b),4)] for a,b in zip(rc[::step2],pc[::step2])]
    metrics={
        "roc_auc":   round(roc_auc_score(y_test,y_prob),4),
        "avg_prec":  round(average_precision_score(y_test,y_prob),4),
        "f1":        round(f1_score(y_test,y_pred),4),
        "precision": round(precision_score(y_test,y_pred,zero_division=0),4),
        "recall":    round(recall_score(y_test,y_pred),4),
        "mcc":       round(matthews_corrcoef(y_test,y_pred),4),
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
        "test_total":int(len(y_test)),"test_fraud":int(y_test.sum()),
        "threshold": round(threshold,4),
        "roc_curve": roc_pts,"pr_curve": pr_pts,
        "y_prob":y_prob.tolist(),"y_test":y_test.tolist(),
    }
    print("\n── Evaluation ─────────────────────────────────────")
    for k,v in metrics.items():
        if k not in ("roc_curve","pr_curve","y_prob","y_test"):
            print(f"  {k:12s}: {v}")
    return metrics,y_prob,y_pred


# 7. SHAP
def build_shap(raw_clf, X_train, feat_cols_all):
    print(f"[{_t()}][SHAP] Building explainer...")
    try:
        explainer = shap.TreeExplainer(raw_clf)
        sample    = X_train.sample(min(800,len(X_train)),random_state=42)
        shap_vals = explainer.shap_values(sample)
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        pairs = sorted(zip(feat_cols_all[:len(mean_abs)],mean_abs),key=lambda x:-x[1])[:15]
        fi = [{"feature":f,"importance":round(float(v),5)} for f,v in pairs]
        print(f"[{_t()}][SHAP] Done."); return explainer,fi,shap_vals,sample
    except Exception as e:
        print(f"[SHAP] Warning: {e}"); return None,[],None,None


# 8. PLOTS
def plot_confusion_matrix(y_test,y_pred):
    cm=confusion_matrix(y_test,y_pred); fig,ax=plt.subplots(figsize=(6,5))
    ax.imshow(cm,cmap=LinearSegmentedColormap.from_list("",["#0D1018","#F5A623"]))
    for (i,j),lbl,clr in zip([(0,0),(0,1),(1,0),(1,1)],["TN","FP","FN","TP"],[GREEN,RED,BLUE,AMBER]):
        ax.text(j,i,f"{lbl}\n{cm[i,j]:,}",ha="center",va="center",fontsize=13,fontweight="bold",color=clr)
    ax.set_xticks([0,1]);ax.set_yticks([0,1])
    ax.set_xticklabels(["Predicted\nLegit","Predicted\nFraud"])
    ax.set_yticklabels(["Actual\nLegit","Actual\nFraud"])
    ax.set_title("Confusion Matrix"); _save(fig,"confusion_matrix.png")

def plot_roc(y_test,y_prob,auc):
    fpr,tpr,_=roc_curve(y_test,y_prob); fig,ax=plt.subplots(figsize=(7,5))
    ax.plot(fpr,tpr,color=AMBER,lw=2.2,label=f"AUC = {auc:.4f}")
    ax.plot([0,1],[0,1],color=DIM,lw=1,ls="--",label="Random")
    ax.fill_between(fpr,tpr,alpha=0.08,color=AMBER)
    ax.set(xlabel="FPR",ylabel="TPR",title="ROC Curve",xlim=[0,1],ylim=[0,1.01])
    ax.legend(facecolor="#0D1018",edgecolor=DIM);ax.grid(True,alpha=0.3);_save(fig,"roc_curve.png")

def plot_pr(y_test,y_prob,avg_prec):
    prec,rec,_=precision_recall_curve(y_test,y_prob); bl=y_test.sum()/len(y_test)
    fig,ax=plt.subplots(figsize=(7,5))
    ax.plot(rec,prec,color=BLUE,lw=2.2,label=f"AP = {avg_prec:.4f}")
    ax.axhline(bl,color=DIM,lw=1,ls="--",label=f"Baseline ({bl:.4f})")
    ax.fill_between(rec,prec,alpha=0.08,color=BLUE)
    ax.set(xlabel="Recall",ylabel="Precision",title="Precision-Recall Curve",xlim=[0,1],ylim=[0,1.01])
    ax.legend(facecolor="#0D1018",edgecolor=DIM);ax.grid(True,alpha=0.3);_save(fig,"pr_curve.png")

def plot_classification_report(y_test,y_pred):
    rep=classification_report(y_test,y_pred,target_names=["Legit","Fraud"],output_dict=True)
    rows=["Legit","Fraud","macro avg","weighted avg"]
    data=[[f"{rep[r][c]:.4f}" if c!="support" else f"{int(rep[r][c]):,}"
           for c in ["precision","recall","f1-score","support"]] for r in rows]
    fig,ax=plt.subplots(figsize=(8,3.5)); ax.axis("off")
    tbl=ax.table(cellText=data,rowLabels=rows,colLabels=["Precision","Recall","F1","Support"],cellLoc="center",loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.2,2.0)
    for (i,j),cell in tbl.get_celld().items():
        cell.set_facecolor("#0D1018" if i>0 else "#111520"); cell.set_edgecolor("#1A2030")
        cell.set_text_props(color=AMBER if i==0 or j==-1 else "#CDD5E0",fontweight="bold" if i==0 else "normal")
    ax.set_title("Classification Report",pad=20); fig.patch.set_facecolor("#080B11"); _save(fig,"classification_report.png")

def plot_feature_importance(fi):
    if not fi: return
    names=[d["feature"] for d in fi[:15]]; vals=[d["importance"] for d in fi[:15]]
    fig,ax=plt.subplots(figsize=(8,6))
    bars=ax.barh(names[::-1],vals[::-1],color=AMBER,alpha=0.85,height=0.65)
    for bar,val in zip(bars,vals[::-1]):
        ax.text(bar.get_width()+max(vals)*0.01,bar.get_y()+bar.get_height()/2,f"{val:.4f}",va="center",fontsize=9,color="#CDD5E0")
    ax.set(xlabel="Mean |SHAP value|",title="Top 15 Features (SHAP)")
    ax.grid(True,axis="x",alpha=0.3); ax.set_xlim([0,max(vals)*1.15]); _save(fig,"feature_importance.png")

def plot_shap_summary(shap_vals,sample,feat_cols):
    if shap_vals is None: return
    try:
        feat_names=list(feat_cols)[:shap_vals.shape[1]]
        mean_abs=np.abs(shap_vals).mean(axis=0); top_idx=np.argsort(mean_abs)[-15:][::-1]
        fig,ax=plt.subplots(figsize=(9,7))
        for rank,fi in enumerate(top_idx[::-1]):
            sv=shap_vals[:,fi]; fv=sample.iloc[:,fi].values; fv_n=(fv-fv.min())/(fv.ptp()+1e-9)
            sc=ax.scatter(sv,np.full(len(sv),rank)+np.random.uniform(-0.3,0.3,len(sv)),c=fv_n,cmap="RdYlGn_r",alpha=0.4,s=8)
        ax.set_yticks(range(15)); ax.set_yticklabels([feat_names[i] for i in top_idx[::-1]],fontsize=9)
        ax.axvline(0,color=DIM,lw=1,ls="--"); ax.set(xlabel="SHAP value",title="SHAP Summary — Top 15 Features")
        plt.colorbar(sc,ax=ax,pad=0.01).set_label("Feature value (low→high)",fontsize=9)
        ax.grid(True,axis="x",alpha=0.2); _save(fig,"shap_summary.png")
    except Exception as e: print(f"[SHAP PLOT] Skipped: {e}")

def plot_calibration(y_test,y_prob):
    frac,mean_pred=calibration_curve(y_test,y_prob,n_bins=15,strategy="quantile")
    fig,ax=plt.subplots(figsize=(6,5))
    ax.plot([0,1],[0,1],color=DIM,lw=1,ls="--",label="Perfect")
    ax.plot(mean_pred,frac,color=PURPLE,lw=2.2,marker="o",ms=5,label="Model")
    ax.fill_between(mean_pred,frac,mean_pred,alpha=0.12,color=PURPLE)
    ax.set(xlabel="Mean predicted prob",ylabel="Fraction positives",title="Calibration Curve",xlim=[0,1],ylim=[0,1])
    ax.legend(facecolor="#0D1018",edgecolor=DIM); ax.grid(True,alpha=0.3); _save(fig,"calibration_curve.png")

def plot_score_distribution(y_test,y_prob):
    bins=np.linspace(0,1,50); fig,ax=plt.subplots(figsize=(8,5))
    ax.hist(y_prob[y_test==0],bins=bins,color=GREEN,alpha=0.65,label="Legit",density=True)
    ax.hist(y_prob[y_test==1],bins=bins,color=RED,alpha=0.75,label="Fraud",density=True)
    ax.set(xlabel="Fraud probability",ylabel="Density",title="Score Distribution")
    ax.legend(facecolor="#0D1018",edgecolor=DIM); ax.grid(True,alpha=0.3); _save(fig,"score_distribution.png")

def plot_threshold_sweep(y_test,y_prob):
    thrs=np.linspace(0.01,0.99,200); f1s,precs,recs=[],[],[]
    for t in thrs:
        yp=(y_prob>=t).astype(int)
        f1s.append(f1_score(y_test,yp,zero_division=0))
        precs.append(precision_score(y_test,yp,zero_division=0))
        recs.append(recall_score(y_test,yp))
    best_t=thrs[np.argmax(f1s)]
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(thrs,f1s,color=AMBER,lw=2.0,label="F1")
    ax.plot(thrs,precs,color=GREEN,lw=1.5,ls="--",label="Precision")
    ax.plot(thrs,recs,color=RED,lw=1.5,ls="--",label="Recall")
    ax.axvline(best_t,color=PURPLE,lw=1.2,ls=":",label=f"Best={best_t:.2f}")
    ax.set(xlabel="Threshold",ylabel="Score",title="F1/Precision/Recall vs Threshold",xlim=[0,1],ylim=[0,1.05])
    ax.legend(facecolor="#0D1018",edgecolor=DIM); ax.grid(True,alpha=0.3); _save(fig,"threshold_sweep.png")

def plot_amount_distribution(df):
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,log_s,ttl in zip(axes,[False,True],["Amount","log(Amount+1)"]):
        vl=df.loc[df.Class==0,"Amount"]; vf=df.loc[df.Class==1,"Amount"]
        if log_s: vl,vf=np.log1p(vl),np.log1p(vf)
        bins=np.linspace(min(vl.min(),vf.min()),max(vl.max(),vf.max()),60)
        ax.hist(vl,bins=bins,color=GREEN,alpha=0.6,label="Legit",density=True)
        ax.hist(vf,bins=bins,color=RED,alpha=0.8,label="Fraud",density=True)
        ax.set(xlabel=ttl,ylabel="Density",title=f"{ttl} by Class")
        ax.legend(facecolor="#0D1018",edgecolor=DIM); ax.grid(True,alpha=0.3)
    fig.suptitle("Transaction Amount Distribution",color=AMBER,fontsize=13,fontweight="bold")
    _save(fig,"amount_distribution.png")


# 9. SAVE
def save_artifacts(clf, iso, scaler, explainer,
                   feat_cols_base, feat_cols_all,
                   metrics, fi, threshold, amount_mean, amount_std):
    mc = {k:v for k,v in metrics.items() if k not in ("y_prob","y_test")}
    joblib.dump(clf,            f"{MODEL_DIR}/xgb_model.joblib")
    joblib.dump(iso,            f"{MODEL_DIR}/iso_forest.joblib")
    joblib.dump(scaler,         f"{MODEL_DIR}/scaler.joblib")
    joblib.dump(feat_cols_base, f"{MODEL_DIR}/feature_cols_base.joblib")
    joblib.dump(feat_cols_all,  f"{MODEL_DIR}/feature_cols_all.joblib")
    if explainer: joblib.dump(explainer, f"{MODEL_DIR}/shap_explainer.joblib")
    summary = {
        "model":"XGBoost + Isolation Forest Hybrid",
        "novelty":[
            "Isolation Forest anomaly score as supervised meta-feature",
            "V-feature interaction terms (V14xV4, V11xV12)",
            "SMOTE k=5 minority oversampling for class balance",
            "XGBoost scale_pos_weight cost-sensitive learning",
            "Precision-Recall optimal threshold tuning (maximises F1)",
            "Isotonic probability calibration for reliable scores",
            "SHAP TreeExplainer per-prediction explainability",
            "Dataset stats saved for exact inference-time z-scoring",
        ],
        "metrics":mc, "feature_importance":fi, "train_features":feat_cols_all,
        "threshold":round(threshold,4),
        "dataset_stats":{"amount_mean":round(float(amount_mean),4),
                         "amount_std": round(float(amount_std),4)},
    }
    with open(f"{MODEL_DIR}/metrics.json","w") as f: json.dump(summary,f,indent=2)
    print(f"\n✓ Artifacts → {MODEL_DIR}/")
    for fn in sorted(os.listdir(MODEL_DIR)):
        print(f"  {fn:40s} {os.path.getsize(f'{MODEL_DIR}/{fn}')//1024:>6} KB")
    print(f"\n✓ Plots → {PLOTS_DIR}/")
    for fn in sorted(os.listdir(PLOTS_DIR)):
        print(f"  {fn:40s} {os.path.getsize(f'{PLOTS_DIR}/{fn}')//1024:>6} KB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    print("="*60)
    print("  ML FRAUD DETECTION PROJECT — TRAINING PIPELINE")
    print("  XGBoost + IsoForest | SMOTE | Calibration | SHAP")
    print("="*60)

    df = load_data(args.data)
    plot_amount_distribution(df)
    (X_train,X_test,y_train,y_test,
     scaler,feat_cols_base,
     amount_mean,amount_std) = preprocess(df)
    X_train,X_test,iso = add_isolation_score(X_train,X_test,feat_cols_base)
    feat_cols_all = list(X_train.columns)
    cal_clf, raw_clf = train_model(X_train, y_train)

    X_val = X_train.sample(min(20000,len(X_train)),random_state=99)
    y_val = pd.Series(y_train,index=X_train.index)[X_val.index]
    threshold = find_best_threshold(y_val, cal_clf.predict_proba(X_val)[:,1])

    metrics,y_prob,y_pred = evaluate(cal_clf, X_test, y_test, threshold)
    explainer,fi,shap_vals,shap_sample = build_shap(raw_clf, X_train, feat_cols_all)

    print(f"\n[{_t()}][PLOTS] Generating...")
    plot_confusion_matrix(y_test,y_pred)
    plot_roc(y_test,y_prob,metrics["roc_auc"])
    plot_pr(y_test,y_prob,metrics["avg_prec"])
    plot_classification_report(y_test,y_pred)
    plot_feature_importance(fi)
    plot_shap_summary(shap_vals,shap_sample,feat_cols_all)
    plot_calibration(y_test,y_prob)
    plot_score_distribution(y_test,y_prob)
    plot_threshold_sweep(y_test,y_prob)

    save_artifacts(cal_clf,iso,scaler,explainer,
                   feat_cols_base,feat_cols_all,
                   metrics,fi,threshold,amount_mean,amount_std)

    total=time.time()-_T0
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE  ({int(total//60)}m {int(total%60)}s)")
    print(f"  AUC-ROC : {metrics['roc_auc']}    AUC-PR : {metrics['avg_prec']}")
    print(f"  F1      : {metrics['f1']}    Recall : {metrics['recall']}")
    print(f"  MCC     : {metrics['mcc']}    Threshold: {threshold:.4f}")
    print(f"  Run: python app.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
