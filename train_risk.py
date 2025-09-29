# features/train_risk.py (updated with consistent metric keys)
from pathlib import Path
import sqlite3, importlib.util, joblib, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
DB   = ROOT / "db" / "retailrocket_mini.db"
MODEL_DIR = ROOT / "models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "risk_model.pkl"

def load_features_df():
    try:
        conn = sqlite3.connect(DB)
        df = pd.read_sql("SELECT * FROM customer_features", conn)
        conn.close()
        if not df.empty:
            return df
    except Exception:
        pass
    spec = importlib.util.spec_from_file_location("bf", str(ROOT / "features" / "build_features.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.build_features(DB)

from sklearn.metrics import accuracy_score, confusion_matrix

def _eval_probs(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # predict hard labels
    y_pred = (y_prob >= 0.5).astype(int)

    acc   = float(accuracy_score(y_true, y_pred))
    auc   = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    prauc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    brier = float(brier_score_loss(y_true, y_prob))

    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    return {"Accuracy": acc, "ROC-AUC": auc, "PR-AUC": prauc, "Brier": brier}


def train_and_save():
    df = load_features_df()
    if df.empty:
        raise RuntimeError("No features available. Build features first.")

    y = df["churn_flag"].astype(int)
    counts = y.value_counts().to_dict()
    print("Label counts:", counts)
    if len(counts) < 2:
        raise RuntimeError(
            "Your churn_flag has only one class.\n"
            "Fix by either enlarging the mini DB window or lowering inactivity threshold."
        )

    Xcols = [
        "days_since_last_event","views","carts","purchases","total_events",
        "cart_rate","purchase_rate","avg_gap_days","orders","total_spend",
        "avg_order_value","distinct_categories_viewed","active_days"
    ]
    for c in Xcols:
        if c not in df.columns: df[c] = 0.0
    X = df[Xcols].replace([np.inf,-np.inf], np.nan).fillna(0)

    # Always use stratified split for balance
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=y
    )

    X_tr, X_te = X.loc[train_idx], X.loc[test_idx]
    y_tr, y_te = y.loc[train_idx], y.loc[test_idx]

    candidates = {}

    # 1) Logistic Regression (scaled)
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    pipe_lr.fit(X_tr, y_tr)
    prob_lr = pipe_lr.predict_proba(X_te)[:,1]
    candidates["LogisticRegression"] = {"est": pipe_lr, "metrics": _eval_probs(y_te, prob_lr)}

    # 2) Random Forest (tree model, no scaling)
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    prob_rf = rf.predict_proba(X_te)[:,1]
    candidates["RandomForest"] = {"est": rf, "metrics": _eval_probs(y_te, prob_rf)}

    # choose best by PR-AUC → ROC-AUC → Brier
    def _score_key(name):
        m = candidates[name]["metrics"]
        pr = m.get("PR-AUC", -1)
        au = m.get("ROC-AUC", -1)
        br = -m.get("Brier", 0)  # smaller Brier is better
        return (pr, au, br)

    best_name = sorted(candidates.keys(), key=_score_key, reverse=True)[0]
    best_est  = candidates[best_name]["est"]
    chosen_metrics = candidates[best_name]["metrics"]

    # Build payload with all models
    payload = {
        "best_name": best_name,
        "pipeline": best_est,
        "Xcols": Xcols,
        "models": {
            model_name: {
                "pipeline": cand["est"],
                "metrics": cand["metrics"]
            }
            for model_name, cand in candidates.items()
        },
        "metrics_by_model": {
            model_name: cand["metrics"] for model_name, cand in candidates.items()
        },
        "chosen_metrics": chosen_metrics
    }

    joblib.dump(payload, MODEL_PATH)
    print(f"✅ Saved model comparison payload with best = {best_name} to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save()
