"""
train.py
========
Trains the AI Return Risk Manager models.

Methodology
-----------
1. Load transactions.csv
2. Drop POST-RETURN fields (leakage prevention) - see data/README.md
3. Split: 80% train_full / 20% HELD-OUT TEST (stratified, random_state=42)
4. From train_full, carve out a validation split (75/25 of train_full,
   i.e. ~60/20/20 overall) used ONLY for model selection and threshold
   tuning. The held-out test set is NEVER touched until final evaluation.
5. Train Logistic Regression (baseline) and Random Forest (main model)
   inside sklearn Pipelines with a ColumnTransformer.
6. Select the best model using VALIDATION precision/recall/F1 (not
   accuracy, since the data is imbalanced).
7. Evaluate the chosen model EXACTLY ONCE on the held-out test set.
8. Save the fitted pipeline + all metrics/artifacts to disk.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "transactions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ml", "saved_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "ml", "metrics.json")
FEATURE_META_PATH = os.path.join(BASE_DIR, "ml", "feature_meta.json")

# ----------------------------------------------------------------------
# FEATURE COLUMNS - INPUTS AVAILABLE AT DECISION TIME ONLY.
# POST-RETURN FIELDS (days_to_return, delivery_to_return_days,
# return_reason) ARE DELIBERATELY EXCLUDED. See data/README.md.
# ----------------------------------------------------------------------
RAW_NUMERIC_FEATURES = [
    "order_amount", "discount_percent", "delivery_distance_km",
    "customer_age", "customer_account_age_days", "previous_orders",
    "previous_returns", "return_rate", "returns_last_30_days",
    "refunds_last_90_days", "chargebacks_last_90_days",
    "coupon_usage_count", "high_value_order", "location_risk_score",
    "device_change_count", "orders_last_30_days", "previous_fraud_flag",
]
CATEGORICAL_FEATURES = ["product_category", "payment_method", "customer_risk_history"]

# Engineered features - derived ONLY from decision-time-available fields
# (no post-return information), added to help the model capture
# non-linear risk combinations legitimately, without leakage.
ENGINEERED_FEATURES = [
    "recent_return_intensity",   # returns_last_30_days + 0.5*refunds_last_90_days
    "discount_value",            # order_amount * discount_percent / 100
    "risk_flag_sum",             # chargebacks_last_90_days + previous_fraud_flag
    "new_account_high_value",    # 1 if account<30d AND high_value_order
    "order_velocity_risk",       # orders_last_30_days * device_change_count
]

NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
FEATURE_COLUMNS = RAW_NUMERIC_FEATURES + CATEGORICAL_FEATURES  # columns read from CSV


def add_engineered_features(df):
    df = df.copy()
    df["recent_return_intensity"] = df["returns_last_30_days"] + 0.5 * df["refunds_last_90_days"]
    df["discount_value"] = df["order_amount"] * df["discount_percent"] / 100
    df["risk_flag_sum"] = df["chargebacks_last_90_days"] + df["previous_fraud_flag"]
    df["new_account_high_value"] = (
        (df["customer_account_age_days"] < 30) & (df["high_value_order"] == 1)
    ).astype(int)
    df["order_velocity_risk"] = df["orders_last_30_days"] * df["device_change_count"]
    return df
TARGET = "target_return_abuse"
POST_RETURN_FIELDS = ["days_to_return", "delivery_to_return_days", "return_reason"]
ID_COLUMNS = ["transaction_id", "customer_id"]


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def evaluate(model, X, y, label=""):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }
    if label:
        print(f"\n--- {label} (threshold=0.5) ---")
        for k, v in metrics.items():
            print(f"  {k:10s}: {v:.4f}")
    return metrics, proba


def main():
    df = pd.read_csv(DATA_PATH)
    df = add_engineered_features(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()

    # 80% train_full / 20% held-out test, stratified, seed=42
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    # From train_full: 75/25 -> ~60% train / 20% validation overall
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, stratify=y_train_full,
        random_state=RANDOM_SEED
    )

    print(f"Train: {len(X_train)}  Validation: {len(X_val)}  "
          f"HELD-OUT TEST: {len(X_test)}")
    print(f"Train positive rate: {y_train.mean():.4f}")
    print(f"Validation positive rate: {y_val.mean():.4f}")
    print(f"Test positive rate: {y_test.mean():.4f}")

    preprocessor = build_preprocessor()

    # ---- Baseline: Logistic Regression ----
    logreg = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                    random_state=RANDOM_SEED)),
    ])
    logreg.fit(X_train, y_train)
    logreg_val_metrics, _ = evaluate(logreg, X_val, y_val, "Logistic Regression - VALIDATION")

    # ---- Main model: Random Forest ----
    rf = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=600, max_depth=12, min_samples_leaf=3,
            min_samples_split=6, max_features="sqrt",
            class_weight="balanced", random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])
    rf.fit(X_train, y_train)
    rf_val_metrics, _ = evaluate(rf, X_val, y_val, "Random Forest - VALIDATION")

    # ---- Model selection on VALIDATION F1 (not accuracy; imbalanced data) ----
    if rf_val_metrics["f1"] >= logreg_val_metrics["f1"]:
        chosen_name, chosen_model = "RandomForest", rf
    else:
        chosen_name, chosen_model = "LogisticRegression", logreg
    print(f"\n>>> Selected model based on validation F1: {chosen_name}")

    # Refit chosen model on train+val (train_full) before final test evaluation,
    # a standard practice once model selection is finalized and threshold-free
    # architecture decisions are locked in. This still never touches the test set.
    chosen_model.fit(X_train_full, y_train_full)

    # ---------------- HELD-OUT TEST PERFORMANCE ----------------
    test_metrics, test_proba = evaluate(chosen_model, X_test, y_test,
                                         "HELD-OUT TEST PERFORMANCE")
    pred_05 = (test_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred_05).tolist()  # [[TN, FP],[FN, TP]]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0

    # ---------------- THRESHOLD ANALYSIS (computed on VALIDATION set) ----------------
    FP_COST = 150
    FN_COST = 1200
    val_proba = chosen_model.predict_proba(X_val)[:, 1]
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    threshold_rows = []
    for t in thresholds:
        pred_t = (val_proba >= t).astype(int)
        cm_t = confusion_matrix(y_val, pred_t)
        tn_t, fp_t, fn_t, tp_t = cm_t[0][0], cm_t[0][1], cm_t[1][0], cm_t[1][1]
        row = {
            "threshold": t,
            "precision": precision_score(y_val, pred_t, zero_division=0),
            "recall": recall_score(y_val, pred_t, zero_division=0),
            "f1": f1_score(y_val, pred_t, zero_division=0),
            "false_positives": int(fp_t),
            "false_negatives": int(fn_t),
            "fp_cost": int(fp_t) * FP_COST,
            "fn_cost": int(fn_t) * FN_COST,
            "total_cost": int(fp_t) * FP_COST + int(fn_t) * FN_COST,
        }
        threshold_rows.append(row)

    best_threshold_row = min(threshold_rows, key=lambda r: r["total_cost"])
    chosen_threshold = best_threshold_row["threshold"]
    print("\n--- THRESHOLD ANALYSIS (on VALIDATION set, cost-based) ---")
    for r in threshold_rows:
        print(f"  t={r['threshold']:.2f}  P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f}  FP={r['false_positives']:3d} FN={r['false_negatives']:3d}  "
              f"TotalCost=Rs.{r['total_cost']}")
    print(f"\n>>> Selected threshold (min validation cost): {chosen_threshold}")

    # ---------------- Re-evaluate HELD-OUT TEST at chosen threshold ----------------
    pred_chosen = (test_proba >= chosen_threshold).astype(int)
    cm_chosen = confusion_matrix(y_test, pred_chosen)
    tn_c, fp_c, fn_c, tp_c = cm_chosen[0][0], cm_chosen[0][1], cm_chosen[1][0], cm_chosen[1][1]
    test_metrics_chosen_threshold = {
        "threshold": chosen_threshold,
        "precision": precision_score(y_test, pred_chosen, zero_division=0),
        "recall": recall_score(y_test, pred_chosen, zero_division=0),
        "f1": f1_score(y_test, pred_chosen, zero_division=0),
        "accuracy": accuracy_score(y_test, pred_chosen),
        "confusion_matrix": {"tn": int(tn_c), "fp": int(fp_c), "fn": int(fn_c), "tp": int(tp_c)},
        "false_positive_rate": fp_c / (fp_c + tn_c) if (fp_c + tn_c) else 0,
        "false_negative_rate": fn_c / (fn_c + tp_c) if (fn_c + tp_c) else 0,
        "fp_cost": int(fp_c) * FP_COST,
        "fn_cost": int(fn_c) * FN_COST,
        "total_cost": int(fp_c) * FP_COST + int(fn_c) * FN_COST,
    }
    print(f"\n--- HELD-OUT TEST @ chosen threshold {chosen_threshold} ---")
    for k, v in test_metrics_chosen_threshold.items():
        print(f"  {k}: {v}")

    # "Without model" baseline: approve everyone (do nothing) -> every actual
    # abuse case becomes an undetected loss (FN_COST each); no FP cost since
    # nothing is ever flagged.
    n_actual_positive_test = int(y_test.sum())
    without_model_cost = n_actual_positive_test * FN_COST
    with_model_cost = test_metrics_chosen_threshold["total_cost"]
    estimated_loss_prevented = without_model_cost - with_model_cost

    # ---------------- Feature importance (for explainability) ----------------
    feature_names = (
        NUMERIC_FEATURES +
        list(chosen_model.named_steps["prep"]
             .named_transformers_["cat"].named_steps["onehot"]
             .get_feature_names_out(CATEGORICAL_FEATURES))
    )
    clf = chosen_model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        importances = np.abs(clf.coef_[0])
    importance_map = dict(sorted(zip(feature_names, importances.tolist()),
                                  key=lambda x: -x[1]))

    # ---------------- Save artifacts ----------------
    joblib.dump({
        "model": chosen_model,
        "model_name": chosen_name,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "threshold": chosen_threshold,
        "feature_importance": importance_map,
    }, MODEL_PATH)

    metrics_out = {
        "model_name": chosen_name,
        "random_seed": RANDOM_SEED,
        "data_split": {
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "train_positive_rate": float(y_train.mean()),
            "validation_positive_rate": float(y_val.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
        "validation_metrics": {
            "logistic_regression": logreg_val_metrics,
            "random_forest": rf_val_metrics,
        },
        "held_out_test_performance_default_threshold_0.5": {
            **test_metrics,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
        },
        "threshold_analysis_on_validation": threshold_rows,
        "chosen_threshold": chosen_threshold,
        "held_out_test_performance_chosen_threshold": test_metrics_chosen_threshold,
        "financial_model": {
            "fp_cost_per_case_inr": FP_COST,
            "fn_cost_per_case_inr": FN_COST,
            "without_model_total_cost_inr": without_model_cost,
            "with_model_total_cost_inr": with_model_cost,
            "estimated_loss_prevented_inr": estimated_loss_prevented,
            "note": "Estimates are derived entirely from the synthetic held-out "
                    "test set and assumed per-case costs. They are illustrative "
                    "for the hackathon demo and do NOT represent real-world "
                    "financial savings.",
        },
        "feature_importance": importance_map,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)

    with open(FEATURE_META_PATH, "w") as f:
        json.dump({
            "feature_columns": FEATURE_COLUMNS,
            "post_return_fields_excluded": POST_RETURN_FIELDS,
        }, f, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
