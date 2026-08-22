"""
score_all.py
============
Scores every transaction in transactions.csv with the trained model, for
the Transaction Explorer / dashboard / /transactions API endpoints.

IMPORTANT: This is DEMO SCORING ONLY (so the dashboard has something to
browse). It is NOT how the model's honest performance is measured -
that number comes exclusively from ml/metrics.json's
`held_out_test_performance_*` sections, computed once on the untouched
20% test split in ml/train.py. This file re-scores the whole dataset
(including rows the model was trained on) purely so the Transaction
Explorer table has realistic-looking rows to browse and filter.
"""
import json
import os
import sys

import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "ml"))
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

from train import add_engineered_features  # noqa: E402
from risk_engine import score_transaction  # noqa: E402

DATA_PATH = os.path.join(BASE_DIR, "data", "transactions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ml", "saved_model.pkl")
OUT_JSON = os.path.join(BASE_DIR, "ml", "scored_transactions.json")


def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    numeric = bundle["numeric_features"]
    categorical = bundle["categorical_features"]
    importance = bundle["feature_importance"]

    df = pd.read_csv(DATA_PATH)
    eng = add_engineered_features(df)
    proba = model.predict_proba(eng[numeric + categorical])[:, 1]

    rows = []
    for i, p in enumerate(proba):
        row_dict = df.iloc[i].to_dict()
        scored = score_transaction(float(p), row_dict, importance)
        rows.append({
            "transaction_id": row_dict["transaction_id"],
            "customer_id": row_dict["customer_id"],
            "order_amount": row_dict["order_amount"],
            "product_category": row_dict["product_category"],
            "payment_method": row_dict["payment_method"],
            "previous_returns": int(row_dict["previous_returns"]),
            "return_rate": row_dict["return_rate"],
            "days_to_return": row_dict["days_to_return"] if pd.notna(row_dict["days_to_return"]) else None,
            "actual_label": int(row_dict["target_return_abuse"]),
            "risk_score": scored["risk_score"],
            "risk_level": scored["risk_level"],
            "recommended_action": scored["recommended_action"],
            "reasons": scored["reasons"],
        })

    with open(OUT_JSON, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Scored {len(rows)} transactions -> {OUT_JSON}")

    from collections import Counter
    dist = Counter(r["risk_level"] for r in rows)
    print("Risk distribution:", dict(dist))


if __name__ == "__main__":
    main()
