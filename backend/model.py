"""
model.py
========
Loads the trained pipeline (saved by ml/train.py) and exposes a single
`predict_one(record)` / `predict_many(records)` interface used by the
Flask API and dashboard.
"""
import json
import os

import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "saved_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "ml", "metrics.json")

_bundle = None
_metrics = None


def load():
    global _bundle, _metrics
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    if _metrics is None:
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)
    return _bundle, _metrics


def get_metrics():
    _, metrics = load()
    return metrics


def get_threshold():
    bundle, _ = load()
    return bundle["threshold"]


def get_feature_importance():
    bundle, _ = load()
    return bundle["feature_importance"]


def predict_proba_df(engineered_df: pd.DataFrame):
    """engineered_df must already have engineered features added (see preprocessing.py)."""
    bundle, _ = load()
    model = bundle["model"]
    numeric = bundle["numeric_features"]
    categorical = bundle["categorical_features"]
    cols = numeric + categorical
    return model.predict_proba(engineered_df[cols])[:, 1]
