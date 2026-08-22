"""
preprocessing.py
=================
Shared helpers so the API applies the EXACT SAME feature engineering
used during training (see ml/train.py::add_engineered_features).
Kept as a thin wrapper to avoid duplicating logic; imports directly
from ml/train.py so there is a single source of truth.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "ml"))

from train import (RAW_NUMERIC_FEATURES, CATEGORICAL_FEATURES,  # noqa: E402
                    FEATURE_COLUMNS, add_engineered_features)

REQUIRED_INPUT_FIELDS = FEATURE_COLUMNS  # raw, decision-time-available fields only


def prepare_row(record: dict):
    """Validate required fields are present and return an engineered-feature dict."""
    missing = [f for f in REQUIRED_INPUT_FIELDS if f not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    import pandas as pd
    df = pd.DataFrame([record])
    df = add_engineered_features(df)
    return df
