import json
import os
import sys

import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "ml"))
sys.path.insert(0, os.path.join(BASE_DIR, "data"))

from generate_data import generate  # noqa: E402
from train import (FEATURE_COLUMNS, POST_RETURN_FIELDS,  # noqa: E402
                    add_engineered_features)


def test_data_generation_row_count():
    df = generate(n_rows=2000, seed=1)
    assert len(df) == 2000


def test_data_generation_reproducible():
    df1 = generate(n_rows=500, seed=42)
    df2 = generate(n_rows=500, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_positive_rate_in_realistic_band():
    df = generate(n_rows=6000, seed=42)
    rate = df["target_return_abuse"].mean()
    assert 0.03 <= rate <= 0.15, f"positive rate {rate} outside realistic imbalance band"


def test_label_not_trivially_dependent_on_single_feature():
    df = generate(n_rows=4000, seed=7)
    # No single feature should perfectly separate the classes
    for col in ["previous_fraud_flag", "chargebacks_last_90_days", "high_value_order"]:
        corr = df[col].astype(float).corr(df["target_return_abuse"].astype(float))
        assert abs(corr) < 0.6, f"{col} is suspiciously predictive alone (corr={corr:.2f})"


def test_no_leakage_columns():
    """POST_RETURN_FIELDS must never appear in the model's feature columns."""
    for field in POST_RETURN_FIELDS:
        assert field not in FEATURE_COLUMNS


def test_engineered_features_do_not_use_post_return_fields():
    df = generate(n_rows=200, seed=3)
    eng = add_engineered_features(df)
    # engineered columns should be derivable even if post-return fields are dropped
    df_no_post = df.drop(columns=POST_RETURN_FIELDS)
    eng2 = add_engineered_features(df_no_post)
    for col in ["recent_return_intensity", "discount_value", "risk_flag_sum",
                "new_account_high_value", "order_velocity_risk"]:
        pd.testing.assert_series_equal(eng[col], eng2[col])


@pytest.fixture(scope="module")
def metrics():
    metrics_path = os.path.join(BASE_DIR, "ml", "metrics.json")
    if not os.path.exists(metrics_path):
        pytest.skip("metrics.json not found - run ml/train.py first")
    with open(metrics_path) as f:
        return json.load(f)


def test_metrics_file_has_required_sections(metrics):
    for key in ["held_out_test_performance_default_threshold_0.5",
                "held_out_test_performance_chosen_threshold",
                "threshold_analysis_on_validation",
                "financial_model", "data_split"]:
        assert key in metrics


def test_held_out_test_metrics_are_plausible(metrics):
    test_perf = metrics["held_out_test_performance_default_threshold_0.5"]
    for key in ["precision", "recall", "f1", "accuracy", "roc_auc"]:
        assert 0.0 <= test_perf[key] <= 1.0
    # sanity: a random classifier would get ~0.5 AUC; our model should beat it
    assert test_perf["roc_auc"] > 0.55


def test_financial_model_costs_are_documented(metrics):
    fin = metrics["financial_model"]
    assert fin["fp_cost_per_case_inr"] > 0
    assert fin["fn_cost_per_case_inr"] > 0
    assert "note" in fin and "synthetic" in fin["note"].lower()


def test_saved_model_loads():
    model_path = os.path.join(BASE_DIR, "ml", "saved_model.pkl")
    if not os.path.exists(model_path):
        pytest.skip("saved_model.pkl not found - run ml/train.py first")
    import joblib
    bundle = joblib.load(model_path)
    assert "model" in bundle
    assert "threshold" in bundle
