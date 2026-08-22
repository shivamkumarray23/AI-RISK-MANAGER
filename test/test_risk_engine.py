import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

from risk_engine import (build_reasons, level_to_action,  # noqa: E402
                          probability_to_score, score_to_level,
                          score_transaction)


def test_probability_to_score_bounds():
    assert probability_to_score(0.0) == 0
    assert probability_to_score(1.0) == 100
    assert probability_to_score(0.5) == 50


def test_probability_to_score_clips_out_of_range():
    assert probability_to_score(-0.2) == 0
    assert probability_to_score(1.5) == 100


def test_score_to_level_bands():
    assert score_to_level(0) == "LOW"
    assert score_to_level(39) == "LOW"
    assert score_to_level(40) == "MEDIUM"
    assert score_to_level(69) == "MEDIUM"
    assert score_to_level(70) == "HIGH"
    assert score_to_level(100) == "HIGH"


def test_level_to_action_mapping():
    assert level_to_action("LOW") == "APPROVE"
    assert level_to_action("MEDIUM") == "VERIFY"
    assert level_to_action("HIGH") == "MANUAL_REVIEW"


def test_high_risk_never_auto_blocks_only_manual_review():
    # Defensive safeguard: HIGH must always route to a human, never an
    # automated block/decline action.
    action = level_to_action("HIGH")
    assert action == "MANUAL_REVIEW"
    assert action not in ("BLOCK", "DECLINE", "AUTO_REJECT")


def test_build_reasons_uses_only_present_signals():
    feature_row = {"return_rate": 0.5, "previous_fraud_flag": 0, "chargebacks_last_90_days": 0}
    importance = {"return_rate": 0.9, "previous_fraud_flag": 0.05}
    reasons = build_reasons(feature_row, importance)
    assert any("return rate" in r.lower() for r in reasons)
    assert not any("fraud" in r.lower() for r in reasons)  # flag was 0, shouldn't be cited


def test_score_transaction_full_output_shape():
    feature_row = {"return_rate": 0.8, "chargebacks_last_90_days": 1, "previous_fraud_flag": 1}
    importance = {"return_rate": 0.5, "chargebacks_last_90_days": 0.3, "previous_fraud_flag": 0.2}
    result = score_transaction(0.85, feature_row, importance)
    assert set(result.keys()) == {"risk_score", "risk_level", "recommended_action",
                                   "action_description", "reasons"}
    assert result["risk_level"] == "HIGH"
    assert result["recommended_action"] == "MANUAL_REVIEW"
    assert len(result["reasons"]) > 0


def test_language_never_accuses_customer_directly():
    """No hard-coded action description should declare the customer a fraudster."""
    from risk_engine import ACTION_DESCRIPTION
    for desc in ACTION_DESCRIPTION.values():
        assert "fraudster" not in desc.lower()
        assert "is committing fraud" not in desc.lower()
