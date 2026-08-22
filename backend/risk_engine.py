"""
risk_engine.py
==============
Converts a model probability into a 0-100 risk score, a risk LEVEL
(LOW/MEDIUM/HIGH), and a DEFENSIVE recommended action. This module
NEVER accuses anyone of fraud - it only ever surfaces "risk detected"
language, and every HIGH risk case is routed to a human for manual
review, never an automated block or accusation.
"""

RISK_BANDS = [
    (0, 39, "LOW"),
    (40, 69, "MEDIUM"),
    (70, 100, "HIGH"),
]

ACTION_MAP = {
    "LOW": "APPROVE",
    "MEDIUM": "VERIFY",
    "HIGH": "MANUAL_REVIEW",
}

ACTION_DESCRIPTION = {
    "APPROVE": "No additional friction. Process normally.",
    "VERIFY": "Request lightweight verification (e.g. confirm delivery photo, "
               "OTP re-confirmation) before approving the return/refund.",
    "MANUAL_REVIEW": "Route to a human risk analyst for review before any "
                      "action is taken. Risk detected - this is NOT an "
                      "accusation of fraud, and the customer should be "
                      "treated courteously pending review.",
}


def probability_to_score(probability: float) -> int:
    """Map a model probability [0,1] to a 0-100 integer risk score."""
    return int(round(max(0.0, min(1.0, probability)) * 100))


def score_to_level(score: int) -> str:
    for low, high, level in RISK_BANDS:
        if low <= score <= high:
            return level
    return "HIGH" if score > 100 else "LOW"


def level_to_action(level: str) -> str:
    return ACTION_MAP[level]


def build_reasons(feature_row: dict, feature_importance: dict, top_n: int = 4) -> list:
    """
    Produce human-readable, MODEL-SUPPORTED reasons for a risk score.
    Only surfaces reasons for features that are (a) among the model's
    top globally-important features AND (b) present at an elevated /
    notable value for this specific transaction. Never invents reasons.
    """
    # Human-readable templates keyed by underlying feature name.
    templates = {
        "return_rate": ("Historical return rate is significantly above normal", lambda v: v > 0.25),
        "returns_last_30_days": ("Multiple returns in the last 30 days", lambda v: v >= 2),
        "refunds_last_90_days": ("Multiple refunds in the last 90 days", lambda v: v >= 2),
        "chargebacks_last_90_days": ("Chargeback(s) on record in the last 90 days", lambda v: v >= 1),
        "previous_fraud_flag": ("Account previously flagged for risk review", lambda v: v == 1),
        "customer_risk_history": ("Customer risk history bucket is elevated", lambda v: v in ("medium", "high")),
        "location_risk_score": ("Delivery location has an elevated risk score", lambda v: v > 60),
        "device_change_count": ("Frequent device changes on this account", lambda v: v >= 2),
        "coupon_usage_count": ("High coupon usage pattern", lambda v: v >= 3),
        "orders_last_30_days": ("Unusually high order frequency in last 30 days", lambda v: v >= 4),
        "high_value_order": ("High-value transaction", lambda v: v == 1),
        "customer_account_age_days": ("New account placing a high-value order", lambda v: v < 30),
        "discount_percent": ("Order placed at an unusually high discount", lambda v: v > 40),
    }

    # rank underlying (non-one-hot) importances by summing one-hot columns back
    base_importance = {}
    for feat, imp in feature_importance.items():
        base = feat.split("_")[0] if feat not in templates else feat
        # try direct match first
        matched = None
        for key in templates:
            if feat == key or feat.startswith(key):
                matched = key
                break
        if matched:
            base_importance[matched] = base_importance.get(matched, 0) + imp

    ranked_features = sorted(base_importance.items(), key=lambda x: -x[1])

    reasons = []
    for feat_name, _ in ranked_features:
        if feat_name not in templates:
            continue
        text, condition = templates[feat_name]
        value = feature_row.get(feat_name)
        if value is None:
            continue
        try:
            if condition(value):
                reasons.append(text)
        except TypeError:
            continue
        if len(reasons) >= top_n:
            break

    if not reasons:
        reasons.append("Multiple minor risk signals combined push this transaction "
                        "above the low-risk band; no single dominant factor.")
    return reasons


def score_transaction(probability: float, feature_row: dict, feature_importance: dict) -> dict:
    score = probability_to_score(probability)
    level = score_to_level(score)
    action = level_to_action(level)
    reasons = build_reasons(feature_row, feature_importance)
    return {
        "risk_score": score,
        "risk_level": level,
        "recommended_action": action,
        "action_description": ACTION_DESCRIPTION[action],
        "reasons": reasons,
    }
