"""
generate_data.py
=================
Generates a realistic SYNTHETIC e-commerce transaction dataset for the
AI Return Risk Manager.

-----------------------------------------------------------------------
HOW THE LABEL (target_return_abuse) IS GENERATED - FULL DOCUMENTATION
-----------------------------------------------------------------------
The label is NOT a copy of any single feature. It is generated from a
weighted combination ("latent risk score") of MANY behavioral signals,
Gaussian noise is added, the result is passed through a logistic
function, and the final label is a Bernoulli DRAW from that probability
- so two customers with identical risk factors can still get different
outcomes, exactly like real-world abuse behavior.

Risk factors that push the latent risk score UP (with weights):
  - high historical return_rate                              (w = 2.6)
  - many returns_last_30_days                                (w = 1.8)
  - many refunds_last_90_days                                (w = 1.4)
  - any chargebacks_last_90_days                              (w = 2.2)
  - previous_fraud_flag = 1                                   (w = 2.0)
  - high customer_risk_history bucket                         (w = 1.6)
  - high location_risk_score                                  (w = 1.1)
  - high device_change_count (account sharing/farming)        (w = 1.0)
  - high order_amount combined with high discount             (w = 0.9)
  - high coupon_usage_count (coupon abuse pattern)             (w = 0.8)
  - many orders_last_30_days (order farming)                  (w = 0.7)
  - new account (low customer_account_age_days) + high value  (w = 1.0)
  - very short delivery_to_return_days (fast "return after use")  (up to 1.5)
  - very short days_to_return (near-instant returns)              (up to 1.2)
  - return_reason in {"changed_mind_repeated","not_as_described_repeated"} (1.3)

Random Gaussian noise (sigma=1.3) is added to the latent score before
the logistic transform. This guarantees:
  1) The label is NOT deterministic from any single feature.
  2) Some legitimate customers will still look "risky" (irreducible
     false-positive potential), and some abusive customers will look
     "normal" (irreducible false-negative potential) - matching
     real-world fraud detection, where perfect separability never exists.

We then rank-transform the resulting probabilities and rescale so the
population base rate lands in a realistic 5-10% band (randomly chosen
per generation run), and draw target_return_abuse ~ Bernoulli(p) per row.
This produces a realistically IMBALANCED, NOISY, NON-TRIVIAL label.

-----------------------------------------------------------------------
LEAKAGE NOTE (see README for full discussion)
-----------------------------------------------------------------------
Three fields - return_reason, days_to_return, delivery_to_return_days -
describe what happens DURING/AFTER a return. They are legitimately used
here to help construct the ground-truth LABEL (the true outcome is only
fully knowable after a return completes), but they are EXCLUDED from
the model's INPUT FEATURES at inference/training time, because they
would not be available when the merchant actually needs a risk score
(at checkout, at shipping, or at the moment a return is first
requested). ml/train.py's FEATURE_COLUMNS list never includes these
three fields, and tests/test_model.py asserts they are absent from the
trained pipeline's input schema.
-----------------------------------------------------------------------
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ROWS = 6000

CATEGORIES = ["Electronics", "Fashion", "Home", "Beauty", "Sports",
              "Grocery", "Books", "Toys", "Footwear", "Mobile"]
PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "COD", "Net Banking", "Wallet"]
RETURN_REASONS = ["not_needed", "size_issue", "defective", "not_as_described",
                   "changed_mind", "not_as_described_repeated",
                   "changed_mind_repeated"]
CUSTOMER_RISK_BUCKETS = ["low", "medium", "high"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n_rows=N_ROWS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    transaction_id = [f"TX{100000+i}" for i in range(n_rows)]
    customer_id = [f"CUST{rng.integers(1, n_rows // 2):06d}" for _ in range(n_rows)]

    order_amount = np.round(rng.gamma(shape=2.2, scale=900, size=n_rows) + 150, 2)
    product_category = rng.choice(CATEGORIES, size=n_rows,
                                   p=[.16, .18, .12, .08, .07, .10, .06, .07, .09, .07])
    payment_method = rng.choice(PAYMENT_METHODS, size=n_rows,
                                 p=[.30, .22, .18, .15, .10, .05])
    discount_percent = np.clip(rng.normal(15, 12, n_rows), 0, 80).round(1)
    delivery_distance_km = np.clip(rng.exponential(12, n_rows), 0.5, 200).round(1)

    customer_age = np.clip(rng.normal(32, 10, n_rows), 18, 70).astype(int)
    customer_account_age_days = np.clip(rng.exponential(400, n_rows), 1, 3000).astype(int)

    previous_orders = np.clip(rng.poisson(8, n_rows), 0, 200)
    return_propensity = np.clip(rng.beta(1.3, 12, n_rows), 0, 1)
    previous_returns = np.minimum(
        rng.binomial(np.maximum(previous_orders, 1), return_propensity),
        previous_orders
    )
    return_rate = np.where(previous_orders > 0,
                            previous_returns / np.maximum(previous_orders, 1), 0).round(3)

    returns_last_30_days = np.minimum(rng.poisson(return_propensity * 3, n_rows),
                                       previous_returns)
    refunds_last_90_days = np.minimum(
        rng.poisson(return_propensity * 4, n_rows), previous_orders)
    chargebacks_last_90_days = rng.binomial(1, np.clip(return_propensity * 0.35, 0, 0.4), n_rows)

    has_return_mask = rng.random(n_rows) < 0.35
    days_to_return = np.where(
        has_return_mask, np.clip(rng.exponential(6, n_rows), 0, 30).round(1), np.nan
    )
    delivery_to_return_days = np.where(
        has_return_mask,
        np.clip(days_to_return - rng.exponential(1.5, n_rows), 0, 30).round(1),
        np.nan
    )

    coupon_usage_count = rng.poisson(1.4, n_rows)
    high_value_order = (order_amount > np.percentile(order_amount, 85)).astype(int)

    customer_risk_history = rng.choice(CUSTOMER_RISK_BUCKETS, size=n_rows, p=[.75, .18, .07])
    risk_hist_numeric = pd.Series(customer_risk_history).map({"low": 0, "medium": 1, "high": 2}).values

    location_risk_score = np.clip(rng.beta(2, 6, n_rows) * 100, 0, 100).round(1)
    device_change_count = rng.poisson(0.6, n_rows)
    orders_last_30_days = rng.poisson(1.6, n_rows)
    previous_fraud_flag = rng.binomial(1, 0.03, n_rows)

    return_reason = np.where(
        has_return_mask,
        rng.choice(RETURN_REASONS, size=n_rows),
        "no_return"
    )

    def z_norm(arr):
        arr = arr.astype(float)
        return (arr - arr.mean()) / (arr.std() + 1e-6)

    z = (
        2.6 * z_norm(return_rate)
        + 1.8 * z_norm(returns_last_30_days)
        + 1.4 * z_norm(refunds_last_90_days)
        + 2.2 * chargebacks_last_90_days
        + 2.0 * previous_fraud_flag
        + 1.6 * z_norm(risk_hist_numeric)
        + 1.1 * z_norm(location_risk_score)
        + 1.0 * z_norm(device_change_count)
        + 0.9 * z_norm(order_amount * discount_percent / 100)
        + 0.8 * z_norm(coupon_usage_count)
        + 0.7 * z_norm(orders_last_30_days)
        + 1.0 * ((customer_account_age_days < 30) & (high_value_order == 1)).astype(float)
    )

    fast_return_bonus = np.where(
        has_return_mask, np.clip(1.5 * (1 - np.nan_to_num(delivery_to_return_days) / 15), 0, 1.5), 0
    )
    fast_return_bonus2 = np.where(
        has_return_mask, np.clip(1.2 * (1 - np.nan_to_num(days_to_return) / 20), 0, 1.2), 0
    )
    suspicious_reason_bonus = np.where(
        np.isin(return_reason, ["changed_mind_repeated", "not_as_described_repeated"]), 1.3, 0
    )

    z = z + fast_return_bonus + fast_return_bonus2 + suspicious_reason_bonus

    noise = rng.normal(0, 1.3, n_rows)
    latent = z + noise
    p = sigmoid(latent - latent.mean())

    target_base_rate = rng.uniform(0.05, 0.10)
    rank_pct = pd.Series(p).rank(pct=True).values
    # E[rank_pct^2] ~= 1/3 for a uniform rank distribution, so multiplying by 3
    # makes the population mean of p_rescaled converge to target_base_rate.
    p_rescaled = np.clip(target_base_rate * (rank_pct ** 2) * 3 + 0.001, 0.001, 0.95)

    target_return_abuse = rng.binomial(1, p_rescaled)

    df = pd.DataFrame({
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "order_amount": order_amount,
        "product_category": product_category,
        "payment_method": payment_method,
        "discount_percent": discount_percent,
        "delivery_distance_km": delivery_distance_km,
        "customer_age": customer_age,
        "customer_account_age_days": customer_account_age_days,
        "previous_orders": previous_orders,
        "previous_returns": previous_returns,
        "return_rate": return_rate,
        "returns_last_30_days": returns_last_30_days,
        "refunds_last_90_days": refunds_last_90_days,
        "chargebacks_last_90_days": chargebacks_last_90_days,
        "days_to_return": days_to_return,
        "delivery_to_return_days": delivery_to_return_days,
        "coupon_usage_count": coupon_usage_count,
        "high_value_order": high_value_order,
        "customer_risk_history": customer_risk_history,
        "return_reason": return_reason,
        "location_risk_score": location_risk_score,
        "device_change_count": device_change_count,
        "orders_last_30_days": orders_last_30_days,
        "previous_fraud_flag": previous_fraud_flag,
        "target_return_abuse": target_return_abuse,
    })

    return df


if __name__ == "__main__":
    df = generate()
    out_path = "/home/claude/ai-risk-manager/data/transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print("Positive rate (target_return_abuse=1): "
          f"{df['target_return_abuse'].mean():.4f}")
    print(df['target_return_abuse'].value_counts())
