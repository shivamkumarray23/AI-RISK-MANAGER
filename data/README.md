# Dataset: `transactions.csv`

**100% SYNTHETIC DATA.** No real customer, order, or transaction data was
used anywhere in this project.

- Rows: 6,000
- Positive rate (`target_return_abuse=1`): ~8.5% (realistic imbalance, regenerable in the 5–10% band since it depends on a random per-run base rate draw with `random_state=42`)
- Generator: `generate_data.py` (fully documented label logic in the file's docstring)

## Feature availability split

### Available at decision time (used as MODEL INPUT FEATURES)
`order_amount, product_category, payment_method, discount_percent,
delivery_distance_km, customer_age, customer_account_age_days,
previous_orders, previous_returns, return_rate, returns_last_30_days,
refunds_last_90_days, chargebacks_last_90_days, coupon_usage_count,
high_value_order, customer_risk_history, location_risk_score,
device_change_count, orders_last_30_days, previous_fraud_flag`

These describe the customer's **history up to and including this
order**, not the outcome of this order's return. A merchant can compute
every one of these the moment an order is placed or a return is
*requested* — before anyone knows whether the return itself will be
abusive.

### Post-return information (NEVER used as model input; label-construction only)
`days_to_return, delivery_to_return_days, return_reason`

These only exist **after** a return has been processed. They are used
purely to help simulate the ground-truth label in synthetic data
generation (the same way, in real life, you'd only learn whether a
return was abusive after investigating it). They are dropped before
training/inference — see `ml/train.py::FEATURE_COLUMNS` and
`tests/test_model.py::test_no_leakage_columns`.

## Reproducibility
`random_state = 42` is used throughout generation, splitting, and
training.
