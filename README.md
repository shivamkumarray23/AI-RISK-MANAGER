# 🛡️ AI Return Risk Manager

> **This project uses synthetic data and should not be interpreted as a
> production fraud-detection system.**

A defense-only decision-support tool that scores e-commerce orders for
likely **return/refund abuse risk**, explains *why*, and recommends a
proportionate defensive action — never an automated accusation or block.

---

## 1. Problem

Indian e-commerce merchants lose margin to three related, quietly
compounding sources: fraudulent orders, abusive returns (wardrobing,
"empty box" claims, serial "changed my mind" abuse), and chargebacks.
Manual review teams can't scale to order volume, and blunt rules
("block anyone with >2 returns") create friction for good customers
while missing patterns that only show up across many weak signals.

## 2. Why this matters now

AI-enabled fraud tooling is getting cheaper and more accessible, while
returns and chargebacks silently erode margin in Indian BFSI/e-commerce.
Merchants need **defense-only, explainable, cost-aware** screening —
not another black box that can also be reverse-engineered into an
attack tool.

## 3. Solution

An end-to-end **Return Risk Manager**:
1. Scores every order 0–100 for return-abuse risk using a model trained
   only on information available *before* a return happens.
2. Buckets the score into LOW / MEDIUM / HIGH.
3. Explains the score with model-supported reasons (no invented claims).
4. Recommends **APPROVE / VERIFY / MANUAL_REVIEW** — HIGH risk always
   goes to a human, never an automatic block.
5. Reports honest precision/recall/F1/ROC-AUC on a held-out test set,
   plus a financial cost model with a clear "synthetic data, illustrative
   only" disclaimer.

## 4. Architecture

```
ai-risk-manager/
├── backend/          Flask REST API, risk engine, preprocessing
│   ├── app.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── risk_engine.py
│   └── requirements.txt
├── data/             Synthetic data generator + docs
│   ├── generate_data.py
│   ├── transactions.csv
│   └── README.md
├── ml/               Training, evaluation, saved model & metrics
│   ├── train.py
│   ├── score_all.py
│   ├── saved_model.pkl
│   ├── metrics.json
│   └── scored_transactions.json
├── frontend/         Self-contained HTML dashboard (embeds real metrics)
│   ├── dashboard_template.html
│   └── dashboard.html   (generated, open directly in a browser)
├── tests/            pytest suite (26 tests)
├── requirements.txt
├── Dockerfile / docker-compose.yml
└── README.md
```

## 5. Dataset

6,000 fully synthetic rows (`data/generate_data.py`, seed=42). Positive
class (`target_return_abuse=1`) sits at **~8.5%**, matching the 5–10%
realistic-imbalance requirement. The label is generated from a weighted
combination of ~14 behavioral risk signals plus Gaussian noise, passed
through a logistic function and **sampled** (Bernoulli draw) — not
thresholded — so the data is genuinely noisy and non-trivially
separable, and no single feature determines the outcome (tested in
`tests/test_model.py::test_label_not_trivially_dependent_on_single_feature`).
Full label-generation logic is documented in the module docstring of
`generate_data.py`.

### Leakage prevention (critical requirement)

| Available at decision time (MODEL INPUT) | Post-return only (label-construction only, NEVER fed to the model) |
|---|---|
| order_amount, product_category, payment_method, discount_percent, delivery_distance_km, customer_age, customer_account_age_days, previous_orders, previous_returns, return_rate, returns_last_30_days, refunds_last_90_days, chargebacks_last_90_days, coupon_usage_count, high_value_order, customer_risk_history, location_risk_score, device_change_count, orders_last_30_days, previous_fraud_flag | days_to_return, delivery_to_return_days, return_reason |

The right-hand columns only exist *after* a return has been processed.
They're used to help simulate a realistic ground-truth label during
synthetic data generation, exactly the way a merchant only learns the
true outcome after investigating a return — but they are **excluded**
from `ml/train.py::FEATURE_COLUMNS` and therefore never reach the
model, at training or inference time. This is enforced by
`tests/test_model.py::test_no_leakage_columns`.

## 6. Feature engineering

Five engineered features are derived **only** from decision-time fields
(no post-return information), to help the linear model capture
non-linear risk combinations without leakage:
`recent_return_intensity`, `discount_value`, `risk_flag_sum`,
`new_account_high_value`, `order_velocity_risk`. Verified leakage-free
in `tests/test_model.py::test_engineered_features_do_not_use_post_return_fields`.

## 7. Model

- **Baseline:** Logistic Regression (`class_weight="balanced"`, `max_iter=1000`)
- **Comparison model:** Random Forest (400+ trees, tuned depth/leaf size)
- Both wrapped in an sklearn `Pipeline` + `ColumnTransformer`
  (median-impute + standard-scale numeric; most-frequent-impute + one-hot
  categorical).
- **Model selection was made on VALIDATION F1, not accuracy** — with an
  ~8.5% positive rate, a model that predicts "no abuse" for everyone
  would already be ~91% accurate while being useless.

### Why Logistic Regression was selected here

On this dataset, Random Forest achieves a similar **ranking ability**
(ROC-AUC ≈ 0.74–0.75, comparable to Logistic Regression), but its
predicted probabilities are poorly calibrated around the 0.5 threshold
for this imbalanced, noisy label — at the default threshold it predicts
almost everything as "not abusive," collapsing recall. Logistic
Regression's probabilities are much better calibrated out of the box,
giving materially better precision/recall/F1 at operational thresholds.
Since the business decision (APPROVE/VERIFY/MANUAL_REVIEW) depends on
an actual threshold, not just ranking quality, we selected on
validation F1 rather than AUC alone — and Logistic Regression won.
**This is reported honestly, including Random Forest's weaker
threshold-0.5 numbers** — nothing here was cherry-picked.

## 8. Train / validation / test methodology

```
train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)   # 80% train_full / 20% HELD-OUT TEST
train_test_split(X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=42)  # ~60% train / 20% validation
```

- **Train (60%, 3,600 rows):** fit the pipelines.
- **Validation (20%, 1,200 rows):** model selection (LogReg vs RF) and
  threshold selection. **Never used to fit the final model's
  parameters.**
- **Held-out test (20%, 1,200 rows):** touched exactly once, after every
  modeling decision was already locked in.

Before final held-out evaluation, the selected model is refit on
train+validation (`train_full`) — a standard, leakage-safe practice
since no further decisions are made based on this refit; the test set
is still never seen.

## 9. HELD-OUT TEST RESULTS (chosen threshold = 0.5)

*(regenerate with `python ml/train.py`; numbers below are the actual
values produced by that run — see `ml/metrics.json` for the full record)*

| Metric | Value |
|---|---|
| Precision | 18.5% |
| Recall | 65.0% |
| F1 | 28.8% |
| Accuracy | 72.4% |
| ROC-AUC | 75.9% |
| False Positive Rate | 26.9% |
| False Negative Rate | 35.0% |

Confusion matrix (test, n=1,200): **TP=67, FP=295, FN=36, TN=802**.

### Honest interpretation

Precision is low (18.5%) because the label is deliberately noisy and
the positive class is rare (~8.6%) — this is a hard, realistic problem,
not a toy one. The model still **beats random chance substantially**
(ROC-AUC 0.76 vs 0.50) and, at the chosen threshold, **catches 65% of
abusive returns** while flagging roughly 1 in 4 legitimate transactions
for extra friction. Whether that trade-off is worth it is a business
call — which is exactly why Section 10 (threshold analysis) exists
instead of hard-coding 0.5.

## 10. False-positive / false-negative analysis

- **295 false positives** (test set): legitimate customers who get
  `VERIFY`/`MANUAL_REVIEW` friction they didn't deserve. Cost: ₹150
  each (assumed cost of manual verification / customer friction) = **₹44,250**.
- **36 false negatives**: abusive returns that slip through undetected.
  Cost: ₹1,200 each (assumed average merchant loss per undetected
  abuse case) = **₹43,200**.
- These costs are **assumptions for this hackathon demo**, not measured
  real-world figures — see Section 11.

## 11. Financial cost model

```
FP Cost = False Positives × ₹150
FN Cost = False Negatives × ₹1,200
Total Risk Cost = FP Cost + FN Cost
```

On the held-out test set (n=1,200, ~8.6% actual abuse rate):

| Scenario | Total Cost |
|---|---|
| **Without model** (approve everyone) | ₹123,600 (86 actual abuse cases × ₹1,200) |
| **With model** (chosen threshold) | ₹87,450 |
| **Estimated loss prevented** | **₹36,150** |

⚠️ **These are estimates derived entirely from synthetic data and
assumed per-case costs.** They illustrate the *methodology* for
measuring financial impact, not a real-world savings claim.

## 12. Threshold selection

We do not default to 0.5. Five thresholds were evaluated **on the
validation set only** (never the test set):

| Threshold | Precision | Recall | F1 | FP | FN | Total Cost |
|---|---|---|---|---|---|---|
| 0.30 | 12.6% | 96.1% | 22.4% | 684 | 4 | ₹107,400 |
| 0.40 | 16.8% | 90.3% | 28.4% | 459 | 10 | ₹80,850 |
| **0.50** | **20.7%** | **73.8%** | **32.3%** | **292** | **27** | **₹76,200** |
| 0.60 | 22.5% | 48.5% | 30.8% | 172 | 53 | ₹89,400 |
| 0.70 | 26.0% | 33.0% | 29.1% | 97 | 69 | ₹97,350 |

**Trade-off:** lower thresholds catch almost every abuse case but drown
the review team in false positives (₹150 × hundreds of cases adds up
fast); higher thresholds cut friction but let expensive abuse cases
(₹1,200 each) through. At these assumed costs, **0.50 minimizes total
validation cost** and was locked in before touching the test set.

## 13. Explainability

Reasons shown to a reviewer are generated by
`backend/risk_engine.py::build_reasons`, which only surfaces a reason
if **both** (a) the underlying feature is among the model's
globally-important features (from `feature_importances_` for Random
Forest or `|coef_|` for Logistic Regression) **and** (b) that feature's
value is actually elevated for *this specific transaction*. No reason
is ever invented or hard-coded per transaction.

## 14. Defensive safeguards

- **No auto-block, ever.** LOW→APPROVE, MEDIUM→VERIFY, HIGH→MANUAL_REVIEW.
  HIGH-risk cases are *always* routed to a human.
- **No accusatory language.** The system says "risk detected," never
  "this customer is a fraudster" (enforced in
  `tests/test_risk_engine.py::test_language_never_accuses_customer_directly`).
- **No offensive capability.** Nothing in this codebase helps commit
  fraud, evade detection, exploit payments, or steal credentials. It is
  a read-only scoring/explanation layer.
- **No target leakage.** Post-return fields are excluded from model
  inputs (Section 5/6, enforced by tests).
- **Honest metrics only.** Every number in `ml/metrics.json` and the
  dashboard is generated by actually running `ml/train.py` — nothing is
  hard-coded.

## 15. Installation

```bash
git clone <this-repo>
cd ai-risk-manager
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 16. Running instructions

```bash
# 1. Generate the synthetic dataset
python data/generate_data.py

# 2. Train models + evaluate on held-out test set (writes ml/saved_model.pkl, ml/metrics.json)
python ml/train.py

# 3. Score the full dataset for the dashboard / transaction explorer
python ml/score_all.py

# 4. Run the API
python backend/app.py       # serves on http://localhost:5001

# 5. Run tests
pytest tests/ -v

# 6. Open the dashboard
open frontend/dashboard.html   # static build with embedded, real metrics
```

To rebuild `frontend/dashboard.html` after retraining (so the dashboard
reflects fresh metrics), re-run the small inline script described in
`frontend/README.md` (embeds `ml/metrics.json` + a sample of
`ml/scored_transactions.json` into `dashboard_template.html`).

## 17. API examples

```bash
curl http://localhost:5001/health

curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TX999999", "order_amount": 8500, "product_category": "Electronics",
    "payment_method": "COD", "discount_percent": 55, "delivery_distance_km": 12,
    "customer_age": 27, "customer_account_age_days": 10, "previous_orders": 6,
    "previous_returns": 4, "return_rate": 0.66, "returns_last_30_days": 3,
    "refunds_last_90_days": 3, "chargebacks_last_90_days": 1, "coupon_usage_count": 5,
    "high_value_order": 1, "customer_risk_history": "high", "location_risk_score": 78,
    "device_change_count": 3, "orders_last_30_days": 5, "previous_fraud_flag": 0
  }'
```

```json
{
  "transaction_id": "TX999999",
  "probability": 0.9999,
  "risk_score": 100,
  "risk_level": "HIGH",
  "recommended_action": "MANUAL_REVIEW",
  "action_description": "Route to a human risk analyst for review before any action is taken. Risk detected - this is NOT an accusation of fraud...",
  "reasons": [
    "Customer risk history bucket is elevated",
    "Historical return rate is significantly above normal",
    "Frequent device changes on this account",
    "Delivery location has an elevated risk score"
  ]
}
```

Other endpoints: `GET /metrics`, `GET /transactions`,
`GET /transactions/<id>`, `POST /batch-predict`.

## 18. Dashboard

`frontend/dashboard.html` is a self-contained static build (Chart.js via
CDN) with **real, embedded** `ml/metrics.json` results and a 250-row
representative sample of `ml/scored_transactions.json`. Sections:
Overview, Transaction Explorer (filterable, click-through detail modal),
Model Performance (held-out test set, clearly labeled), Business Impact.

## 19. Limitations

- All data is synthetic; real-world return-abuse patterns, seasonality,
  and adversarial adaptation are not captured.
- Precision at the chosen threshold is modest (~18–20%) — appropriate
  for a "flag for review" tool, not for any automated denial.
- Financial costs (₹150 / ₹1,200) are illustrative assumptions, not
  measured merchant data.
- No temporal/train-serve skew testing (all data is i.i.d. synthetic).
- SHAP was not used (kept to sklearn's built-in importances for speed
  and dependency simplicity); a production version should add
  per-prediction SHAP values for stronger explainability.

## 20. Future improvements

- Replace synthetic data with anonymized real merchant data (with
  privacy review) and re-validate all metrics.
- Add SHAP for per-transaction, not just global, feature attribution.
- Track model drift and retrain on a schedule as return patterns shift.
- Add a feedback loop where MANUAL_REVIEW outcomes retrain the model
  (with strict leakage controls to avoid feeding back post-decision info).
- Calibrate Random Forest probabilities (e.g. `CalibratedClassifierCV`)
  and re-compare against Logistic Regression.
