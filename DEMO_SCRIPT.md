# 3-Minute Demo Script — AI Return Risk Manager

**0:00–0:30 — Problem**
> "Indian e-commerce merchants lose margin three ways: fraud, abusive
> returns, and chargebacks. Manual review can't scale, and blunt rules
> like 'block anyone with 2+ returns' punish good customers while
> missing patterns that only show up across many weak signals. We built
> a defense-only Return Risk Manager that scores every order for
> return-abuse risk, explains why, and routes only the riskiest cases
> to a human — never an automatic block."

**0:30–1:00 — Solution**
> "It's an end-to-end pipeline: synthetic 6,000-order dataset with a
> realistically noisy, ~8.5%-imbalanced label; a Logistic Regression
> model trained only on information available *before* a return
> happens — no leakage from post-return fields like return reason or
> days-to-return; a 0–100 risk score with LOW/MEDIUM/HIGH bands; and a
> three-tier action: APPROVE, VERIFY, or MANUAL_REVIEW."

**1:00–1:30 — Live prediction**
*(hit the `/predict` endpoint or click a HIGH-risk row in the dashboard's Transaction Explorer)*
> "Here's a ₹8,500 electronics order, COD, 55% discount, from a 10-day-old
> account with a 66% historical return rate and a chargeback on file.
> The model scores it 100/100 — HIGH risk — and gives model-supported
> reasons: elevated risk history, high return rate, frequent device
> changes, elevated location risk. The action is MANUAL_REVIEW, not
> 'blocked' — a human makes the final call."

**1:30–2:00 — Model metrics**
*(Model Performance tab)*
> "On the held-out 20% test set — never touched during training or
> threshold tuning — we get 65% recall and 76% ROC-AUC. Precision is a
> modest 18–20%, and we say that plainly: this is a hard, noisy,
> realistic problem, not a cherry-picked demo. We chose the operating
> threshold using validation-set cost analysis across five thresholds,
> not by defaulting to 0.5."

**2:00–2:30 — Financial impact**
*(Business Impact tab)*
> "On the test set, false positives cost ₹150 each in review friction;
> false negatives cost ₹1,200 each in missed abuse. Without any model,
> screening everyone through would cost ₹123,600 in undetected abuse.
> With the model at our chosen threshold, total cost drops to ₹87,450 —
> about ₹36,000 prevented on this test set alone. We're explicit this
> is a synthetic-data estimate of methodology, not a real savings claim."

**2:30–3:00 — Why this is safe, explainable, and useful**
> "Every reason shown is model-supported, never invented. HIGH risk
> always goes to a human — the system never says 'this is a fraudster,'
> only 'risk detected.' There's zero offensive capability in this
> codebase: no attack techniques, no evasion methods, nothing that
> helps commit fraud. It's a read-only, explainable, cost-aware
> decision-support layer — exactly what the 'Risk Manager' track asks
> for."
