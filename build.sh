#!/usr/bin/env bash
# build.sh - run on Render during deploy.
# Regenerates the synthetic dataset, retrains the model, and scores the
# dataset for the dashboard - so every deployed instance has fresh,
# genuinely-computed artifacts (not stale committed binaries).
set -e

pip install -r requirements.txt

python data/generate_data.py
python ml/train.py
python ml/score_all.py

python3 - << 'EOF'
import json, random
from collections import defaultdict

with open("ml/metrics.json") as f:
    metrics = json.load(f)
with open("ml/scored_transactions.json") as f:
    rows = json.load(f)

random.seed(42)
buckets = defaultdict(list)
for r in rows:
    buckets[r["risk_level"]].append(r)
sample = []
for lvl, n in [("HIGH", 80), ("MEDIUM", 80), ("LOW", 90)]:
    sample.extend(random.sample(buckets[lvl], min(n, len(buckets[lvl]))))
random.shuffle(sample)

with open("frontend/dashboard_template.html") as f:
    tpl = f.read()
tpl = tpl.replace("__METRICS_JSON__", json.dumps(metrics))
tpl = tpl.replace("__SAMPLE_JSON__", json.dumps(sample))
with open("frontend/dashboard.html", "w") as f:
    f.write(tpl)

print("Rebuilt frontend/dashboard.html with fresh metrics")
EOF

echo "Build complete."
