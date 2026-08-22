# Frontend: Static Dashboard

`dashboard.html` is generated from `dashboard_template.html` by
embedding real, freshly computed JSON (no hard-coded numbers).

## Rebuild after retraining

```bash
python3 - << 'EOF'
import json

with open("../ml/metrics.json") as f:
    metrics = json.load(f)

# a representative sample across risk levels, so the Transaction
# Explorer has realistic rows without embedding the full ~6000-row file
with open("../ml/scored_transactions.json") as f:
    rows = json.load(f)

from collections import defaultdict
import random
random.seed(42)
buckets = defaultdict(list)
for r in rows:
    buckets[r["risk_level"]].append(r)
sample = []
for lvl, n in [("HIGH", 80), ("MEDIUM", 80), ("LOW", 90)]:
    sample.extend(random.sample(buckets[lvl], min(n, len(buckets[lvl]))))
random.shuffle(sample)

with open("dashboard_template.html") as f:
    tpl = f.read()

tpl = tpl.replace("__METRICS_JSON__", json.dumps(metrics))
tpl = tpl.replace("__SAMPLE_JSON__", json.dumps(sample))

with open("dashboard.html", "w") as f:
    f.write(tpl)

print("Rebuilt dashboard.html")
EOF
```

The dashboard is a static file — open it directly in a browser. It does
NOT call the live Flask API; it's a self-contained demo view. For a
live, wired-up view (fetching `/predict`, `/transactions`, etc. from
`backend/app.py`), point a small fetch-based frontend at
`http://localhost:5001` — the API is CORS-enabled for this purpose.
