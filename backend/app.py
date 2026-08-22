"""
app.py
======
Flask REST API for the AI Return Risk Manager.

Endpoints
---------
GET  /health
POST /predict
GET  /metrics
GET  /transactions
GET  /transactions/<transaction_id>
POST /batch-predict

This is a DEFENSIVE decision-support tool. It never automatically
blocks a customer or declares fraud - HIGH risk always routes to
"MANUAL_REVIEW" for a human to make the final call.
"""
import json
import os
import sys

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))
sys.path.insert(0, os.path.join(BASE_DIR, "ml"))

import model as model_module  # noqa: E402
from preprocessing import prepare_row, REQUIRED_INPUT_FIELDS  # noqa: E402
from risk_engine import score_transaction  # noqa: E402

app = Flask(__name__)
CORS(app)

SCORED_TX_PATH = os.path.join(BASE_DIR, "ml", "scored_transactions.json")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


@app.route("/", methods=["GET"])
def dashboard():
    """Serve the static dashboard so a single Render web service can host
    both the API and the UI (no separate static site needed)."""
    return send_from_directory(FRONTEND_DIR, "dashboard.html")


def _load_scored_transactions():
    if not os.path.exists(SCORED_TX_PATH):
        return []
    with open(SCORED_TX_PATH) as f:
        return json.load(f)


@app.route("/health", methods=["GET"])
def health():
    try:
        model_module.load()
        return jsonify({"status": "ok", "model_loaded": True})
    except Exception as e:
        return jsonify({"status": "error", "model_loaded": False, "detail": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Missing JSON body"}), 400
    try:
        eng_df = prepare_row(payload)
    except ValueError as e:
        return jsonify({"error": str(e), "required_fields": REQUIRED_INPUT_FIELDS}), 400

    proba = model_module.predict_proba_df(eng_df)[0]
    importance = model_module.get_feature_importance()
    result = score_transaction(float(proba), payload, importance)
    result["transaction_id"] = payload.get("transaction_id", "N/A")
    result["probability"] = round(float(proba), 4)
    return jsonify(result)


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    payload = request.get_json(force=True, silent=True)
    if not payload or "transactions" not in payload:
        return jsonify({"error": "Body must be {'transactions': [ {...}, ... ]}"}), 400

    importance = model_module.get_feature_importance()
    results = []
    for record in payload["transactions"]:
        try:
            eng_df = prepare_row(record)
        except ValueError as e:
            results.append({"transaction_id": record.get("transaction_id", "N/A"),
                             "error": str(e)})
            continue
        proba = model_module.predict_proba_df(eng_df)[0]
        result = score_transaction(float(proba), record, importance)
        result["transaction_id"] = record.get("transaction_id", "N/A")
        result["probability"] = round(float(proba), 4)
        results.append(result)
    return jsonify({"results": results, "count": len(results)})


@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify(model_module.get_metrics())


@app.route("/transactions", methods=["GET"])
def transactions():
    rows = _load_scored_transactions()

    risk_level = request.args.get("risk_level")
    category = request.args.get("product_category")
    payment_method = request.args.get("payment_method")
    min_amount = request.args.get("min_amount", type=float)
    max_amount = request.args.get("max_amount", type=float)
    limit = request.args.get("limit", default=200, type=int)

    if risk_level:
        rows = [r for r in rows if r["risk_level"].upper() == risk_level.upper()]
    if category:
        rows = [r for r in rows if r["product_category"] == category]
    if payment_method:
        rows = [r for r in rows if r["payment_method"] == payment_method]
    if min_amount is not None:
        rows = [r for r in rows if r["order_amount"] >= min_amount]
    if max_amount is not None:
        rows = [r for r in rows if r["order_amount"] <= max_amount]

    return jsonify({"count": len(rows), "transactions": rows[:limit]})


@app.route("/transactions/<transaction_id>", methods=["GET"])
def transaction_detail(transaction_id):
    rows = _load_scored_transactions()
    for r in rows:
        if r["transaction_id"] == transaction_id:
            return jsonify(r)
    return jsonify({"error": "transaction not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
