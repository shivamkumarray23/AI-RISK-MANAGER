import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))
sys.path.insert(0, os.path.join(BASE_DIR, "ml"))

MODEL_PATH = os.path.join(BASE_DIR, "ml", "saved_model.pkl")

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="saved_model.pkl not found - run `python ml/train.py` and "
           "`python ml/score_all.py` first"
)

SAMPLE_TX = {
    "transaction_id": "TX999999", "order_amount": 8500, "product_category": "Electronics",
    "payment_method": "COD", "discount_percent": 55, "delivery_distance_km": 12,
    "customer_age": 27, "customer_account_age_days": 10, "previous_orders": 6,
    "previous_returns": 4, "return_rate": 0.66, "returns_last_30_days": 3,
    "refunds_last_90_days": 3, "chargebacks_last_90_days": 1, "coupon_usage_count": 5,
    "high_value_order": 1, "customer_risk_history": "high", "location_risk_score": 78,
    "device_change_count": 3, "orders_last_30_days": 5, "previous_fraud_flag": 0,
}


@pytest.fixture
def client():
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.get_json()
    assert "held_out_test_performance_default_threshold_0.5" in body


def test_predict_returns_expected_shape(client):
    r = client.post("/predict", json=SAMPLE_TX)
    assert r.status_code == 200
    body = r.get_json()
    for key in ["risk_score", "risk_level", "recommended_action", "reasons"]:
        assert key in body
    assert 0 <= body["risk_score"] <= 100
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_predict_missing_fields_returns_400(client):
    r = client.post("/predict", json={"order_amount": 100})
    assert r.status_code == 400


def test_batch_predict(client):
    r = client.post("/batch-predict", json={"transactions": [SAMPLE_TX, SAMPLE_TX]})
    assert r.status_code == 200
    body = r.get_json()
    assert body["count"] == 2


def test_transactions_listing(client):
    r = client.get("/transactions?limit=5")
    assert r.status_code == 200
    body = r.get_json()
    assert len(body["transactions"]) <= 5


def test_transactions_filter_by_risk_level(client):
    r = client.get("/transactions?risk_level=HIGH")
    assert r.status_code == 200
    body = r.get_json()
    assert all(t["risk_level"] == "HIGH" for t in body["transactions"])


def test_transaction_detail_not_found(client):
    r = client.get("/transactions/DOES_NOT_EXIST")
    assert r.status_code == 404
