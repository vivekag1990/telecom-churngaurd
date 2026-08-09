"""INTEGRATION TESTS - complete FastAPI request path."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churnguard.api import main as api_main
from churnguard.models.predictor import ChurnPredictor


@pytest.fixture(scope="module")
def client(artifact_path):
    api_main.predictor = ChurnPredictor(artifact_path)
    with TestClient(api_main.app) as test_client:
        yield test_client


class TestOperationalEndpoints:
    def test_health_reports_a_loaded_model(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_model_info_exposes_the_model_card(self, client):
        response = client.get("/v1/model/info")
        assert response.status_code == 200
        body = response.json()
        assert body["decision_threshold"] > 0
        assert len(body["input_features"]) == 10
        assert body["hyperparameters"]["calibration"] == "isotonic"

    def test_latency_header_is_present(self, client):
        headers = {key.lower() for key in client.get("/health").headers}
        assert "x-process-time-ms" in headers

    def test_openapi_schema_is_served(self, client):
        schema = client.get("/openapi.json").json()
        assert "/v1/predict" in schema["paths"]
        assert "/v1/predict/batch" in schema["paths"]


class TestPredictEndpoint:
    def test_happy_path_returns_200_and_full_body(self, client, valid_payload):
        response = client.post("/v1/predict", json=valid_payload)
        assert response.status_code == 200
        body = response.json()
        assert body["customer_id"] == valid_payload["customer_id"]
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["risk_band"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert body["recommended_action"]

    def test_high_risk_profile_is_flagged(self, client, valid_payload):
        response = client.post("/v1/predict", json=valid_payload)
        assert response.json()["churn_prediction"] == 1

    def test_low_risk_profile_is_not_flagged(self, client, valid_payload):
        safe = {
            **valid_payload,
            "tenure_months": 68,
            "support_tickets_6m": 0,
            "contract_type": "Two year",
            "payment_method": "Credit card",
            "tech_support": "Yes",
        }
        response = client.post("/v1/predict", json=safe)
        assert response.json()["churn_prediction"] == 0

    def test_missing_required_field_returns_422(self, client, valid_payload):
        del valid_payload["contract_type"]
        response = client.post("/v1/predict", json=valid_payload)
        assert response.status_code == 422

    def test_out_of_range_value_returns_422(self, client, valid_payload):
        response = client.post("/v1/predict", json={**valid_payload, "tenure_months": -5})
        assert response.status_code == 422

    def test_invalid_category_returns_422(self, client, valid_payload):
        response = client.post("/v1/predict", json={**valid_payload, "contract_type": "Lifetime"})
        assert response.status_code == 422

    def test_wrong_type_returns_422(self, client, valid_payload):
        response = client.post("/v1/predict", json={**valid_payload, "monthly_charges": "free"})
        assert response.status_code == 422

    def test_unknown_extra_field_is_rejected(self, client, valid_payload):
        response = client.post("/v1/predict", json={**valid_payload, "tenure_month": 3})
        assert response.status_code == 422

    def test_wrong_http_method_returns_405(self, client):
        assert client.get("/v1/predict").status_code == 405

    def test_unknown_route_returns_404(self, client):
        assert client.get("/v1/does-not-exist").status_code == 404


class TestBatchEndpoint:
    def test_batch_returns_one_prediction_per_customer(self, client, valid_payload):
        response = client.post("/v1/predict/batch", json={"customers": [valid_payload] * 5})
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 5
        assert len(body["predictions"]) == 5
        assert 0.0 <= body["mean_churn_probability"] <= 1.0

    def test_batch_agrees_with_single_endpoint(self, client, valid_payload):
        single = client.post("/v1/predict", json=valid_payload).json()
        batch = client.post("/v1/predict/batch", json={"customers": [valid_payload]}).json()
        assert batch["predictions"][0]["churn_probability"] == single["churn_probability"]

    def test_empty_batch_returns_422(self, client):
        response = client.post("/v1/predict/batch", json={"customers": []})
        assert response.status_code == 422

    def test_oversized_batch_returns_422(self, client, valid_payload):
        response = client.post(
            "/v1/predict/batch",
            json={"customers": [valid_payload] * 501},
        )
        assert response.status_code == 422


class TestDegradedMode:
    def test_predict_returns_503_when_no_model_is_loaded(self, tmp_path, valid_payload):
        api_main.predictor = ChurnPredictor(tmp_path / "absent.joblib")
        with TestClient(api_main.app) as degraded_client:
            health = degraded_client.get("/health").json()
            assert health["status"] == "degraded"
            response = degraded_client.post("/v1/predict", json=valid_payload)
            assert response.status_code == 503
            assert response.json()["error"] == "model_unavailable"
