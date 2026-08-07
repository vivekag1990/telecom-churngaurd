"""INTEGRATION TESTS -- the HTTP layer end to end."""

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
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] is True


class TestPredictEndpoint:
    def test_happy_path_returns_200_and_full_body(self, client, valid_payload):
        response = client.post("/v1/predict", json=valid_payload)
        assert response.status_code == 200
        body = response.json()
        assert body["customer_id"] == valid_payload["customerID"]
        assert 0.0 <= body["churn_probability"] <= 1.0

    def test_invalid_category_returns_422(self, client, valid_payload):
        assert (
            client.post(
                "/v1/predict", json={**valid_payload, "Contract": "Lifetime"}
            ).status_code
            == 422
        )
