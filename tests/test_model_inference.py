"""ML BEHAVIOURAL TESTS - model inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnguard.config import SETTINGS
from churnguard.exceptions import InferenceError, ModelNotLoadedError
from churnguard.models.predictor import ChurnPredictor, assign_risk_band


@pytest.fixture(scope="module")
def predictor(artifact_path) -> ChurnPredictor:
    return ChurnPredictor(artifact_path).load()


@pytest.fixture
def base_record(valid_payload) -> dict:
    return {key: value for key, value in valid_payload.items() if key in SETTINGS.all_features}


class TestMinimumFunctionality:
    def test_output_shape_matches_input(self, predictor, split):
        probabilities = predictor.predict_proba(split.x_test)
        assert probabilities.shape == (len(split.x_test),)

    def test_probabilities_are_in_the_unit_interval(self, predictor, split):
        probabilities = predictor.predict_proba(split.x_test)
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))

    def test_no_nan_or_inf_in_output(self, predictor, split):
        assert np.all(np.isfinite(predictor.predict_proba(split.x_test)))

    def test_single_record_returns_a_complete_prediction(self, predictor, base_record):
        prediction = predictor.predict_one(base_record)
        assert 0.0 <= prediction.churn_probability <= 1.0
        assert prediction.churn_prediction in {0, 1}
        assert prediction.risk_band in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert prediction.recommended_action

    def test_prediction_is_consistent_with_threshold(self, predictor, split):
        predictions = predictor.predict(split.x_test.head(100))
        for prediction in predictions:
            expected = int(prediction.churn_probability >= predictor.threshold)
            assert prediction.churn_prediction == expected


class TestInvariance:
    def test_row_order_does_not_change_individual_predictions(self, predictor, split):
        sample = split.x_test.head(40).reset_index(drop=True)
        original = predictor.predict_proba(sample)
        reversed_frame = sample.iloc[::-1].reset_index(drop=True)
        reversed_probabilities = predictor.predict_proba(reversed_frame)[::-1]
        np.testing.assert_allclose(original, reversed_probabilities, rtol=1e-9)

    def test_batching_matches_one_by_one_scoring(self, predictor, split):
        sample = split.x_test.head(20).reset_index(drop=True)
        batched = predictor.predict_proba(sample)
        individually = np.array(
            [predictor.predict_proba(sample.iloc[[index]])[0] for index in range(len(sample))]
        )
        np.testing.assert_allclose(batched, individually, rtol=1e-9)

    def test_customer_id_does_not_influence_the_score(self, predictor, base_record):
        first = predictor.predict_one({**base_record, "customer_id": "CUST-000001"})
        second = predictor.predict_one({**base_record, "customer_id": "ZZZZ-999999"})
        assert first.churn_probability == second.churn_probability

    def test_column_order_is_irrelevant(self, predictor, split):
        sample = split.x_test.head(25)
        shuffled = sample[list(reversed(sample.columns))]
        np.testing.assert_allclose(
            predictor.predict_proba(sample),
            predictor.predict_proba(shuffled),
            rtol=1e-9,
        )


class TestDirectionalExpectations:
    def test_more_support_tickets_increases_risk(self, predictor, split):
        sample = split.x_test.head(60).copy()
        low = sample.assign(support_tickets_6m=0)
        high = sample.assign(support_tickets_6m=10)
        assert predictor.predict_proba(high).mean() > predictor.predict_proba(low).mean()

    def test_longer_tenure_decreases_risk(self, predictor, split):
        sample = split.x_test.head(60).copy()
        new = sample.assign(tenure_months=2)
        loyal = sample.assign(tenure_months=70)
        assert predictor.predict_proba(loyal).mean() < predictor.predict_proba(new).mean()

    def test_two_year_contract_is_safer_than_month_to_month(self, predictor, split):
        sample = split.x_test.head(60).copy()
        monthly = sample.assign(contract_type="Month-to-month")
        biennial = sample.assign(contract_type="Two year")
        assert predictor.predict_proba(biennial).mean() < predictor.predict_proba(monthly).mean()

    def test_tech_support_reduces_risk(self, predictor, split):
        sample = split.x_test.head(60).copy()
        with_support = predictor.predict_proba(sample.assign(tech_support="Yes")).mean()
        without_support = predictor.predict_proba(sample.assign(tech_support="No")).mean()
        assert with_support < without_support


class TestRobustnessAndErrors:
    def test_unloaded_predictor_raises(self, tmp_path):
        with pytest.raises(ModelNotLoadedError):
            ChurnPredictor(tmp_path / "missing.joblib").predict_one({})

    def test_missing_feature_raises_inference_error(self, predictor, base_record):
        del base_record["contract_type"]
        with pytest.raises(InferenceError, match="Missing required features"):
            predictor.predict_one(base_record)

    def test_empty_batch_is_rejected(self, predictor):
        empty = pd.DataFrame(columns=SETTINGS.all_features)
        with pytest.raises(InferenceError, match="empty batch"):
            predictor.predict_proba(empty)

    def test_unseen_category_degrades_gracefully(self, predictor, base_record):
        prediction = predictor.predict_one({**base_record, "internet_service": "Satellite"})
        assert 0.0 <= prediction.churn_probability <= 1.0

    def test_null_optional_features_are_imputed_not_fatal(self, predictor, base_record):
        prediction = predictor.predict_one(
            {**base_record, "total_charges": None, "avg_monthly_gb": None}
        )
        assert np.isfinite(prediction.churn_probability)


class TestRiskBands:
    @pytest.mark.parametrize(
        ("probability", "expected"),
        [
            (0.05, "LOW"),
            (0.29, "LOW"),
            (0.30, "MEDIUM"),
            (0.59, "MEDIUM"),
            (0.60, "HIGH"),
            (0.84, "HIGH"),
            (0.85, "CRITICAL"),
            (0.99, "CRITICAL"),
        ],
    )
    def test_band_boundaries(self, probability, expected):
        assert assign_risk_band(probability)[0] == expected

    def test_bands_are_monotonic_in_probability(self):
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        ranks = [
            order[assign_risk_band(probability)[0]] for probability in np.linspace(0, 0.999, 50)
        ]
        assert ranks == sorted(ranks)
