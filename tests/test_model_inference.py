"""ML BEHAVIOURAL TESTS -- model INFERENCE."""

from __future__ import annotations

import numpy as np
import pytest

from churnguard.config import SETTINGS
from churnguard.exceptions import ModelNotLoadedError
from churnguard.models.predictor import ChurnPredictor


@pytest.fixture(scope="module")
def predictor(artifact_path) -> ChurnPredictor:
    return ChurnPredictor(artifact_path).load()


@pytest.fixture
def base_record(valid_payload) -> dict:
    return {k: v for k, v in valid_payload.items() if k in SETTINGS.all_features}


class TestMinimumFunctionality:
    def test_output_shape_matches_input(self, predictor, split):
        probabilities = predictor.predict_proba(split.x_test)
        assert probabilities.shape == (len(split.x_test),)

    def test_probabilities_are_in_the_unit_interval(self, predictor, split):
        probabilities = predictor.predict_proba(split.x_test)
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


class TestDirectionalExpectations:
    def test_longer_tenure_decreases_risk(self, predictor, split):
        sample = split.x_test.head(60).copy()
        new = sample.assign(tenure=2)
        loyal = sample.assign(tenure=70)
        assert (
            predictor.predict_proba(loyal).mean() < predictor.predict_proba(new).mean()
        )


class TestRobustnessAndErrors:
    def test_unloaded_predictor_raises(self, tmp_path):
        with pytest.raises(ModelNotLoadedError):
            ChurnPredictor(tmp_path / "missing.joblib").predict_one({})
