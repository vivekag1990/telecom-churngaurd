"""UNIT TESTS -- feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from churnguard.features.engineering import (
    DERIVED_FEATURES,
    ChurnFeatureEngineer,
    apply_derived_features,
    avg_spend_per_month,
    has_no_protection,
    is_new_customer,
)


class TestPureFeatureFunctions:
    def test_avg_spend_per_month_is_arithmetically_correct(self):
        frame = pd.DataFrame({"TotalCharges": [1200.0], "tenure": [12]})
        assert avg_spend_per_month(frame).iloc[0] == pytest.approx(100.0)

    def test_is_new_customer_boundary(self):
        frame = pd.DataFrame({"tenure": [5, 6, 7]})
        assert is_new_customer(frame).tolist() == [1, 1, 0]

    def test_has_no_protection(self):
        frame = pd.DataFrame(
            {
                "TechSupport": ["No", "Yes"],
                "InternetService": ["Fiber optic", "Fiber optic"],
            }
        )
        assert has_no_protection(frame).tolist() == [1, 0]

    def test_functions_are_pure_and_do_not_mutate_input(self, feature_frame):
        before = feature_frame.copy(deep=True)
        apply_derived_features(feature_frame)
        pd.testing.assert_frame_equal(feature_frame, before)


class TestChurnFeatureEngineer:
    def test_fit_transform_shape(self, feature_frame):
        out = ChurnFeatureEngineer().fit_transform(feature_frame)
        assert out.shape == (
            len(feature_frame),
            feature_frame.shape[1] + len(DERIVED_FEATURES),
        )
