"""UNIT TESTS - feature engineering functions and transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnguard.exceptions import FeatureEngineeringError
from churnguard.features.engineering import (
    DERIVED_FEATURES,
    ChurnFeatureEngineer,
    apply_derived_features,
    avg_spend_per_month,
    charge_to_usage_ratio,
    is_new_customer,
    tickets_per_year,
)


class TestPureFeatureFunctions:
    def test_avg_spend_per_month_is_arithmetically_correct(self):
        frame = pd.DataFrame({"total_charges": [1200.0], "tenure_months": [12]})
        assert avg_spend_per_month(frame).iloc[0] == pytest.approx(100.0)

    def test_avg_spend_per_month_never_divides_by_zero(self):
        frame = pd.DataFrame({"total_charges": [50.0], "tenure_months": [0]})
        assert np.isfinite(avg_spend_per_month(frame).iloc[0])

    def test_tickets_per_year_annualises(self):
        frame = pd.DataFrame({"support_tickets_6m": [3]})
        assert tickets_per_year(frame).iloc[0] == 6.0

    def test_is_new_customer_boundary_at_six_months(self):
        frame = pd.DataFrame({"tenure_months": [5, 6, 7]})
        assert is_new_customer(frame).tolist() == [1, 1, 0]

    def test_charge_to_usage_ratio_handles_null_usage(self):
        frame = pd.DataFrame({"monthly_charges": [60.0], "avg_monthly_gb": [np.nan]})
        assert charge_to_usage_ratio(frame).iloc[0] == pytest.approx(60.0)

    def test_functions_are_pure_and_do_not_mutate_input(self, feature_frame):
        before = feature_frame.copy(deep=True)
        apply_derived_features(feature_frame)
        pd.testing.assert_frame_equal(feature_frame, before)


class TestApplyDerivedFeatures:
    def test_all_registered_features_are_produced(self, feature_frame):
        out = apply_derived_features(feature_frame)
        assert set(DERIVED_FEATURES).issubset(out.columns)

    def test_row_count_is_preserved(self, feature_frame):
        assert len(apply_derived_features(feature_frame)) == len(feature_frame)

    def test_missing_input_column_raises_domain_error(self, feature_frame):
        with pytest.raises(FeatureEngineeringError, match="needs column"):
            apply_derived_features(feature_frame.drop(columns=["total_charges"]))


class TestChurnFeatureEngineer:
    def test_fit_transform_shape(self, feature_frame):
        out = ChurnFeatureEngineer().fit_transform(feature_frame)
        assert out.shape == (
            len(feature_frame),
            feature_frame.shape[1] + len(DERIVED_FEATURES),
        )

    def test_transform_before_fit_raises(self, feature_frame):
        with pytest.raises(FeatureEngineeringError, match="before fit"):
            ChurnFeatureEngineer().transform(feature_frame)

    def test_column_order_is_stable_under_shuffled_input(self, feature_frame):
        engineer = ChurnFeatureEngineer().fit(feature_frame)
        shuffled = feature_frame[list(reversed(feature_frame.columns))]
        pd.testing.assert_frame_equal(
            engineer.transform(feature_frame), engineer.transform(shuffled)
        )

    def test_non_dataframe_input_is_rejected(self, feature_frame):
        with pytest.raises(FeatureEngineeringError):
            ChurnFeatureEngineer().fit(feature_frame.to_numpy())
