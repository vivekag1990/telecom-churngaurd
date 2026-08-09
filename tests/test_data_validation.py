"""DATA VALIDATION TESTS - schema, missingness, ranges and drift."""

from __future__ import annotations

import numpy as np
import pytest

from churnguard.data.validation import (
    DataValidator,
    ValidationReport,
    population_stability_index,
    write_reference_profile,
)
from churnguard.exceptions import SchemaValidationError


class TestSchemaContract:
    def test_clean_training_data_passes(self, raw_frame):
        report = DataValidator().validate(raw_frame)
        assert report.passed, report.errors

    def test_renamed_column_is_caught(self, raw_frame):
        broken = raw_frame.rename(columns={"tenure_months": "tenure"})
        report = DataValidator().validate(broken)
        assert not report.passed
        assert any("tenure_months" in error for error in report.errors)

    def test_wrong_dtype_is_caught(self, raw_frame):
        broken = raw_frame.copy()
        broken["monthly_charges"] = broken["monthly_charges"].astype(str)
        report = DataValidator().validate(broken)
        assert not report.passed
        assert any("must be numeric" in error for error in report.errors)

    def test_unexpected_category_level_is_caught(self, raw_frame):
        broken = raw_frame.copy()
        broken.loc[broken.index[0], "contract_type"] = "Lifetime"
        report = DataValidator().validate(broken)
        assert not report.passed
        assert any("Lifetime" in error for error in report.errors)


class TestMissingValues:
    def test_low_missingness_is_only_a_warning(self, raw_frame):
        report = DataValidator().validate(raw_frame)
        assert report.passed
        assert report.warnings

    def test_missing_flood_fails_the_gate(self, raw_frame):
        broken = raw_frame.copy()
        broken.loc[broken.index[:600], "monthly_charges"] = np.nan
        report = DataValidator().validate(broken)
        assert not report.passed
        assert any("null" in error for error in report.errors)

    def test_per_column_missing_metric_is_recorded(self, raw_frame):
        report = DataValidator().validate(raw_frame)
        assert report.metrics["missing_fraction__total_charges"] > 0
        assert report.metrics["missing_fraction__tenure_months"] == 0.0


class TestRangeChecks:
    def test_negative_charges_are_rejected(self, raw_frame):
        broken = raw_frame.copy()
        broken.loc[broken.index[0], "monthly_charges"] = -12.0
        report = DataValidator().validate(broken)
        assert not report.passed
        assert any("outside" in error for error in report.errors)

    def test_unit_change_dollars_to_cents_is_rejected(self, raw_frame):
        broken = raw_frame.copy()
        broken["monthly_charges"] = broken["monthly_charges"] * 100
        assert not DataValidator().validate(broken).passed


class TestDriftDetection:
    def test_psi_of_identical_distributions_is_near_zero(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(50, 10, 5000)
        assert population_stability_index(sample, sample.copy()) < 0.01

    def test_psi_grows_with_a_mean_shift(self):
        rng = np.random.default_rng(1)
        reference = rng.normal(50, 10, 5000)
        small = population_stability_index(reference, rng.normal(52, 10, 5000))
        large = population_stability_index(reference, rng.normal(70, 10, 5000))
        assert small < large
        assert large > 0.25

    def test_drifted_traffic_fails_validation(self, raw_frame, tmp_path):
        profile = write_reference_profile(raw_frame, tmp_path / "profile.json")
        drifted = raw_frame.copy()
        drifted["tenure_months"] = (drifted["tenure_months"] * 0.25).round().astype(int)
        report = DataValidator().validate(drifted, reference_profile=profile)
        assert not report.passed
        assert any("drifted" in error for error in report.errors)
        assert report.metrics["psi__tenure_months"] > 0.20

    def test_stable_traffic_passes_drift_check(self, raw_frame, tmp_path):
        profile = write_reference_profile(raw_frame, tmp_path / "profile.json")
        sample = raw_frame.sample(frac=0.5, random_state=3)
        report = DataValidator().validate(sample, profile)
        assert report.passed, report.errors


class TestReportBehaviour:
    def test_raise_for_status_raises_on_failure(self):
        report = ValidationReport()
        report.add_error("boom")
        with pytest.raises(SchemaValidationError) as caught:
            report.raise_for_status()
        assert caught.value.violations == ["boom"]

    def test_raise_for_status_is_silent_on_success(self):
        ValidationReport().raise_for_status()
