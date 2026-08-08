"""DATA VALIDATION TESTS -- the data contract, missing values, ranges and drift."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from churnguard.data.validation import (
    DataValidator,
    population_stability_index,
    write_reference_profile,
)


class TestSchemaContract:
    def test_clean_training_data_passes(self, raw_frame):
        report = DataValidator().validate(raw_frame)
        assert report.passed, report.errors

    def test_renamed_column_is_caught(self, raw_frame):
        broken = raw_frame.rename(columns={"tenure": "tenure_months"})
        report = DataValidator().validate(broken)
        assert not report.passed
        assert any("tenure" in e for e in report.errors)

    def test_wrong_dtype_is_caught(self, raw_frame):
        broken = raw_frame.copy()
        broken["MonthlyCharges"] = broken["MonthlyCharges"].astype(str)
        report = DataValidator().validate(broken)
        assert not report.passed
        assert any("must be numeric" in e for e in report.errors)

    def test_unexpected_category_level_is_caught(self, raw_frame):
        broken = raw_frame.copy()
        broken.loc[broken.index[0], "Contract"] = "Lifetime"
        report = DataValidator().validate(broken)
        assert not report.passed
        assert any("Lifetime" in e for e in report.errors)


class TestRangeChecks:
    def test_negative_charges_are_rejected(self, raw_frame):
        broken = raw_frame.copy()
        broken.loc[broken.index[0], "MonthlyCharges"] = -12.0
        report = DataValidator().validate(broken)
        assert not report.passed
        assert any("outside" in e for e in report.errors)

    def test_unit_change_dollars_to_cents_is_rejected(self, raw_frame):
        broken = raw_frame.copy()
        broken["MonthlyCharges"] = (
            pd.to_numeric(broken["MonthlyCharges"], errors="coerce") * 100
        )
        report = DataValidator().validate(broken)
        assert not report.passed


class TestDriftDetection:
    def test_psi_is_zero_for_identical_distributions(self):
        rng = np.random.default_rng(42)
        reference = rng.normal(50, 10, size=2000)
        psi = population_stability_index(reference, reference)
        assert psi < 1e-6

    def test_psi_is_near_zero_for_same_population_resample(self, raw_frame):
        rng = np.random.default_rng(7)
        shuffled = raw_frame.sample(frac=1.0, random_state=rng.integers(1_000_000))
        half = len(shuffled) // 2
        reference = shuffled["tenure"].head(half).to_numpy(dtype=float)
        current = shuffled["tenure"].tail(len(shuffled) - half).to_numpy(dtype=float)
        psi = population_stability_index(reference, current)
        assert psi < 0.10  # below the "moderate shift" threshold

    def test_psi_is_high_for_a_genuinely_shifted_population(self, raw_frame):
        reference = raw_frame["tenure"].to_numpy(dtype=float)
        # Simulate an intake of exclusively brand-new, long-tenured customers --
        # a plausible real shift (e.g. a bulk enterprise onboarding).
        shifted = np.concatenate(
            [
                np.zeros(200),
                np.full(200, 118.0),
            ]  # bimodal: instant churn risk + max tenure
        )
        psi = population_stability_index(reference, shifted)
        assert psi > 0.20  # above the "significant shift" gate

    def test_validate_flags_drifted_feature_as_error(self, raw_frame):
        reference_profile = write_reference_profile(
            raw_frame, Path("/tmp/psi_test_reference_profile.json")
        )
        shifted = raw_frame.copy()
        shifted["MonthlyCharges"] = 9999.0  # clamped later by range check too,
        # but drift must independently flag it regardless of range validity
        shifted.loc[shifted.index, "tenure"] = np.where(shifted.index % 2 == 0, 0, 118)
        report = DataValidator().validate(shifted, reference_profile=reference_profile)
        assert not report.passed
        assert any("drifted" in e for e in report.errors)
        assert report.metrics["psi__tenure"] > 0.20
