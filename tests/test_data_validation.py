"""DATA VALIDATION TESTS -- the data contract, missing values, ranges and drift."""

from __future__ import annotations

import pandas as pd

from churnguard.data.validation import DataValidator


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
