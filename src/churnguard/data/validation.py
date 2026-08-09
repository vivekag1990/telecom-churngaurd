"""Data quality: schema contract, missing-value gates and drift detection."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from churnguard.config import SETTINGS
from churnguard.exceptions import SchemaValidationError

logger = logging.getLogger(__name__)

NUMERIC_CONTRACT: dict[str, tuple[float, float]] = {
    "tenure_months": (0.0, 120.0),
    "monthly_charges": (0.0, 500.0),
    "total_charges": (0.0, 60_000.0),
    "support_tickets_6m": (0.0, 60.0),
    "avg_monthly_gb": (0.0, 1_000.0),
}

CATEGORICAL_CONTRACT: dict[str, set[str]] = {
    "contract_type": {"Month-to-month", "One year", "Two year"},
    "internet_service": {"Fiber optic", "DSL", "No"},
    "payment_method": {
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card",
    },
    "tech_support": {"Yes", "No"},
    "paperless_billing": {"Yes", "No"},
}


@dataclass
class ValidationReport:
    """Structured, serialisable outcome of a validation run."""

    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.passed = False
        self.errors.append(message)
        logger.error("DATA-QUALITY FAIL | %s", message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
        logger.warning("DATA-QUALITY WARN | %s", message)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }

    def raise_for_status(self) -> None:
        if not self.passed:
            raise SchemaValidationError(
                f"Data validation failed with {len(self.errors)} error(s)", self.errors
            )


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10
) -> float:
    """Return PSI for two numeric samples using reference quantile bins."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if reference.size == 0 or current.size == 0:
        return 0.0

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(reference, quantiles))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=edges)[0] / reference.size
    cur_pct = np.histogram(current, bins=edges)[0] / current.size
    # Clipping prevents division by zero when a bin is empty in one sample.
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


class DataValidator:
    """Validates a DataFrame against the ChurnGuard data contract."""

    def __init__(self, max_missing_fraction: float | None = None) -> None:
        self.max_missing_fraction = (
            SETTINGS.gates.max_missing_fraction
            if max_missing_fraction is None
            else max_missing_fraction
        )

    def validate_schema(self, frame: pd.DataFrame, report: ValidationReport) -> None:
        for column in SETTINGS.all_features:
            if column not in frame.columns:
                report.add_error(f"Missing required column '{column}'")
        for column in NUMERIC_CONTRACT:
            if column in frame.columns and not pd.api.types.is_numeric_dtype(frame[column]):
                report.add_error(f"Column '{column}' must be numeric, got {frame[column].dtype}")
        for column, levels in CATEGORICAL_CONTRACT.items():
            if column not in frame.columns:
                continue
            unexpected = set(frame[column].dropna().unique()) - levels
            if unexpected:
                report.add_error(f"Column '{column}' has unexpected levels {sorted(unexpected)}")

    def validate_missing(self, frame: pd.DataFrame, report: ValidationReport) -> None:
        overall = float(
            frame[[c for c in SETTINGS.all_features if c in frame]].isna().mean().mean()
        )
        report.metrics["missing_fraction_overall"] = round(overall, 5)
        for column in SETTINGS.all_features:
            if column not in frame.columns:
                continue
            fraction = float(frame[column].isna().mean())
            report.metrics[f"missing_fraction__{column}"] = round(fraction, 5)
            if fraction > self.max_missing_fraction:
                report.add_error(
                    f"Column '{column}' is {fraction:.2%} null "
                    f"(limit {self.max_missing_fraction:.0%})"
                )
            elif fraction > 0:
                report.add_warning(f"Column '{column}' has {fraction:.2%} nulls (imputed)")

    def validate_ranges(self, frame: pd.DataFrame, report: ValidationReport) -> None:
        for column, (low, high) in NUMERIC_CONTRACT.items():
            if column not in frame.columns or not pd.api.types.is_numeric_dtype(frame[column]):
                continue
            series = frame[column].dropna()
            out_of_range = int(((series < low) | (series > high)).sum())
            if out_of_range:
                report.add_error(
                    f"Column '{column}' has {out_of_range} " f"value(s) outside [{low}, {high}]"
                )

    def validate_drift(
        self,
        frame: pd.DataFrame,
        reference_profile: dict[str, list[float]],
        report: ValidationReport,
        max_psi: float | None = None,
    ) -> None:
        limit = SETTINGS.gates.max_psi if max_psi is None else max_psi
        for column, reference_values in reference_profile.items():
            if column not in frame.columns:
                continue
            psi = population_stability_index(np.asarray(reference_values), frame[column].to_numpy())
            report.metrics[f"psi__{column}"] = round(psi, 4)
            if psi > limit:
                report.add_error(f"Feature '{column}' drifted: PSI={psi:.3f} > {limit:.2f}")
            elif psi > 0.10:
                report.add_warning(f"Feature '{column}' moderately shifted: PSI={psi:.3f}")

    def validate(
        self,
        frame: pd.DataFrame,
        reference_profile: dict[str, list[float]] | None = None,
    ) -> ValidationReport:
        logger.info("Validating dataset: %d rows x %d columns", *frame.shape)
        report = ValidationReport()
        report.metrics["n_rows"] = int(len(frame))
        self.validate_schema(frame, report)
        self.validate_missing(frame, report)
        self.validate_ranges(frame, report)
        if reference_profile:
            self.validate_drift(frame, reference_profile, report)
        logger.info(
            "Validation %s | %d error(s), %d warning(s)",
            "PASSED" if report.passed else "FAILED",
            len(report.errors),
            len(report.warnings),
        )
        return report


def write_reference_profile(frame: pd.DataFrame, path: Path) -> dict[str, list[float]]:
    """Save numeric training values used by later drift checks."""
    profile = {
        column: frame[column].dropna().astype(float).tolist()
        for column in SETTINGS.model.numeric_features
        if column in frame.columns
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v for k, v in profile.items()}))
    logger.info("Reference profile with %d features written to %s", len(profile), path)
    return profile


def load_reference_profile(path: Path) -> dict[str, list[float]]:
    """Load a drift reference profile when one is available."""
    if not path.exists():
        logger.warning("Reference profile %s not found; drift checks skipped", path)
        return {}
    return json.loads(path.read_text())
