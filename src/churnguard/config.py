"""Centralised, immutable configuration.

Configuration is a *value*, not a set of scattered module-level constants, so it
is expressed as a frozen dataclass. Values may be overridden through environment
variables, which is what lets the same code run unchanged on a laptop, in CI and
in a container.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. All paths are absolute and derived from PROJECT_ROOT."""

    root: Path = PROJECT_ROOT
    raw_data: Path = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
    reference_profile: Path = PROJECT_ROOT / "artifacts" / "reference_profile.json"
    model_artifact: Path = PROJECT_ROOT / "artifacts" / "churn_model.joblib"
    metrics_report: Path = PROJECT_ROOT / "reports" / "model_metrics.json"
    log_file: Path = PROJECT_ROOT / "logs" / "churnguard.log"


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters and the target/feature contract."""

    target: str = "churn"
    id_column: str = "customer_id"
    numeric_features: tuple[str, ...] = (
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "support_tickets_6m",
        "avg_monthly_gb",
    )
    categorical_features: tuple[str, ...] = (
        "contract_type",
        "internet_service",
        "payment_method",
        "tech_support",
        "paperless_billing",
    )
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    max_depth: int = 12
    min_samples_leaf: int = 8
    class_weight: str = "balanced"
    #: Tuned on the validation split (see reports/threshold_sweep.csv). 0.50 is NOT
    #: the right default here: losing a customer costs ~18x the price of a retention
    #: discount, so the threshold is deliberately pushed down to buy recall. F1 peaks
    #: at 0.34-0.35 on held-out data.
    decision_threshold: float = _env_float("CHURN_DECISION_THRESHOLD", 0.35)


@dataclass(frozen=True)
class QualityGates:
    """Thresholds a model or dataset must clear to be considered acceptable."""

    min_roc_auc: float = 0.75
    min_f1: float = 0.55
    max_brier: float = 0.20
    max_missing_fraction: float = 0.05
    max_psi: float = 0.20


@dataclass(frozen=True)
class Settings:
    """Top-level settings object consumed by every module."""

    paths: Paths = field(default_factory=Paths)
    model: ModelConfig = field(default_factory=ModelConfig)
    gates: QualityGates = field(default_factory=QualityGates)
    log_level: str = _env("CHURN_LOG_LEVEL", "INFO")
    api_version: str = "v1"

    @property
    def all_features(self) -> list[str]:
        return list(self.model.numeric_features) + list(self.model.categorical_features)


SETTINGS = Settings()
