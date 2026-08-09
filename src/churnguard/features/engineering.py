"""Feature engineering."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnguard.config import SETTINGS
from churnguard.exceptions import FeatureEngineeringError

logger = logging.getLogger(__name__)

FeatureFn = Callable[[pd.DataFrame], pd.Series]


def avg_spend_per_month(frame: pd.DataFrame) -> pd.Series:
    """Return lifetime spend per active month."""
    return (frame["total_charges"] / frame["tenure_months"].clip(lower=1)).astype(float)


def tickets_per_year(frame: pd.DataFrame) -> pd.Series:
    """Return annualised support-ticket count."""
    return (frame["support_tickets_6m"] * 2.0).astype(float)


def is_new_customer(frame: pd.DataFrame) -> pd.Series:
    return (frame["tenure_months"] <= 6).astype(int)


def charge_to_usage_ratio(frame: pd.DataFrame) -> pd.Series:
    """Return monthly charges per GB used."""
    return (frame["monthly_charges"] / frame["avg_monthly_gb"].fillna(0.0).clip(lower=1.0)).astype(
        float
    )


def has_no_protection(frame: pd.DataFrame) -> pd.Series:
    return ((frame["tech_support"] == "No") & (frame["internet_service"] == "Fiber optic")).astype(
        int
    )


DERIVED_FEATURES: Mapping[str, FeatureFn] = {
    "avg_spend_per_month": avg_spend_per_month,
    "tickets_per_year": tickets_per_year,
    "is_new_customer": is_new_customer,
    "charge_to_usage_ratio": charge_to_usage_ratio,
    "has_no_protection": has_no_protection,
}


def apply_derived_features(
    frame: pd.DataFrame, registry: Mapping[str, FeatureFn] = DERIVED_FEATURES
) -> pd.DataFrame:
    """Append registered features without modifying the input frame."""
    out = frame.copy()
    for name, fn in registry.items():
        try:
            out[name] = fn(frame)
        except KeyError as exc:
            logger.error("Cannot compute '%s': missing input column %s", name, exc)
            raise FeatureEngineeringError(f"Feature '{name}' needs column {exc}") from exc
        except (TypeError, ValueError) as exc:
            logger.error("Feature '%s' failed: %s", name, exc)
            raise FeatureEngineeringError(f"Feature '{name}' could not be computed") from exc
    return out


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Expose feature functions through the scikit-learn transformer API."""

    def __init__(self, registry: Mapping[str, FeatureFn] | None = None) -> None:
        self.registry = registry or DERIVED_FEATURES

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ChurnFeatureEngineer:
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("ChurnFeatureEngineer expects a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        # Store output order so request-column order cannot affect inference.
        self.output_columns_ = list(apply_derived_features(X.head(2), self.registry).columns)
        logger.info(
            "ChurnFeatureEngineer fitted: %d in -> %d out",
            len(self.feature_names_in_),
            len(self.output_columns_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "output_columns_"):
            raise FeatureEngineeringError("transform() called before fit()")
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("ChurnFeatureEngineer expects a pandas DataFrame")
        missing = set(self.feature_names_in_) - set(X.columns)
        if missing:
            raise FeatureEngineeringError(f"Input is missing columns: {sorted(missing)}")
        return apply_derived_features(X, self.registry)[self.output_columns_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.output_columns_, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Build numeric and categorical preprocessing pipelines."""
    numeric_columns = list(SETTINGS.model.numeric_features) + [
        "avg_spend_per_month",
        "tickets_per_year",
        "is_new_customer",
        "charge_to_usage_ratio",
        "has_no_protection",
    ]
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            (
                "categorical",
                categorical_pipeline,
                list(SETTINGS.model.categorical_features),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
