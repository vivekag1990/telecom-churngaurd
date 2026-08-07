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
    total = pd.to_numeric(frame["TotalCharges"], errors="coerce").fillna(0.0)
    return (total / frame["tenure"].clip(lower=1)).astype(float)


def is_new_customer(frame: pd.DataFrame) -> pd.Series:
    return (frame["tenure"] <= 6).astype(int)


def has_no_protection(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["TechSupport"] == "No") & (frame["InternetService"] == "Fiber optic")
    ).astype(int)


DERIVED_FEATURES: Mapping[str, FeatureFn] = {
    "avg_spend_per_month": avg_spend_per_month,
    "is_new_customer": is_new_customer,
    "has_no_protection": has_no_protection,
}


def apply_derived_features(
    frame: pd.DataFrame, registry: Mapping[str, FeatureFn] = DERIVED_FEATURES
) -> pd.DataFrame:
    out = frame.copy()

    # Fix TotalCharges to be numeric before computing features
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(
            out["TotalCharges"], errors="coerce"
        ).fillna(0.0)

    for name, fn in registry.items():
        try:
            out[name] = fn(out)
        except KeyError as exc:
            logger.error("Cannot compute '%s': missing input column %s", name, exc)
            raise FeatureEngineeringError(
                f"Feature '{name}' needs column {exc}"
            ) from exc
        except (TypeError, ValueError) as exc:
            logger.error("Feature '%s' failed: %s", name, exc)
            raise FeatureEngineeringError(
                f"Feature '{name}' could not be computed"
            ) from exc
    return out


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, registry: Mapping[str, FeatureFn] | None = None) -> None:
        self.registry = registry or DERIVED_FEATURES

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ChurnFeatureEngineer:
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError(
                "ChurnFeatureEngineer expects a pandas DataFrame"
            )
        self.feature_names_in_ = list(X.columns)
        self.output_columns_ = list(
            apply_derived_features(X.head(2), self.registry).columns
        )
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
            raise FeatureEngineeringError(
                "ChurnFeatureEngineer expects a pandas DataFrame"
            )
        missing = set(self.feature_names_in_) - set(X.columns)
        if missing:
            raise FeatureEngineeringError(
                f"Input is missing columns: {sorted(missing)}"
            )
        return apply_derived_features(X, self.registry)[self.output_columns_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.output_columns_, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    numeric_columns = list(SETTINGS.model.numeric_features) + [
        "avg_spend_per_month",
        "is_new_customer",
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
