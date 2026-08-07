"""Model training."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from churnguard import __version__
from churnguard.config import SETTINGS, ModelConfig
from churnguard.exceptions import ModelTrainingError
from churnguard.features.engineering import ChurnFeatureEngineer, build_preprocessor
from churnguard.models.evaluation import ModelMetrics, evaluate

logger = logging.getLogger(__name__)


@dataclass
class TrainingArtifact:
    pipeline: Pipeline
    metrics: ModelMetrics
    metadata: dict[str, object] = field(default_factory=dict)


class ChurnModelTrainer:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or SETTINGS.model
        self.pipeline: Pipeline | None = None

    def build_pipeline(self) -> Pipeline:
        forest = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        classifier = CalibratedClassifierCV(forest, method="isotonic", cv=3)
        pipeline = Pipeline(
            [
                ("features", ChurnFeatureEngineer()),
                ("preprocess", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
        logger.info(
            "Pipeline built with steps: %s", [name for name, _ in pipeline.steps]
        )
        return pipeline

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        if len(x_train) != len(y_train):
            raise ModelTrainingError(
                f"X has {len(x_train)} rows but y has {len(y_train)} "
                "-- refusing to train"
            )

        # Handle string y_train
        if pd.api.types.is_string_dtype(y_train):
            y_train = (y_train == "Yes").astype(int)

        if pd.Series(y_train).nunique() < 2:
            logger.error("Training target contains a single class")
            raise ModelTrainingError("Training requires at least two target classes")

        logger.info(
            "Training on %d rows (churn rate %.3f)", len(x_train), float(y_train.mean())
        )
        self.pipeline = self.build_pipeline()
        try:
            self.pipeline.fit(x_train, y_train)
        except Exception as exc:
            logger.exception("Training failed")
            raise ModelTrainingError(f"Model training failed: {exc}") from exc
        logger.info("Training complete")
        return self.pipeline

    def cross_validate(
        self, x: pd.DataFrame, y: pd.Series, folds: int = 5
    ) -> dict[str, float]:
        if pd.api.types.is_string_dtype(y):
            y = (y == "Yes").astype(int)

        cv = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=self.config.random_state
        )
        scores = cross_val_score(
            self.build_pipeline(), x, y, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        result = {
            "cv_folds": folds,
            "cv_roc_auc_mean": float(np.mean(scores)),
            "cv_roc_auc_std": float(np.std(scores)),
        }
        logger.info(
            "Cross-validation ROC-AUC = %.4f (+/- %.4f) over %d folds",
            result["cv_roc_auc_mean"],
            result["cv_roc_auc_std"],
            folds,
        )
        return result

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
        if self.pipeline is None:
            raise ModelTrainingError("evaluate() called before train()")

        if pd.api.types.is_string_dtype(y_test):
            y_test = (y_test == "Yes").astype(int)

        probabilities = self.pipeline.predict_proba(x_test)[:, 1]
        return evaluate(
            y_test.to_numpy(), probabilities, self.config.decision_threshold
        )

    def save(
        self, path: Path, metrics: ModelMetrics, extra: dict | None = None
    ) -> Path:
        if self.pipeline is None:
            raise ModelTrainingError("save() called before train()")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "metrics": metrics.to_dict(),
            "metadata": {
                "code_version": __version__,
                "trained_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "sklearn_pipeline_steps": [name for name, _ in self.pipeline.steps],
                "input_features": SETTINGS.all_features,
                "decision_threshold": self.config.decision_threshold,
                "hyperparameters": {
                    "n_estimators": self.config.n_estimators,
                    "max_depth": self.config.max_depth,
                    "min_samples_leaf": self.config.min_samples_leaf,
                    "class_weight": self.config.class_weight,
                    "calibration": "isotonic",
                },
                **(extra or {}),
            },
        }
        joblib.dump(payload, path)
        logger.info("Artefact saved to %s (%.1f KB)", path, path.stat().st_size / 1024)
        return path

    @staticmethod
    def write_metrics_report(
        metrics: ModelMetrics, path: Path, extra: dict | None = None
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**metrics.to_dict(), **(extra or {})}, indent=2))
        logger.info("Metrics report written to %s", path)
