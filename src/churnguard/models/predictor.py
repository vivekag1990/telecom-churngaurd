"""Inference service."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churnguard.config import SETTINGS
from churnguard.exceptions import InferenceError, ModelNotLoadedError

logger = logging.getLogger(__name__)

RISK_BANDS: tuple[tuple[float, str, str], ...] = (
    (0.30, "LOW", "No action - monitor quarterly"),
    (0.60, "MEDIUM", "Add to nurture campaign; offer plan review"),
    (0.85, "HIGH", "Assign retention agent within 7 days"),
    (1.01, "CRITICAL", "Immediate outreach with targeted discount offer"),
)


@dataclass(frozen=True)
class Prediction:
    customer_id: str | None
    churn_probability: float
    churn_prediction: int
    risk_band: str
    recommended_action: str
    threshold_used: float
    model_version: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assign_risk_band(probability: float) -> tuple[str, str]:
    for upper, band, action in RISK_BANDS:
        if probability < upper:
            return band, action
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]


class ChurnPredictor:
    def __init__(self, artifact_path: Path | None = None, threshold: float | None = None) -> None:
        self.artifact_path = artifact_path or SETTINGS.paths.model_artifact
        self.threshold = SETTINGS.model.decision_threshold if threshold is None else threshold
        self._pipeline = None
        self._metadata: dict = {}
        self._metrics: dict = {}

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    @property
    def metrics(self) -> dict:
        return dict(self._metrics)

    @property
    def model_version(self) -> str:
        return str(self._metadata.get("code_version", "unknown"))

    def load(self) -> ChurnPredictor:
        logger.info("Loading model artefact from %s", self.artifact_path)
        if not Path(self.artifact_path).exists():
            logger.error("Model artefact not found at %s", self.artifact_path)
            raise ModelNotLoadedError(f"Model artefact not found: {self.artifact_path}")
        try:
            payload = joblib.load(self.artifact_path)
        except (OSError, EOFError, KeyError, ValueError) as exc:
            logger.error("Corrupt model artefact %s: %s", self.artifact_path, exc)
            raise ModelNotLoadedError(f"Could not deserialise artefact: {exc}") from exc

        self._pipeline = payload["pipeline"]
        self._metadata = payload.get("metadata", {})
        self._metrics = payload.get("metrics", {})
        logger.info(
            "Model v%s loaded (trained %s)",
            self.model_version,
            self._metadata.get("trained_at_utc", "unknown"),
        )
        return self

    def _require_model(self):
        if self._pipeline is None:
            raise ModelNotLoadedError("Model is not loaded. Call load() first.")
        return self._pipeline

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        pipeline = self._require_model()
        if frame.empty:
            raise InferenceError("Received an empty batch")
        missing = [c for c in SETTINGS.all_features if c not in frame.columns]
        if missing:
            logger.error("Inference payload missing features: %s", missing)
            raise InferenceError(f"Missing required features: {missing}")
        try:
            probabilities = pipeline.predict_proba(frame[SETTINGS.all_features])[:, 1]
        except Exception as exc:
            logger.exception("Inference failed for a batch of %d rows", len(frame))
            raise InferenceError(f"Inference failed: {exc}") from exc

        if not np.all((probabilities >= 0.0) & (probabilities <= 1.0)):
            logger.error("Model returned out-of-range probabilities -- refusing to serve")
            raise InferenceError("Model produced probabilities outside [0, 1]")
        return probabilities

    def predict(self, records: pd.DataFrame | list[dict]) -> list[Prediction]:
        frame = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records
        probabilities = self.predict_proba(frame)
        ids = (
            frame[SETTINGS.model.id_column].astype(str).tolist()
            if SETTINGS.model.id_column in frame.columns
            else [None] * len(frame)
        )
        predictions = []
        for customer_id, probability in zip(ids, probabilities, strict=True):
            band, action = assign_risk_band(float(probability))
            predictions.append(
                Prediction(
                    customer_id=customer_id,
                    churn_probability=round(float(probability), 4),
                    churn_prediction=int(probability >= self.threshold),
                    risk_band=band,
                    recommended_action=action,
                    threshold_used=self.threshold,
                    model_version=self.model_version,
                )
            )
        logger.info(
            "Scored %d record(s) | mean P(churn)=%.4f | flagged=%d",
            len(predictions),
            float(np.mean(probabilities)),
            sum(p.churn_prediction for p in predictions),
        )
        return predictions

    def predict_one(self, record: dict) -> Prediction:
        return self.predict([record])[0]
