"""Model-quality metrics."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churnguard.config import SETTINGS

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    brier: float
    ece: float
    threshold: float
    n_samples: int
    confusion: list[list[int]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def passes_gates(self) -> tuple[bool, list[str]]:
        gates = SETTINGS.gates
        failures: list[str] = []
        if self.roc_auc < gates.min_roc_auc:
            failures.append(f"roc_auc {self.roc_auc:.3f} < {gates.min_roc_auc}")
        if self.f1 < gates.min_f1:
            failures.append(f"f1 {self.f1:.3f} < {gates.min_f1}")
        if self.brier > gates.max_brier:
            failures.append(f"brier {self.brier:.3f} > {gates.max_brier}")
        return (not failures), failures


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        error += (mask.mean()) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(error)


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean predicted and observed probabilities for populated bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)
    predicted: list[float] = []
    observed: list[float] = []
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if mask.sum() < 5:
            continue
        predicted.append(float(y_prob[mask].mean()))
        observed.append(float(y_true[mask].mean()))
    return np.asarray(predicted), np.asarray(observed)


def evaluate(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float | None = None
) -> ModelMetrics:
    threshold = SETTINGS.model.decision_threshold if threshold is None else threshold
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")

    y_pred = (y_prob >= threshold).astype(int)

    metrics = ModelMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        ece=expected_calibration_error(y_true, y_prob),
        threshold=float(threshold),
        n_samples=int(y_true.size),
        confusion=confusion_matrix(y_true, y_pred).tolist(),
    )

    logger.info(
        "Evaluation @thr=%.2f | AUC=%.4f F1=%.4f acc=%.4f Brier=%.4f ECE=%.4f",
        threshold,
        metrics.roc_auc,
        metrics.f1,
        metrics.accuracy,
        metrics.brier,
        metrics.ece,
    )
    return metrics
