"""Produce the figures and threshold evidence used in the assignment report."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "churnguard-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import (  # noqa: E402
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from churnguard.config import SETTINGS  # noqa: E402
from churnguard.data.ingestion import CsvDataSource, DataIngestor  # noqa: E402
from churnguard.data.validation import population_stability_index  # noqa: E402
from churnguard.models.evaluation import evaluate, reliability_curve  # noqa: E402
from churnguard.models.predictor import ChurnPredictor  # noqa: E402

logging.disable(logging.INFO)
FIGURE_DIR = Path("reports/figures")
INK, ACCENT, WARNING = "#1b2a41", "#2a6f97", "#c1121f"


def save(figure, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Wrote {path}")
    return path


def _confusion_matrix(metrics) -> None:
    matrix = np.array(metrics.confusion)
    figure, axis = plt.subplots(figsize=(4.4, 3.8))
    axis.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(
            column,
            row,
            str(value),
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="white" if value > matrix.max() * 0.55 else INK,
        )
    axis.set_xticks([0, 1], ["Pred: Stay", "Pred: Churn"])
    axis.set_yticks([0, 1], ["True: Stay", "True: Churn"])
    axis.set_title(f"Confusion Matrix (threshold = {metrics.threshold})")
    save(figure, "confusion_matrix.png")


def _roc_curve(y_true: np.ndarray, probabilities: np.ndarray, metrics) -> None:
    false_positive, true_positive, _ = roc_curve(y_true, probabilities)
    figure, axis = plt.subplots(figsize=(4.6, 3.9))
    axis.plot(
        false_positive,
        true_positive,
        color=ACCENT,
        lw=2.2,
        label=f"ChurnGuard (AUC = {metrics.roc_auc:.3f})",
    )
    axis.plot(
        [0, 1],
        [0, 1],
        "--",
        color="#94a3b8",
        lw=1.2,
        label="Random (AUC = 0.500)",
    )
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC Curve - held-out test set")
    axis.legend(loc="lower right", frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    save(figure, "roc_curve.png")


def _calibration(y_true: np.ndarray, probabilities: np.ndarray, metrics) -> None:
    predicted, observed = reliability_curve(y_true, probabilities, n_bins=10)
    figure, axis = plt.subplots(figsize=(4.6, 3.9))
    axis.plot(
        [0, 1],
        [0, 1],
        "--",
        color="#94a3b8",
        lw=1.2,
        label="Perfect calibration",
    )
    axis.plot(
        predicted,
        observed,
        "o-",
        color=ACCENT,
        lw=2,
        ms=6,
        label="ChurnGuard (isotonic)",
    )
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed churn rate")
    axis.set_title(f"Reliability Diagram (ECE = {metrics.ece:.4f})")
    axis.legend(loc="upper left", frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    save(figure, "calibration_curve.png")


def _threshold_sweep(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.10, 0.86, 0.01):
        predicted = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision_score(y_true, predicted, zero_division=0),
                "recall": recall_score(y_true, predicted, zero_division=0),
                "f1": f1_score(y_true, predicted, zero_division=0),
            }
        )
    sweep = pd.DataFrame(rows)
    sweep.to_csv("reports/threshold_sweep.csv", index=False)
    figure, axis = plt.subplots(figsize=(5.6, 3.9))
    axis.plot(sweep.threshold, sweep.precision, label="Precision")
    axis.plot(sweep.threshold, sweep.recall, label="Recall")
    axis.plot(sweep.threshold, sweep.f1, color=WARNING, lw=2.4, label="F1")
    axis.axvline(SETTINGS.model.decision_threshold, color=INK, ls=":")
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Score")
    axis.set_title("Threshold selection: precision / recall trade-off")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    save(figure, "threshold_sweep.png")
    return sweep


def _drift_figure(split) -> tuple[float, float]:
    reference = split.x_train["tenure_months"].to_numpy()
    drifted = np.clip(reference * 0.35, 0, None)
    stable_psi = population_stability_index(reference, split.x_test["tenure_months"].to_numpy())
    drifted_psi = population_stability_index(reference, drifted)
    figure, axis = plt.subplots(figsize=(5.8, 3.7))
    bins = np.linspace(0, 72, 25)
    axis.hist(reference, bins=bins, alpha=0.65, color=ACCENT, label="Reference")
    axis.hist(drifted, bins=bins, alpha=0.6, color=WARNING, label="Drifted")
    axis.set_xlabel("tenure_months")
    axis.set_ylabel("Customers")
    axis.set_title(f"Drift: PSI(stable)={stable_psi:.3f} | PSI(drifted)={drifted_psi:.3f}")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    save(figure, "drift_psi.png")
    return stable_psi, drifted_psi


def _risk_bands(predictor: ChurnPredictor, split) -> None:
    counts = pd.Series(
        [prediction.risk_band for prediction in predictor.predict(split.x_test)]
    ).value_counts()
    order = [band for band in ["LOW", "MEDIUM", "HIGH", "CRITICAL"] if band in counts.index]
    colours = {
        "LOW": "#0f766e",
        "MEDIUM": "#eab308",
        "HIGH": "#ea580c",
        "CRITICAL": WARNING,
    }
    figure, axis = plt.subplots(figsize=(5.0, 3.5))
    axis.bar(
        order,
        [counts[band] for band in order],
        color=[colours[band] for band in order],
        width=0.6,
    )
    for index, band in enumerate(order):
        axis.text(index, counts[band] + 6, str(counts[band]), ha="center")
    axis.set_ylabel("Customers in test set")
    axis.set_title("Retention workload by risk band")
    axis.spines[["top", "right"]].set_visible(False)
    save(figure, "risk_bands.png")


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#8894a3"})
    split = DataIngestor(CsvDataSource(SETTINGS.paths.raw_data)).split()
    predictor = ChurnPredictor().load()
    probabilities = predictor.predict_proba(split.x_test)
    y_true = split.y_test.to_numpy()
    metrics = evaluate(y_true, probabilities)

    _confusion_matrix(metrics)
    _roc_curve(y_true, probabilities, metrics)
    _calibration(y_true, probabilities, metrics)
    sweep = _threshold_sweep(y_true, probabilities)
    stable_psi, drifted_psi = _drift_figure(split)
    _risk_bands(predictor, split)
    print(f"PSI stable={stable_psi:.4f} drifted={drifted_psi:.4f}")
    print(f"Best F1 threshold={sweep.loc[sweep.f1.idxmax(), 'threshold']}")


if __name__ == "__main__":
    main()
