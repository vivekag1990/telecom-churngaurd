"""Training entry point: ingest -> validate -> train -> evaluate -> gate -> persist.

Run: python scripts/train.py --cv 5

Exit code 1 on a data-contract breach or a model that misses its quality gates,
so the same command can be dropped straight into a CI job.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churnguard.config import SETTINGS
from churnguard.data.ingestion import CsvDataSource, DataIngestor
from churnguard.data.validation import (
    DataValidator,
    load_reference_profile,
    write_reference_profile,
)
from churnguard.logging_config import configure_logging
from churnguard.models.trainer import ChurnModelTrainer

logger = logging.getLogger("churnguard.train")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the ChurnGuard model.")
    parser.add_argument("--data", type=Path, default=SETTINGS.paths.raw_data)
    parser.add_argument("--cv", type=int, default=5, help="CV folds; 0 to skip")
    parser.add_argument("--skip-gates", action="store_true", help="train even if gates fail")
    args = parser.parse_args()

    configure_logging()
    logger.info("=" * 78)
    logger.info("ChurnGuard training run starting")
    logger.info("=" * 78)

    # Load and validate the training data.
    ingestor = DataIngestor(CsvDataSource(args.data))
    raw = ingestor.load_raw()

    previous_profile = load_reference_profile(SETTINGS.paths.reference_profile)
    report = DataValidator().validate(raw, reference_profile=previous_profile)
    if not report.passed and not args.skip_gates:
        logger.error("Aborting: dataset failed validation -> %s", report.errors)
        return 1

    split = ingestor.split(raw)

    # Refresh the reference distribution after validation succeeds.
    write_reference_profile(split.x_train, SETTINGS.paths.reference_profile)

    # Train and evaluate the full pipeline.
    trainer = ChurnModelTrainer()
    cv_result = {}
    if args.cv:
        cv_result = trainer.cross_validate(split.x_train, split.y_train, folds=args.cv)
    trainer.train(split.x_train, split.y_train)

    metrics = trainer.evaluate(split.x_test, split.y_test)
    passed, failures = metrics.passes_gates()
    if not passed:
        logger.error("Model failed quality gates: %s", failures)
        if not args.skip_gates:
            return 1
    else:
        logger.info("All model quality gates passed")

    # Save artefacts only after all release gates pass.
    trainer.save(SETTINGS.paths.model_artifact, metrics, extra={**cv_result, **split.summary()})
    trainer.write_metrics_report(
        metrics,
        SETTINGS.paths.metrics_report,
        extra={**cv_result, "data_quality": report.to_dict()},
    )

    logger.info("-" * 78)
    logger.info(
        "DONE | ROC-AUC=%.4f  F1=%.4f  Accuracy=%.4f  Brier=%.4f  ECE=%.4f",
        metrics.roc_auc,
        metrics.f1,
        metrics.accuracy,
        metrics.brier,
        metrics.ece,
    )
    logger.info("-" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
