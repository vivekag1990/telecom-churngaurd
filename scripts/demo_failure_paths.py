"""Ad-hoc demonstration of every domain exception + its log level.

Not part of the package -- run manually to capture console evidence for the
report. Each block is wrapped in a try/except so one failure doesn't stop
the rest of the demo.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from churnguard.config import SETTINGS
from churnguard.data.ingestion import CsvDataSource, DataIngestor, InMemoryDataSource
from churnguard.data.validation import DataValidator
from churnguard.exceptions import ChurnGuardError
from churnguard.logging_config import configure_logging
from churnguard.models.predictor import ChurnPredictor
from churnguard.models.trainer import ChurnModelTrainer

logger = logging.getLogger("churnguard.demo")


def banner(title: str) -> None:
    logger.info("")
    logger.info("### %s ###", title)


def run(title: str, fn) -> None:
    banner(title)
    try:
        fn()
    except ChurnGuardError as exc:
        logger.info("Caught %s: %s", type(exc).__name__, exc)


def main() -> None:
    configure_logging(force=True)
    raw = pd.read_csv(SETTINGS.paths.raw_data)

    # 1. Ingestion -- missing file
    def missing_file():
        DataIngestor(CsvDataSource("data/raw/does_not_exist.csv")).load_raw()

    run("Ingestion: file not found -> DataIngestionError", missing_file)

    # 2. Ingestion -- empty file
    def empty_file():
        empty_path = Path("/tmp/empty_churn.csv")
        empty_path.write_text("")
        DataIngestor(CsvDataSource(empty_path)).load_raw()

    run("Ingestion: empty file -> DataIngestionError", empty_file)

    # 3. Ingestion -- missing required columns
    def missing_columns():
        broken = raw.drop(columns=["contract_type", "internet_service"])
        DataIngestor(InMemoryDataSource(broken, "missing-cols")).load_raw()

    run("Ingestion: missing required columns -> DataIngestionError", missing_columns)

    # 4. Ingestion -- single-class target (cannot split)
    def single_class_target():
        single = raw.copy()
        single["churn"] = 0
        DataIngestor(InMemoryDataSource(single, "single-class")).split(single)

    run("Ingestion: single-class target -> DataIngestionError", single_class_target)

    # 5. Validation -- schema breach (bad category level + out-of-range + nulls)
    def schema_breach():
        broken = raw.copy()
        broken.loc[0, "contract_type"] = "Lifetime"
        broken.loc[1, "monthly_charges"] = -50.0
        broken.loc[2:400, "total_charges"] = None
        report = DataValidator().validate(broken)
        report.raise_for_status()

    run(
        "Validation: schema/range/missing breach -> SchemaValidationError",
        schema_breach,
    )

    # 6. Training -- single-class target
    def training_single_class():
        trainer = ChurnModelTrainer()
        x = raw[SETTINGS.all_features].head(50)
        y = pd.Series([0] * 50)
        trainer.train(x, y)

    run("Training: single-class target -> ModelTrainingError", training_single_class)

    # 7. Training -- X/y length mismatch
    def training_mismatch():
        trainer = ChurnModelTrainer()
        x = raw[SETTINGS.all_features].head(50)
        y = raw["churn"].head(30)
        trainer.train(x, y)

    run("Training: X/y row mismatch -> ModelTrainingError", training_mismatch)

    # 8. Inference -- model not loaded
    def inference_not_loaded():
        predictor = ChurnPredictor(artifact_path=Path("/tmp/never_saved.joblib"))
        predictor.predict_one(raw.iloc[0].to_dict())

    run("Inference: artefact missing -> ModelNotLoadedError", inference_not_loaded)

    # 9. Inference -- missing required features in payload
    def inference_missing_features():
        predictor = ChurnPredictor().load()
        bad_payload = {"customer_id": "X", "tenure_months": 5}
        predictor.predict_one(bad_payload)

    run(
        "Inference: payload missing features -> InferenceError",
        inference_missing_features,
    )

    banner("Demo complete")


if __name__ == "__main__":
    main()
