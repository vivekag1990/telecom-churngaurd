"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from churnguard.config import SETTINGS
from churnguard.data.ingestion import DataIngestor, InMemoryDataSource
from churnguard.logging_config import configure_logging
from churnguard.models.trainer import ChurnModelTrainer


@pytest.fixture(scope="session", autouse=True)
def _logging():
    configure_logging(level="WARNING")


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    df = pd.read_csv(SETTINGS.paths.raw_data)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Return 2,500 rows to make tests fast
    return df.head(2500).copy()


@pytest.fixture(scope="session")
def split(raw_frame):
    return DataIngestor(InMemoryDataSource(raw_frame, "pytest")).split()


@pytest.fixture(scope="session")
def trained_trainer(split):
    trainer = ChurnModelTrainer()
    trainer.train(split.x_train, split.y_train)
    return trainer


@pytest.fixture(scope="session")
def trained_pipeline(trained_trainer):
    return trained_trainer.pipeline


@pytest.fixture(scope="session")
def artifact_path(tmp_path_factory, trained_trainer, split) -> Path:
    path = tmp_path_factory.mktemp("artifacts") / "model.joblib"
    metrics = trained_trainer.evaluate(split.x_test, split.y_test)
    trained_trainer.save(path, metrics)
    return path


@pytest.fixture
def valid_payload() -> dict:
    return {
        "customerID": "CUST-000001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 88.4,
        "TotalCharges": 265.2,
    }


@pytest.fixture
def feature_frame(raw_frame) -> pd.DataFrame:
    return raw_frame[SETTINGS.all_features].head(50).copy()
