"""Shared pytest fixtures for the ChurnGuard test suite."""

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
from generate_data import build_dataset


@pytest.fixture(scope="session", autouse=True)
def _logging():
    configure_logging(level="WARNING")


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    """Return a deterministic, statistically meaningful test dataset."""
    return build_dataset(n_rows=2500, seed=7)


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
    """Persist the session model so tests exercise real serialisation."""
    path = tmp_path_factory.mktemp("artifacts") / "model.joblib"
    metrics = trained_trainer.evaluate(split.x_test, split.y_test)
    trained_trainer.save(path, metrics)
    return path


@pytest.fixture
def valid_payload() -> dict:
    return {
        "customer_id": "CUST-000001",
        "tenure_months": 3,
        "monthly_charges": 88.4,
        "total_charges": 265.2,
        "support_tickets_6m": 4,
        "avg_monthly_gb": 41.5,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
        "tech_support": "No",
        "paperless_billing": "Yes",
    }


@pytest.fixture
def feature_frame(raw_frame) -> pd.DataFrame:
    return raw_frame[SETTINGS.all_features].head(50).copy()
