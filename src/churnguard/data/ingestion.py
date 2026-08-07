"""Data ingestion layer.

Design: ``DataSource`` is an abstract interface, so the rest of the system depends
on the *abstraction* rather than on ``pandas.read_csv``. Swapping the CSV for a
Snowflake table or a Parquet file on S3 later means adding one subclass -- no
change to the trainer, the feature pipeline, or the tests (Open/Closed Principle).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churnguard.config import SETTINGS, ModelConfig
from churnguard.exceptions import DataIngestionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataSplit:
    """The four arrays every downstream component needs, kept together."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    def summary(self) -> dict[str, object]:
        train_rate = (
            self.y_train.mean()
            if pd.api.types.is_numeric_dtype(self.y_train)
            else (self.y_train == "Yes").mean()
        )
        test_rate = (
            self.y_test.mean()
            if pd.api.types.is_numeric_dtype(self.y_test)
            else (self.y_test == "Yes").mean()
        )
        return {
            "n_train": int(len(self.x_train)),
            "n_test": int(len(self.x_test)),
            "n_features": int(self.x_train.shape[1]),
            "train_churn_rate": round(float(train_rate), 4),
            "test_churn_rate": round(float(test_rate), 4),
        }


class DataSource(ABC):
    """Abstract read-only data source."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return the raw dataset, or raise ``DataIngestionError``."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier used in logs and lineage metadata."""


class CsvDataSource(DataSource):
    """Reads a dataset from a local (or mounted) CSV file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @property
    def name(self) -> str:
        return f"csv://{self.path}"

    def load(self) -> pd.DataFrame:
        logger.info("Loading data from %s", self.name)
        if not self.path.exists():
            logger.error("Data file not found: %s", self.path)
            raise DataIngestionError(f"Data file not found: {self.path}")
        try:
            frame = pd.read_csv(self.path)
        except pd.errors.EmptyDataError as exc:
            logger.error("Data file %s is empty", self.path)
            raise DataIngestionError(f"Data file is empty: {self.path}") from exc
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            logger.error("Malformed CSV at %s: %s", self.path, exc)
            raise DataIngestionError(f"Could not parse CSV: {self.path}") from exc

        if frame.empty:
            logger.error("Data file %s parsed to zero rows", self.path)
            raise DataIngestionError(f"No rows found in {self.path}")

        logger.info("Loaded %d rows x %d columns", *frame.shape)
        return frame


class InMemoryDataSource(DataSource):
    """Wraps an existing DataFrame -- used by tests and by batch scoring jobs."""

    def __init__(self, frame: pd.DataFrame, label: str = "in-memory") -> None:
        self._frame = frame
        self._label = label

    @property
    def name(self) -> str:
        return f"memory://{self._label}"

    def load(self) -> pd.DataFrame:
        logger.info("Using %s (%d rows)", self.name, len(self._frame))
        return self._frame.copy()


class DataIngestor:
    """Orchestrates load -> sanity-check -> stratified split."""

    def __init__(self, source: DataSource, config: ModelConfig | None = None) -> None:
        self.source = source
        self.config = config or SETTINGS.model

    def load_raw(self) -> pd.DataFrame:
        frame = self.source.load()
        if "TotalCharges" in frame.columns:
            frame["TotalCharges"] = pd.to_numeric(
                frame["TotalCharges"], errors="coerce"
            )

        missing = [
            c
            for c in (*SETTINGS.all_features, self.config.target)
            if c not in frame.columns
        ]
        if missing:
            logger.error(
                "Source %s is missing required columns: %s", self.source.name, missing
            )
            raise DataIngestionError(f"Missing required columns: {missing}")

        duplicates = int(frame.duplicated(subset=[self.config.id_column]).sum())
        if duplicates:
            logger.warning(
                "Dropping %d duplicate %s rows", duplicates, self.config.id_column
            )
            frame = frame.drop_duplicates(subset=[self.config.id_column], keep="first")
        return frame

    def split(self, frame: pd.DataFrame | None = None) -> DataSplit:
        frame = self.load_raw() if frame is None else frame
        target = frame[self.config.target]

        if target.nunique() < 2:
            logger.error(
                "Target '%s' has a single class -- cannot train", self.config.target
            )
            raise DataIngestionError("Target column must contain at least two classes")

        minority = int(target.value_counts().min())
        stratify = target if minority >= 2 else None
        if stratify is None:
            logger.warning(
                "Minority class has %d row(s); stratification disabled", minority
            )

        x_train, x_test, y_train, y_test = train_test_split(
            frame[SETTINGS.all_features],
            target,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify,
        )
        split = DataSplit(x_train, x_test, y_train, y_test)
        logger.info("Split complete: %s", split.summary())
        return split
