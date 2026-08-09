"""UNIT TESTS - data ingestion layer."""

from __future__ import annotations

import pandas as pd
import pytest

from churnguard.data.ingestion import (
    CsvDataSource,
    DataIngestor,
    DataSplit,
    InMemoryDataSource,
)
from churnguard.exceptions import DataIngestionError


class TestCsvDataSource:
    def test_reads_a_valid_csv(self, raw_frame, tmp_path):
        path = tmp_path / "data.csv"
        raw_frame.to_csv(path, index=False)
        loaded = CsvDataSource(path).load()
        assert loaded.shape == raw_frame.shape

    def test_missing_file_raises_domain_error(self, tmp_path):
        with pytest.raises(DataIngestionError, match="not found"):
            CsvDataSource(tmp_path / "nope.csv").load()

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        with pytest.raises(DataIngestionError, match="empty"):
            CsvDataSource(path).load()

    def test_header_only_file_raises(self, tmp_path):
        path = tmp_path / "header.csv"
        path.write_text("customer_id,churn\n")
        with pytest.raises(DataIngestionError, match="No rows"):
            CsvDataSource(path).load()

    def test_name_is_traceable(self, tmp_path):
        assert CsvDataSource(tmp_path / "a.csv").name.startswith("csv://")


class TestDataIngestor:
    def test_rejects_frame_missing_required_columns(self, raw_frame):
        broken = raw_frame.drop(columns=["contract_type"])
        ingestor = DataIngestor(InMemoryDataSource(broken))
        with pytest.raises(DataIngestionError, match="Missing required columns"):
            ingestor.load_raw()

    def test_drops_duplicate_customer_ids(self, raw_frame):
        duplicated = pd.concat([raw_frame, raw_frame.head(10)], ignore_index=True)
        result = DataIngestor(InMemoryDataSource(duplicated)).load_raw()
        assert len(result) == len(raw_frame)

    def test_split_sizes_respect_test_size(self, split, raw_frame):
        assert len(split.x_test) == pytest.approx(0.2 * len(raw_frame), abs=2)
        assert len(split.x_train) + len(split.x_test) == len(raw_frame)

    def test_split_is_stratified(self, split):
        assert abs(split.y_train.mean() - split.y_test.mean()) < 0.02

    def test_no_row_leaks_between_train_and_test(self, split):
        assert not set(split.x_train.index) & set(split.x_test.index)

    def test_split_is_reproducible(self, raw_frame):
        first = DataIngestor(InMemoryDataSource(raw_frame)).split()
        second = DataIngestor(InMemoryDataSource(raw_frame)).split()
        assert list(first.x_test.index) == list(second.x_test.index)

    def test_single_class_target_is_rejected(self, raw_frame):
        constant = raw_frame.copy()
        constant["churn"] = 0
        with pytest.raises(DataIngestionError, match="two classes"):
            DataIngestor(InMemoryDataSource(constant)).split()

    def test_summary_reports_expected_keys(self, split):
        assert set(DataSplit.summary(split)) == {
            "n_train",
            "n_test",
            "n_features",
            "train_churn_rate",
            "test_churn_rate",
        }
