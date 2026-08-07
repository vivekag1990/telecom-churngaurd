"""UNIT TESTS -- data ingestion layer."""

from __future__ import annotations

import pytest

from churnguard.data.ingestion import CsvDataSource
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
