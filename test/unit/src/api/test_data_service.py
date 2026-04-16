"""Unit tests for src/api/services/data_service.py."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.api.services.data_service import DataService


def _make_df(n: int = 100) -> pd.DataFrame:
    import numpy as np

    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"temp": np.random.rand(n), "humidity": np.random.rand(n)}, index=idx)


def _service_with_data(n: int = 100) -> DataService:
    svc = DataService()
    svc._df = _make_df(n)
    svc._df = svc._df.copy()
    svc._df["group"] = "station_1"
    svc._df["time_idx"] = range(len(svc._df))
    return svc


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_data_service_not_ready_before_load() -> None:
    svc = DataService()
    assert svc.is_ready is False


def test_row_count_zero_before_load() -> None:
    svc = DataService()
    assert svc.row_count == 0


def test_date_range_none_before_load() -> None:
    svc = DataService()
    assert svc.date_range is None


# ---------------------------------------------------------------------------
# After loading (injected directly for unit test)
# ---------------------------------------------------------------------------

def test_is_ready_true_after_data_injected() -> None:
    svc = _service_with_data()
    assert svc.is_ready is True


def test_row_count_reflects_injected_data() -> None:
    svc = _service_with_data(50)
    assert svc.row_count == 50


def test_date_range_returns_tuple_of_timestamps() -> None:
    svc = _service_with_data()
    dr = svc.date_range
    assert dr is not None
    assert len(dr) == 2


# ---------------------------------------------------------------------------
# _assert_ready raises RuntimeError when not loaded
# ---------------------------------------------------------------------------

def test_get_context_window_raises_when_not_loaded() -> None:
    svc = DataService()
    with pytest.raises(RuntimeError, match="load"):
        svc.get_context_window(datetime(2024, 1, 2), sequence_length=10)


def test_get_training_slice_raises_when_not_loaded() -> None:
    svc = DataService()
    with pytest.raises(RuntimeError, match="load"):
        svc.get_training_slice(0.8)


# ---------------------------------------------------------------------------
# get_context_window
# ---------------------------------------------------------------------------

def test_get_context_window_returns_correct_length() -> None:
    svc = _service_with_data(100)
    ref = datetime(2024, 1, 5, 0, 0, 0)  # well into the data
    window = svc.get_context_window(ref, sequence_length=24)
    assert len(window) == 24


def test_get_context_window_raises_when_insufficient_data() -> None:
    svc = _service_with_data(10)
    ref = datetime(2024, 1, 1, 5, 0, 0)  # only 5 rows before this date
    with pytest.raises(ValueError):
        svc.get_context_window(ref, sequence_length=50)


# ---------------------------------------------------------------------------
# get_training_slice
# ---------------------------------------------------------------------------

def test_get_training_slice_returns_correct_proportion() -> None:
    svc = _service_with_data(100)
    train = svc.get_training_slice(0.8)
    assert len(train) == 80


# ---------------------------------------------------------------------------
# _read_and_prepare (integration via parquet file)
# ---------------------------------------------------------------------------

def test_read_and_prepare_adds_group_and_time_idx(tmp_path: Path) -> None:
    import numpy as np

    df = pd.DataFrame({"temp": np.random.rand(20)}, index=pd.date_range("2024-01-01", periods=20, freq="h"))
    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(parquet_path)

    svc = DataService()
    result = svc._read_and_prepare(str(parquet_path), ["temp"])

    assert "group" in result.columns
    assert "time_idx" in result.columns
    assert result["group"].iloc[0] == "station_1"
    assert result["time_idx"].iloc[0] == 0


def test_read_and_prepare_warns_on_missing_columns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import numpy as np

    df = pd.DataFrame({"temp": np.random.rand(10)}, index=pd.date_range("2024-01-01", periods=10, freq="h"))
    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(parquet_path)

    svc = DataService()
    result = svc._read_and_prepare(str(parquet_path), ["temp", "nonexistent_col"])

    assert "temp" in result.columns


def test_read_and_prepare_from_directory(tmp_path: Path) -> None:
    import numpy as np

    for i in range(2):
        df = pd.DataFrame(
            {"temp": np.random.rand(10)},
            index=pd.date_range(f"2024-0{i+1}-01", periods=10, freq="h"),
        )
        (tmp_path / f"part_{i}.parquet").parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(tmp_path / f"part_{i}.parquet")

    svc = DataService()
    result = svc._read_and_prepare(str(tmp_path), ["temp"])
    assert len(result) == 20


def test_read_and_prepare_raises_for_empty_directory(tmp_path: Path) -> None:
    svc = DataService()
    with pytest.raises(FileNotFoundError):
        svc._read_and_prepare(str(tmp_path), ["temp"])
