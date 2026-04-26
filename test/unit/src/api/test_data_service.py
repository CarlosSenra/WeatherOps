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


def test_read_and_prepare_removes_infinite_values(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {"temp": [1.0, float("inf"), 2.0, float("-inf")]},
        index=pd.date_range("2024-01-01", periods=4, freq="h"),
    )
    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(parquet_path)

    svc = DataService()
    result = svc._read_and_prepare(str(parquet_path), ["temp"])
    assert len(result) == 2


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


def test_read_and_prepare_from_nested_directories(tmp_path: Path) -> None:
    import numpy as np

    nested = tmp_path / "salvador"
    nested.mkdir(parents=True)
    df = pd.DataFrame(
        {"temp": np.random.rand(8)},
        index=pd.date_range("2024-02-01", periods=8, freq="h"),
    )
    df.to_parquet(nested / "dados.parquet")

    svc = DataService()
    result = svc._read_and_prepare(str(tmp_path), ["temp"])
    assert len(result) == 8


def test_read_and_prepare_falls_back_to_model_root_serving_data(tmp_path: Path) -> None:
    import numpy as np

    parquet_root = tmp_path / "parquet_root"
    parquet_root.mkdir(parents=True)
    model_root = tmp_path / "ml_models" / "weather_forecasting_h72" / "serving_data"
    model_root.mkdir(parents=True)
    df = pd.DataFrame(
        {"temp": np.random.rand(6)},
        index=pd.date_range("2024-03-01", periods=6, freq="h"),
    )
    df.to_parquet(model_root / "dados.parquet")

    svc = DataService()
    result = svc._read_and_prepare(str(parquet_root), ["temp"], str(tmp_path / "ml_models"))
    assert len(result) == 6


def test_read_and_prepare_prioritizes_model_root_over_recursive_spec(tmp_path: Path) -> None:
    import numpy as np

    spec_dir = tmp_path / "spec" / "city_a"
    spec_dir.mkdir(parents=True)
    pd.DataFrame(
        {"temp": np.random.rand(4)},
        index=pd.date_range("2024-01-01", periods=4, freq="h"),
    ).to_parquet(spec_dir / "dados.parquet")

    serving_dir = tmp_path / "ml_models" / "weather_forecasting_h72" / "serving_data"
    serving_dir.mkdir(parents=True)
    pd.DataFrame(
        {"temp": np.random.rand(7)},
        index=pd.date_range("2024-02-01", periods=7, freq="h"),
    ).to_parquet(serving_dir / "dados.parquet")

    svc = DataService()
    result = svc._read_and_prepare(str(tmp_path / "spec"), ["temp"], str(tmp_path / "ml_models"))
    assert len(result) == 7


def test_read_and_prepare_raises_for_empty_directory(tmp_path: Path) -> None:
    svc = DataService()
    with pytest.raises(FileNotFoundError):
        svc._read_and_prepare(str(tmp_path), ["temp"])


# ---------------------------------------------------------------------------
# get_actual_values
# ---------------------------------------------------------------------------


def _service_with_temp_data(n: int = 200) -> DataService:
    """DataService com coluna temp_ar_c indexada por hora."""
    import numpy as np

    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    svc = DataService()
    df = pd.DataFrame({"temp_ar_c": 20.0 + np.arange(n, dtype=float)}, index=idx)
    df["group"] = "station_1"
    df["time_idx"] = range(n)
    svc._df = df
    return svc


def test_get_actual_values_raises_when_not_loaded() -> None:
    svc = DataService()
    with pytest.raises(RuntimeError, match="load"):
        svc.get_actual_values(datetime(2024, 1, 2), start_offset_h=1, end_offset_h=24)


def test_get_actual_values_returns_series_for_valid_window() -> None:
    svc = _service_with_temp_data(200)
    result = svc.get_actual_values(
        reference_date=datetime(2024, 1, 2, 0, 0, 0),
        start_offset_h=1,
        end_offset_h=24,
    )
    assert result is not None
    assert len(result) == 24


def test_get_actual_values_returns_none_when_window_beyond_data() -> None:
    svc = _service_with_temp_data(10)
    # Todos os dados terminam cedo; janela está no futuro distante
    result = svc.get_actual_values(
        reference_date=datetime(2030, 1, 1, 0, 0, 0),
        start_offset_h=1,
        end_offset_h=24,
    )
    assert result is None


def test_get_actual_values_returns_none_for_missing_column() -> None:
    svc = _service_with_temp_data(100)
    result = svc.get_actual_values(
        reference_date=datetime(2024, 1, 2, 0, 0, 0),
        start_offset_h=1,
        end_offset_h=24,
        target_column="coluna_inexistente",
    )
    assert result is None


def test_get_actual_values_respects_offset_boundaries() -> None:
    svc = _service_with_temp_data(200)
    # start=1h, end=6h → deve retornar 6 pontos (horas 1 a 6 após reference_date)
    result = svc.get_actual_values(
        reference_date=datetime(2024, 1, 5, 0, 0, 0),
        start_offset_h=1,
        end_offset_h=6,
    )
    assert result is not None
    assert len(result) == 6


def test_get_actual_values_excludes_reference_date_itself() -> None:
    """O ponto em reference_date não deve entrar na janela (start_offset_h >= 1)."""
    svc = _service_with_temp_data(200)
    result = svc.get_actual_values(
        reference_date=datetime(2024, 1, 5, 0, 0, 0),
        start_offset_h=1,
        end_offset_h=1,
    )
    assert result is not None
    assert len(result) == 1
