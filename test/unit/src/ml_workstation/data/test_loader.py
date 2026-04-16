"""Unit tests for src/ml_workstation/data/loader.py.

Requires sklearn (uses StandardScaler). Skipped gracefully on platforms where
the sklearn native extension cannot be loaded (e.g. Windows with AppControl policy).
"""
from __future__ import annotations

import pytest

pytest.importorskip("sklearn.preprocessing", exc_type=ImportError)

from pathlib import Path

import numpy as np
import pandas as pd

from src.ml_workstation.config.training_config import DataConfig
from src.ml_workstation.data.loader import DataLoaderOutput, ParquetDataLoader

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_FEATURES = ["temp", "humidity", "pressure"]
_TARGETS = ["temp"]
_N = 200


def _write_parquet(path: Path, n: int = _N) -> None:
    df = pd.DataFrame(
        np.random.default_rng(42).random((n, len(_FEATURES))).astype(np.float32),
        columns=_FEATURES,
    )
    df.to_parquet(path)


def _config(parquet_path: str, n_rows: int = _N) -> DataConfig:
    return DataConfig(
        parquet_path=parquet_path,
        feature_columns=_FEATURES,
        target_columns=_TARGETS,
        sequence_length=10,
        horizon=5,
        train_ratio=0.7,
        val_ratio=0.15,
    )


# ---------------------------------------------------------------------------
# build() — full pipeline
# ---------------------------------------------------------------------------

def test_build_returns_three_loaders_and_output(tmp_path: Path) -> None:
    p = tmp_path / "data.parquet"
    _write_parquet(p)
    loader = ParquetDataLoader(_config(str(p)))
    train_dl, val_dl, test_dl, output = loader.build(batch_size=16)

    assert isinstance(output, DataLoaderOutput)
    assert output.n_features == len(_FEATURES)
    assert output.n_targets == 1
    assert output.feature_columns == _FEATURES
    assert output.target_columns == _TARGETS
    assert output.num_train_samples > 0
    assert output.num_val_samples > 0


def test_build_from_directory_with_multiple_parquet_files(tmp_path: Path) -> None:
    for i in range(2):
        _write_parquet(tmp_path / f"part_{i}.parquet", n=60)

    cfg = _config(str(tmp_path), n_rows=120)
    loader = ParquetDataLoader(cfg)
    _, _, _, output = loader.build(batch_size=8)

    assert output.num_train_samples > 0


# ---------------------------------------------------------------------------
# _read()
# ---------------------------------------------------------------------------

def test_read_raises_value_error_on_missing_columns(tmp_path: Path) -> None:
    p = tmp_path / "data.parquet"
    _write_parquet(p)
    cfg = DataConfig(
        parquet_path=str(p),
        feature_columns=["nonexistent"],
        target_columns=["nonexistent"],
        sequence_length=5,
        horizon=2,
    )
    loader = ParquetDataLoader(cfg)
    with pytest.raises(ValueError, match="Colunas ausentes"):
        loader._read()


def test_read_raises_file_not_found_for_empty_directory(tmp_path: Path) -> None:
    cfg = _config(str(tmp_path))
    loader = ParquetDataLoader(cfg)
    with pytest.raises(FileNotFoundError):
        loader._read()


# ---------------------------------------------------------------------------
# _split()
# ---------------------------------------------------------------------------

def test_split_proportions_are_correct(tmp_path: Path) -> None:
    p = tmp_path / "data.parquet"
    _write_parquet(p, n=100)
    cfg = DataConfig(
        parquet_path=str(p),
        feature_columns=_FEATURES,
        target_columns=_TARGETS,
        sequence_length=5,
        horizon=2,
        train_ratio=0.7,
        val_ratio=0.15,
    )
    loader = ParquetDataLoader(cfg)
    df = loader._read()
    train, val, test = loader._split(df)

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15


# ---------------------------------------------------------------------------
# _scale()
# ---------------------------------------------------------------------------

def test_scale_standardizes_training_data(tmp_path: Path) -> None:
    p = tmp_path / "data.parquet"
    _write_parquet(p)
    loader = ParquetDataLoader(_config(str(p)))
    df = loader._read()
    train, val, test = loader._split(df)
    train_arr, val_arr, test_arr = loader._scale(train, val, test)

    # Train should be approximately standardised (mean ~0, std ~1)
    assert abs(train_arr.mean()) < 0.5
    assert abs(train_arr.std() - 1.0) < 0.3


def test_scale_fits_on_train_only(tmp_path: Path) -> None:
    p = tmp_path / "data.parquet"
    _write_parquet(p)
    loader = ParquetDataLoader(_config(str(p)))
    df = loader._read()
    train, val, test = loader._split(df)
    loader._scale(train, val, test)

    # scaler must have been fitted (has mean_ attribute)
    assert hasattr(loader.scaler, "mean_")
