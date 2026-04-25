from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.ml_workstation.evaluation import mlflow_helpers


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


def test_parse_mlflow_param_supports_common_types() -> None:
    assert mlflow_helpers.parse_mlflow_param(" 12 ") == 12
    assert mlflow_helpers.parse_mlflow_param("3.14") == pytest.approx(3.14)
    assert mlflow_helpers.parse_mlflow_param("true") is True
    assert mlflow_helpers.parse_mlflow_param("False") is False
    assert mlflow_helpers.parse_mlflow_param("['a', 'b']") == ["a", "b"]
    assert mlflow_helpers.parse_mlflow_param("raw_text") == "raw_text"


def test_require_param_raises_for_missing_key() -> None:
    with pytest.raises(KeyError):
        mlflow_helpers.require_param({}, "data.parquet_path")


def test_build_data_config_validates_lists() -> None:
    params = {
        "data.parquet_path": "data/spec",
        "data.feature_columns": "['f1', 'f2']",
        "data.target_columns": "['t1']",
        "data.sequence_length": "24",
        "data.horizon": "3",
        "data.train_ratio": "0.8",
        "data.val_ratio": "0.1",
    }

    config = mlflow_helpers.build_data_config(params)

    assert config.feature_columns == ["f1", "f2"]
    assert config.target_columns == ["t1"]
    assert config.horizon == 3

    invalid = {**params, "data.feature_columns": "'not-a-list'"}
    with pytest.raises(ValueError):
        mlflow_helpers.build_data_config(invalid)


def test_build_model_config_uses_defaults() -> None:
    params = {
        "model.model_type": "lstm",
        "model.hidden_size": "32",
        "model.num_layers": "2",
        "model.dropout": "0.1",
    }

    cfg = mlflow_helpers.build_model_config(params)

    assert cfg.num_heads == 4
    assert cfg.ffn_dim == 256


def test_resolve_parquet_path_prefers_local_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec_dir = tmp_path / "data" / "spec"
    spec_dir.mkdir(parents=True)

    monkeypatch.setattr(mlflow_helpers, "workspace_root", lambda: tmp_path)

    resolved = mlflow_helpers.resolve_parquet_path("/app/data/spec")

    assert resolved == str(spec_dir)


def test_resolve_parquet_path_resolves_municipio_subpath(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec_municipio = tmp_path / "data" / "spec" / "salvador"
    spec_municipio.mkdir(parents=True)

    monkeypatch.setattr(mlflow_helpers, "workspace_root", lambda: tmp_path)

    resolved = mlflow_helpers.resolve_parquet_path("/app/data/spec/salvador")

    assert resolved == str(spec_municipio)


def test_load_model_with_fallback_uses_runs_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    run = SimpleNamespace(info=SimpleNamespace(run_id="abc123", experiment_id="1"))
    model = DummyModel()

    monkeypatch.setattr(mlflow_helpers.mlflow.pytorch, "load_model", lambda *args, **kwargs: model)

    loaded, uri = mlflow_helpers.load_model_with_fallback(
        run=run,
        params={},
        data_config=SimpleNamespace(feature_columns=["f1"], target_columns=["t1"], horizon=1),
        device=torch.device("cpu"),
    )

    assert loaded is model
    assert uri == "runs:/abc123/model"


def test_load_model_with_fallback_uses_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = SimpleNamespace(info=SimpleNamespace(run_id="run42", experiment_id="exp7"))
    params = {
        "model.model_type": "lstm",
        "model.hidden_size": "8",
        "model.num_layers": "1",
        "model.dropout": "0.0",
    }
    checkpoint = (
        tmp_path
        / "src"
        / "ml_workstation"
        / "mlruns"
        / "exp7"
        / "run42"
        / "artifacts"
        / "best_model.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    model = DummyModel()

    def _raise(*args, **kwargs):
        raise RuntimeError("no artifact")

    monkeypatch.setattr(mlflow_helpers.mlflow.pytorch, "load_model", _raise)
    monkeypatch.setattr(mlflow_helpers, "workspace_root", lambda: tmp_path)
    monkeypatch.setattr(mlflow_helpers, "build_model", lambda **kwargs: model)
    monkeypatch.setattr(mlflow_helpers.torch, "load", lambda *args, **kwargs: model.state_dict())

    loaded, uri = mlflow_helpers.load_model_with_fallback(
        run=run,
        params=params,
        data_config=SimpleNamespace(feature_columns=["f1"], target_columns=["t1"], horizon=1),
        device=torch.device("cpu"),
    )

    assert loaded is model
    assert uri.endswith("best_model.pt")
