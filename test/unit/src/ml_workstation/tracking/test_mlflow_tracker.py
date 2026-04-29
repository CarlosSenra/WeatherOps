"""Unit tests for src/ml_workstation/tracking/mlflow_tracker.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ml_workstation.config.training_config import TrainingConfig
from src.ml_workstation.tracking.mlflow_tracker import (
    MLflowTracker,
    _flatten_dict,
    _read_email_from_environment,
    _resolve_git_sha,
    _resolve_training_data_version,
)


# ---------------------------------------------------------------------------
# _flatten_dict
# ---------------------------------------------------------------------------

def test_flatten_dict_simple_values() -> None:
    result = _flatten_dict({"a": 1, "b": "hello"})
    assert result == {"a": "1", "b": "hello"}


def test_flatten_dict_nested_one_level() -> None:
    result = _flatten_dict({"data": {"sequence_length": 24, "horizon": 10}})
    assert result["data.sequence_length"] == "24"
    assert result["data.horizon"] == "10"


def test_flatten_dict_deeply_nested() -> None:
    result = _flatten_dict({"a": {"b": {"c": "deep"}}})
    assert result["a.b.c"] == "deep"


def test_flatten_dict_empty() -> None:
    assert _flatten_dict({}) == {}


def test_flatten_dict_mixed_types() -> None:
    result = _flatten_dict({"x": 1.5, "y": True, "z": None})
    assert result["x"] == "1.5"
    assert result["y"] == "True"
    assert result["z"] == "None"


# ---------------------------------------------------------------------------
# _resolve_git_sha
# ---------------------------------------------------------------------------

def test_resolve_git_sha_from_config_value() -> None:
    assert _resolve_git_sha("abc123def") == "abc123def"


def test_resolve_git_sha_strips_config_whitespace() -> None:
    assert _resolve_git_sha("  sha456  ") == "sha456"


def test_resolve_git_sha_from_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_SHA", "envsha789")
    result = _resolve_git_sha(None)
    assert result == "envsha789"


def test_resolve_git_sha_falls_back_to_unknown_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GIT_SHA", raising=False)
    with patch("src.ml_workstation.tracking.mlflow_tracker._read_git_sha", return_value=""):
        result = _resolve_git_sha(None)
    assert result == "unknown"


def test_resolve_git_sha_uses_runtime_git_when_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GIT_SHA", raising=False)
    with patch("src.ml_workstation.tracking.mlflow_tracker._read_git_sha", return_value="runtimesha"):
        result = _resolve_git_sha(None)
    assert result == "runtimesha"


# ---------------------------------------------------------------------------
# _resolve_training_data_version
# ---------------------------------------------------------------------------

def test_resolve_data_version_from_config() -> None:
    assert _resolve_training_data_version("v1.2.3") == "v1.2.3"


def test_resolve_data_version_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAINING_DATA_VERSION", "env_v2")
    assert _resolve_training_data_version(None) == "env_v2"


def test_resolve_data_version_falls_back_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRAINING_DATA_VERSION", raising=False)
    with patch("src.ml_workstation.tracking.mlflow_tracker._read_data_dvc_hash", return_value=None):
        result = _resolve_training_data_version(None)
    assert result == "unknown"


# ---------------------------------------------------------------------------
# _read_email_from_environment
# ---------------------------------------------------------------------------

def test_read_email_from_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMAIL", "test@example.com")
    assert _read_email_from_environment() == "test@example.com"


def test_read_email_strips_quotes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMAIL", '"user@corp.com"')
    assert _read_email_from_environment() == "user@corp.com"


def test_read_email_returns_none_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMAIL", raising=False)
    # Patch _workspace_root to avoid reading real .env file
    with patch(
        "src.ml_workstation.tracking.mlflow_tracker._workspace_root",
        return_value=Path("/nonexistent_dir_xyz"),
    ):
        result = _read_email_from_environment()
    assert result is None


# ---------------------------------------------------------------------------
# MLflowTracker
# ---------------------------------------------------------------------------

def test_tracker_init_sets_mlflow_experiment() -> None:
    config = TrainingConfig(experiment_name="test_exp", run_name="test_run")
    with patch("src.ml_workstation.tracking.mlflow_tracker.mlflow") as mock_mlflow:
        MLflowTracker(config)
        mock_mlflow.set_experiment.assert_called_once_with("test_exp")


def test_tracker_run_id_is_none_before_start_run() -> None:
    config = TrainingConfig(experiment_name="exp")
    with patch("src.ml_workstation.tracking.mlflow_tracker.mlflow"):
        tracker = MLflowTracker(config)
        assert tracker.run_id is None


def test_tracker_start_run_logs_params_and_tags() -> None:
    config = TrainingConfig(experiment_name="exp", run_name="run1")
    with patch("src.ml_workstation.tracking.mlflow_tracker.mlflow") as mock_mlflow:
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value = mock_run

        with patch("src.ml_workstation.tracking.mlflow_tracker._resolve_git_sha", return_value="sha"):
            with patch("src.ml_workstation.tracking.mlflow_tracker._resolve_training_data_version", return_value="v1"):
                tracker = MLflowTracker(config)
                tracker.start_run()

        mock_mlflow.log_params.assert_called()
        mock_mlflow.set_tags.assert_called()


def test_tracker_run_id_set_after_start_run() -> None:
    config = TrainingConfig(experiment_name="exp", run_name="run1")
    with patch("src.ml_workstation.tracking.mlflow_tracker.mlflow") as mock_mlflow:
        mock_run = MagicMock()
        mock_run.info.run_id = "myrunid"
        mock_mlflow.start_run.return_value = mock_run

        with patch("src.ml_workstation.tracking.mlflow_tracker._resolve_git_sha", return_value="sha"):
            with patch("src.ml_workstation.tracking.mlflow_tracker._resolve_training_data_version", return_value="v1"):
                tracker = MLflowTracker(config)
                tracker.start_run()

        assert tracker.run_id == "myrunid"


def test_tracker_log_epoch_metrics_calls_mlflow() -> None:
    config = TrainingConfig(experiment_name="exp")
    with patch("src.ml_workstation.tracking.mlflow_tracker.mlflow") as mock_mlflow:
        tracker = MLflowTracker(config)
        tracker.log_epoch_metrics({"val_loss": 0.42, "mae": 1.5}, epoch=3)
        mock_mlflow.log_metrics.assert_called_once_with({"val_loss": 0.42, "mae": 1.5}, step=3)


def test_tracker_log_governance_metrics_skips_empty() -> None:
    config = TrainingConfig(experiment_name="exp")
    with patch("src.ml_workstation.tracking.mlflow_tracker.mlflow") as mock_mlflow:
        tracker = MLflowTracker(config)
        tracker.log_governance_metrics({})
        mock_mlflow.log_metrics.assert_not_called()


def test_tracker_log_governance_metrics_logs_numeric_values() -> None:
    config = TrainingConfig(experiment_name="exp")
    with patch("src.ml_workstation.tracking.mlflow_tracker.mlflow") as mock_mlflow:
        tracker = MLflowTracker(config)
        tracker.log_governance_metrics({"mape": 5.2, "rmse": 1.1, "label": "str_ignored"})
        mock_mlflow.log_metrics.assert_called_once()
        logged = mock_mlflow.log_metrics.call_args[0][0]
        assert "governance_metric.mape" in logged
        assert "governance_metric.rmse" in logged
        assert "governance_metric.label" not in logged


def test_tracker_log_artifact_calls_mlflow() -> None:
    config = TrainingConfig(experiment_name="exp")
    with patch("src.ml_workstation.tracking.mlflow_tracker.mlflow") as mock_mlflow:
        tracker = MLflowTracker(config)
        tracker.log_artifact("/tmp/checkpoint.pt")
        mock_mlflow.log_artifact.assert_called_once_with("/tmp/checkpoint.pt")


def test_tracker_end_run_calls_mlflow() -> None:
    config = TrainingConfig(experiment_name="exp", run_name="run1")
    with patch("src.ml_workstation.tracking.mlflow_tracker.mlflow") as mock_mlflow:
        mock_run = MagicMock()
        mock_run.info.run_id = "endrunid"
        mock_mlflow.start_run.return_value = mock_run

        with patch("src.ml_workstation.tracking.mlflow_tracker._resolve_git_sha", return_value="sha"):
            with patch("src.ml_workstation.tracking.mlflow_tracker._resolve_training_data_version", return_value="v1"):
                tracker = MLflowTracker(config)
                tracker.start_run()
                tracker.end_run()

        mock_mlflow.end_run.assert_called_once()
