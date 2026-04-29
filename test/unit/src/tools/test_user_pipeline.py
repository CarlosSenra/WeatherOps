from __future__ import annotations

from pathlib import Path

import pytest

from src.tools import user_pipeline as pp


def test_detect_device_cpu_cuda_explicit() -> None:
    assert pp._detect_device("cpu") == "cpu"
    assert pp._detect_device("cuda") == "cuda"


def test_detect_device_auto_falls_back_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pp, "_has_cmd", lambda name: False)
    assert pp._detect_device("auto") == "cpu"


def test_detect_device_auto_selects_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pp, "_has_cmd", lambda name: name == "nvidia-smi")
    monkeypatch.setattr(pp, "_run", lambda cmd, cwd=None: None)
    assert pp._detect_device("auto") == "cuda"


def test_set_year_range_writes_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "inmet_scraping.yml"
    monkeypatch.setattr(pp, "INMET_CONFIG", cfg)
    pp._set_year_range(2024, 2026)
    content = cfg.read_text(encoding="utf-8")
    assert "start_year: 2024" in content
    assert "end_year: 2026" in content


def test_ensure_model_artifacts_ok(tmp_path: Path) -> None:
    root = tmp_path / "ml_models"
    for exp in ("weather_forecasting_h72", "weather_forecasting_h168"):
        model_dir = root / exp
        (model_dir / "data").mkdir(parents=True)
        (model_dir / "MLmodel").write_text("x", encoding="utf-8")
        (model_dir / "manifest.json").write_text("{}", encoding="utf-8")
    pp._ensure_model_artifacts(root, ["weather_forecasting_h72", "weather_forecasting_h168"])


def test_ensure_model_artifacts_raises_when_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "ml_models"
    model_dir = root / "weather_forecasting_h72"
    model_dir.mkdir(parents=True)
    (model_dir / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Export incompleto"):
        pp._ensure_model_artifacts(root, ["weather_forecasting_h72"])


def test_main_bootstrap_uses_existing_models_without_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pp, "_validate_prereqs", lambda: None)
    monkeypatch.setattr(pp, "_set_year_range", lambda start, end: None)
    monkeypatch.setattr(pp, "_detect_device", lambda device: "cpu")
    monkeypatch.setattr(pp, "_airflow_up", lambda: None)
    monkeypatch.setattr(pp, "_run_data_pipeline", lambda: None)
    monkeypatch.setattr(pp, "_ensure_model_artifacts", lambda root, exps: None)
    monkeypatch.setattr(pp, "_up_api", lambda device: None)
    monkeypatch.setattr(pp, "_healthcheck_api", lambda: None)

    called = {"train": False}

    def _mark_training(device: str) -> None:
        called["train"] = True

    monkeypatch.setattr(pp, "_run_training_and_promote", _mark_training)
    exit_code = pp.main(["--mode", "bootstrap", "--skip-api"])
    assert exit_code == 0
    assert called["train"] is False


def test_main_full_triggers_training_when_models_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pp, "_validate_prereqs", lambda: None)
    monkeypatch.setattr(pp, "_set_year_range", lambda start, end: None)
    monkeypatch.setattr(pp, "_detect_device", lambda device: "cpu")
    monkeypatch.setattr(pp, "_airflow_up", lambda: None)
    monkeypatch.setattr(pp, "_run_data_pipeline", lambda: None)
    monkeypatch.setattr(pp, "_has_model_artifacts", lambda root, exps: False)
    monkeypatch.setattr(pp, "_ensure_model_artifacts", lambda root, exps: None)
    monkeypatch.setattr(pp, "_up_api", lambda device: None)
    monkeypatch.setattr(pp, "_healthcheck_api", lambda: None)

    calls = {"count": 0}

    def _run_training_and_promote(device: str) -> None:
        calls["count"] += 1

    monkeypatch.setattr(pp, "_run_training_and_promote", _run_training_and_promote)
    exit_code = pp.main(["--mode", "full", "--skip-api"])
    assert exit_code == 0
    assert calls["count"] == 1

