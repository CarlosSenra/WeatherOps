from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml

import src.ml_workstation.promotion.promote as promote_module
import src.ml_workstation.promotion.loader as loader_module
from src.ml_workstation.promotion.promote import (
    PromotionRejectedError,
    promote_run,
    select_best_run,
)
from src.ml_workstation.promotion.loader import (
    ModelNotInProductionError,
    get_production_info,
    load_production_model,
)
from src.ml_workstation.promotion.run_promote import _build_parser, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run(run_id: str, mape: float, status: str = "FINISHED") -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.data.metrics = {"mape": mape, "val_loss": 0.1}
    run.data.params = {}
    run.info.status = status
    return run


def _write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# select_best_run
# ---------------------------------------------------------------------------

class TestSelectBestRun:
    def test_returns_run_with_lowest_metric(self, monkeypatch: pytest.MonkeyPatch) -> None:
        best = _make_run("run-best", mape=5.0)

        client = MagicMock()
        client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="1")
        client.search_runs.return_value = [best]

        monkeypatch.setattr(promote_module.mlflow, "set_tracking_uri", lambda uri: None)
        monkeypatch.setattr(promote_module, "resolve_tracking_uri", lambda uri: None)

        with patch("src.ml_workstation.promotion.promote.mlflow.MlflowClient", return_value=client):
            result = select_best_run("weather_forecasting_h72", metric="mape")

        assert result.info.run_id == "run-best"
        client.search_runs.assert_called_once()
        call_kwargs = client.search_runs.call_args
        assert call_kwargs.kwargs["order_by"] == ["metrics.mape ASC"]

    def test_raises_when_no_finished_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="1")
        client.search_runs.return_value = []

        monkeypatch.setattr(promote_module, "resolve_tracking_uri", lambda uri: None)

        with patch("src.ml_workstation.promotion.promote.mlflow.MlflowClient", return_value=client):
            with pytest.raises(ValueError, match="Nenhum run finalizado"):
                select_best_run("weather_forecasting_h72")

    def test_raises_when_experiment_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.get_experiment_by_name.return_value = None

        monkeypatch.setattr(promote_module, "resolve_tracking_uri", lambda uri: None)

        with patch("src.ml_workstation.promotion.promote.mlflow.MlflowClient", return_value=client):
            with pytest.raises(ValueError, match="Experimento MLflow não encontrado"):
                select_best_run("unknown_experiment")


# ---------------------------------------------------------------------------
# promote_run
# ---------------------------------------------------------------------------

class TestPromoteRun:
    def _setup_client(self, run: MagicMock) -> MagicMock:
        client = MagicMock()
        client.get_run.return_value = run
        mv = SimpleNamespace(version="3")
        return client, mv

    def test_successful_promotion_updates_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _make_run("run-abc", mape=8.0)
        client, mv = self._setup_client(run)

        monkeypatch.setattr(promote_module, "_PRODUCTION_FILE", tmp_path / "models_production.yaml")
        monkeypatch.setattr(promote_module, "resolve_tracking_uri", lambda uri: None)

        with (
            patch("src.ml_workstation.promotion.promote.mlflow.MlflowClient", return_value=client),
            patch(
                "src.ml_workstation.promotion.promote.mlflow.register_model",
                return_value=SimpleNamespace(version="3"),
            ),
        ):
            version = promote_run(
                run_id="run-abc",
                experiment_name="weather_forecasting_h72",
            )

        assert version == "3"
        client.set_registered_model_alias.assert_called_once_with(
            name="weather_forecasting_h72",
            alias="production",
            version="3",
        )

        saved = yaml.safe_load((tmp_path / "models_production.yaml").read_text())
        entry = saved["experiments"]["weather_forecasting_h72"]
        assert entry["run_id"] == "run-abc"
        assert entry["mape"] == pytest.approx(8.0)
        assert entry["promoted_by"] == "auto"

    def test_rejects_promotion_when_mape_is_worse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_yaml(
            tmp_path / "models_production.yaml",
            {
                "experiments": {
                    "weather_forecasting_h72": {
                        "model_name": "weather_forecasting_h72",
                        "run_id": "run-old",
                        "mape": 5.0,  # atual em produção é mais baixo (melhor)
                        "promoted_at": "2026-01-01",
                        "promoted_by": "auto",
                    }
                }
            },
        )

        run = _make_run("run-bad", mape=10.0)  # candidato é pior
        client = MagicMock()
        client.get_run.return_value = run

        monkeypatch.setattr(promote_module, "_PRODUCTION_FILE", tmp_path / "models_production.yaml")
        monkeypatch.setattr(promote_module, "resolve_tracking_uri", lambda uri: None)

        with patch("src.ml_workstation.promotion.promote.mlflow.MlflowClient", return_value=client):
            with pytest.raises(PromotionRejectedError, match="candidato MAPE=10"):
                promote_run(
                    run_id="run-bad",
                    experiment_name="weather_forecasting_h72",
                )

    def test_force_bypasses_regression_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_yaml(
            tmp_path / "models_production.yaml",
            {
                "experiments": {
                    "weather_forecasting_h72": {
                        "model_name": "weather_forecasting_h72",
                        "run_id": "run-old",
                        "mape": 5.0,
                        "promoted_at": "2026-01-01",
                        "promoted_by": "auto",
                    }
                }
            },
        )

        run = _make_run("run-worse", mape=15.0)
        client = MagicMock()
        client.get_run.return_value = run

        monkeypatch.setattr(promote_module, "_PRODUCTION_FILE", tmp_path / "models_production.yaml")
        monkeypatch.setattr(promote_module, "resolve_tracking_uri", lambda uri: None)

        with (
            patch("src.ml_workstation.promotion.promote.mlflow.MlflowClient", return_value=client),
            patch(
                "src.ml_workstation.promotion.promote.mlflow.register_model",
                return_value=SimpleNamespace(version="4"),
            ),
        ):
            version = promote_run(
                run_id="run-worse",
                experiment_name="weather_forecasting_h72",
                force=True,
            )

        assert version == "4"

    def test_uses_custom_model_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _make_run("run-xyz", mape=3.0)
        client = MagicMock()
        client.get_run.return_value = run

        monkeypatch.setattr(promote_module, "_PRODUCTION_FILE", tmp_path / "models_production.yaml")
        monkeypatch.setattr(promote_module, "resolve_tracking_uri", lambda uri: None)

        with (
            patch("src.ml_workstation.promotion.promote.mlflow.MlflowClient", return_value=client),
            patch(
                "src.ml_workstation.promotion.promote.mlflow.register_model",
                return_value=SimpleNamespace(version="1"),
            ) as mock_register,
        ):
            promote_run(
                run_id="run-xyz",
                experiment_name="weather_forecasting_h72",
                model_name="my_custom_model",
            )

        mock_register.assert_called_once_with(
            model_uri="runs:/run-xyz/model",
            name="my_custom_model",
        )
        saved = yaml.safe_load((tmp_path / "models_production.yaml").read_text())
        assert saved["experiments"]["weather_forecasting_h72"]["model_name"] == "my_custom_model"
        assert saved["experiments"]["weather_forecasting_h72"]["promoted_by"] == "manual"

    def test_horizons_are_independent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Promoção para h72 não deve afetar entrada de h168."""
        run = _make_run("run-h72", mape=6.0)
        client = MagicMock()
        client.get_run.return_value = run

        monkeypatch.setattr(promote_module, "_PRODUCTION_FILE", tmp_path / "models_production.yaml")
        monkeypatch.setattr(promote_module, "resolve_tracking_uri", lambda uri: None)

        with (
            patch("src.ml_workstation.promotion.promote.mlflow.MlflowClient", return_value=client),
            patch(
                "src.ml_workstation.promotion.promote.mlflow.register_model",
                return_value=SimpleNamespace(version="1"),
            ),
        ):
            promote_run(run_id="run-h72", experiment_name="weather_forecasting_h72")

        saved = yaml.safe_load((tmp_path / "models_production.yaml").read_text())
        assert "weather_forecasting_h168" not in saved["experiments"]


# ---------------------------------------------------------------------------
# load_production_model
# ---------------------------------------------------------------------------

class TestLoadProductionModel:
    def test_loads_via_registry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_yaml(
            tmp_path / "models_production.yaml",
            {
                "experiments": {
                    "weather_forecasting_h72": {
                        "model_name": "weather_forecasting_h72",
                        "run_id": "run-ok",
                        "mape": 7.0,
                        "promoted_at": "2026-04-11",
                        "promoted_by": "auto",
                    }
                }
            },
        )

        dummy_model = torch.nn.Linear(4, 2)
        monkeypatch.setattr(loader_module, "_PRODUCTION_FILE", tmp_path / "models_production.yaml")
        monkeypatch.setattr(loader_module, "resolve_tracking_uri", lambda uri: None)
        monkeypatch.setattr(
            loader_module.mlflow.pytorch,
            "load_model",
            lambda uri, map_location=None: dummy_model,
        )

        model = load_production_model("weather_forecasting_h72")
        assert model is dummy_model

    def test_falls_back_to_runs_uri(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_yaml(
            tmp_path / "models_production.yaml",
            {
                "experiments": {
                    "weather_forecasting_h72": {
                        "model_name": "weather_forecasting_h72",
                        "run_id": "run-fallback",
                        "mape": 7.0,
                        "promoted_at": "2026-04-11",
                        "promoted_by": "auto",
                    }
                }
            },
        )

        dummy_model = torch.nn.Linear(4, 2)
        call_uris: list[str] = []

        def _load_side_effect(uri, map_location=None):
            call_uris.append(uri)
            if uri.startswith("models:/"):
                raise RuntimeError("Registry unavailable")
            return dummy_model

        monkeypatch.setattr(loader_module, "_PRODUCTION_FILE", tmp_path / "models_production.yaml")
        monkeypatch.setattr(loader_module, "resolve_tracking_uri", lambda uri: None)
        monkeypatch.setattr(loader_module.mlflow.pytorch, "load_model", _load_side_effect)

        model = load_production_model("weather_forecasting_h72")
        assert model is dummy_model
        assert any(u.startswith("runs:/") for u in call_uris)

    def test_raises_when_yaml_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            loader_module, "_PRODUCTION_FILE", tmp_path / "does_not_exist.yaml"
        )
        with pytest.raises(ModelNotInProductionError, match="models_production.yaml não encontrado"):
            load_production_model("weather_forecasting_h72")

    def test_raises_when_experiment_not_in_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_yaml(tmp_path / "models_production.yaml", {"experiments": {}})
        monkeypatch.setattr(loader_module, "_PRODUCTION_FILE", tmp_path / "models_production.yaml")

        with pytest.raises(ModelNotInProductionError, match="Nenhum modelo em produção"):
            load_production_model("weather_forecasting_h72")

    def test_raises_when_both_sources_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_yaml(
            tmp_path / "models_production.yaml",
            {
                "experiments": {
                    "weather_forecasting_h72": {
                        "model_name": "weather_forecasting_h72",
                        "run_id": "run-broken",
                        "mape": 7.0,
                        "promoted_at": "2026-04-11",
                        "promoted_by": "auto",
                    }
                }
            },
        )

        monkeypatch.setattr(loader_module, "_PRODUCTION_FILE", tmp_path / "models_production.yaml")
        monkeypatch.setattr(loader_module, "resolve_tracking_uri", lambda uri: None)
        monkeypatch.setattr(
            loader_module.mlflow.pytorch,
            "load_model",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("load failed")),
        )

        with pytest.raises(ModelNotInProductionError):
            load_production_model("weather_forecasting_h72")


# ---------------------------------------------------------------------------
# get_production_info
# ---------------------------------------------------------------------------

class TestGetProductionInfo:
    def test_returns_entry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_yaml(
            tmp_path / "models_production.yaml",
            {
                "experiments": {
                    "weather_forecasting_h72": {
                        "model_name": "weather_forecasting_h72",
                        "run_id": "run-info",
                        "mape": 9.5,
                        "promoted_at": "2026-04-11",
                        "promoted_by": "auto",
                    }
                }
            },
        )
        monkeypatch.setattr(loader_module, "_PRODUCTION_FILE", tmp_path / "models_production.yaml")

        info = get_production_info("weather_forecasting_h72")
        assert info["run_id"] == "run-info"
        assert info["mape"] == pytest.approx(9.5)


# ---------------------------------------------------------------------------
# CLI run_promote
# ---------------------------------------------------------------------------

class TestRunPromoteCLI:
    def test_parser_requires_experiment_name(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_accepts_all_args(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "--experiment-name", "weather_forecasting_h72",
            "--run-id", "abc123",
            "--metric", "mape",
            "--model-name", "my_model",
            "--tracking-uri", "http://mlflow:5000",
            "--force",
        ])
        assert args.experiment_name == "weather_forecasting_h72"
        assert args.run_id == "abc123"
        assert args.force is True

    def test_main_promotes_best_when_no_run_id(
        self, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        import src.ml_workstation.promotion.run_promote as cli_module

        monkeypatch.setattr(
            cli_module,
            "_build_parser",
            lambda: _fixed_parser(run_id=None),
        )
        monkeypatch.setattr(cli_module, "promote_best", lambda **kw: "2")
        monkeypatch.setattr(cli_module, "promote_run", lambda **kw: (_ for _ in ()).throw(AssertionError("should not call promote_run")))

        main()
        out = capsys.readouterr().out
        assert "Melhor run promovido" in out

    def test_main_promotes_explicit_run_id(
        self, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        import src.ml_workstation.promotion.run_promote as cli_module

        monkeypatch.setattr(
            cli_module,
            "_build_parser",
            lambda: _fixed_parser(run_id="abc123"),
        )
        monkeypatch.setattr(cli_module, "promote_run", lambda **kw: "5")

        main()
        out = capsys.readouterr().out
        assert "abc123" in out

    def test_main_exits_1_on_rejection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.ml_workstation.promotion.run_promote as cli_module

        monkeypatch.setattr(
            cli_module,
            "_build_parser",
            lambda: _fixed_parser(run_id=None),
        )
        monkeypatch.setattr(
            cli_module,
            "promote_best",
            lambda **kw: (_ for _ in ()).throw(PromotionRejectedError("candidato pior")),
        )

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

class _FixedArgs:
    def __init__(self, run_id):
        self.experiment_name = "weather_forecasting_h72"
        self.run_id = run_id
        self.metric = "mape"
        self.model_name = None
        self.tracking_uri = None
        self.force = False


class _FixedParser:
    def __init__(self, run_id):
        self._run_id = run_id

    def parse_args(self):
        return _FixedArgs(self._run_id)


def _fixed_parser(run_id):
    return _FixedParser(run_id)
