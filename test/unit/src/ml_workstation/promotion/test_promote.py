from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

import src.ml_workstation.promotion.export_local as export_local_module
import src.ml_workstation.promotion.promote as promote_module
import src.ml_workstation.promotion.loader as loader_module
import src.ml_workstation.promotion.run_promote as run_promote_module
from src.ml_workstation.promotion.promote import (
    PromotionRejectedError,
    promote_run,
    select_best_run,
)
from src.ml_workstation.promotion.export_local import export_promoted_model_to_disk
from src.ml_workstation.promotion.loader import (
    ModelNotInProductionError,
    get_production_info,
    load_production_model,
    load_production_scaler,
)
from src.ml_workstation.promotion.run_promote import _build_parser, _resolve_export_dir, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run(run_id: str, mape: float | None, status: str = "FINISHED") -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.data.metrics = ({"mape": mape, "val_loss": 0.1} if mape is not None else {"val_loss": 0.1})
    run.data.params = {}
    run.info.status = status
    return run


def _make_model_version(
    version: str,
    run_id: str,
    mape: float | None = None,
    promoted_at: str = "2026-04-14",
    promoted_by: str = "auto",
    experiment_name: str = "weather_forecasting_h72",
) -> SimpleNamespace:
    """Cria um objeto simples que replica os atributos de mlflow ModelVersion."""
    tags: dict[str, str] = {
        "promoted_at": promoted_at,
        "promoted_by": promoted_by,
        "experiment_name": experiment_name,
        "run_id": run_id,
    }
    if mape is not None:
        tags["mape"] = str(float(mape))
    return SimpleNamespace(version=version, run_id=run_id, tags=tags)


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
    def _make_client(
        self,
        run: MagicMock,
        current_mv: SimpleNamespace | None = None,
    ) -> MagicMock:
        """Cria um MlflowClient mockado com estado de produção opcional."""
        client = MagicMock()
        client.get_run.return_value = run
        if current_mv is None:
            client.get_model_version_by_alias.side_effect = Exception("alias not found")
        else:
            client.get_model_version_by_alias.return_value = current_mv
        return client

    def test_successful_promotion_sets_alias_and_tags(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _make_run("run-abc", mape=8.0)
        client = self._make_client(run, current_mv=None)

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
        # Verifica que as tags foram gravadas
        tag_calls = {
            c.args[2]: c.args[3]
            for c in client.set_model_version_tag.call_args_list
        }
        assert tag_calls["mape"] == str(8.0)
        assert tag_calls["promoted_by"] == "auto"
        assert tag_calls["run_id"] == "run-abc"
        assert tag_calls["experiment_name"] == "weather_forecasting_h72"
        assert "promoted_at" in tag_calls

    def test_rejects_promotion_when_mape_is_worse(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        current_mv = _make_model_version("2", "run-old", mape=5.0)
        run = _make_run("run-bad", mape=10.0)
        client = self._make_client(run, current_mv=current_mv)

        monkeypatch.setattr(promote_module, "resolve_tracking_uri", lambda uri: None)

        with patch("src.ml_workstation.promotion.promote.mlflow.MlflowClient", return_value=client):
            with pytest.raises(PromotionRejectedError, match="candidato MAPE=10"):
                promote_run(
                    run_id="run-bad",
                    experiment_name="weather_forecasting_h72",
                )

        client.set_registered_model_alias.assert_not_called()
        client.set_model_version_tag.assert_not_called()
        client.get_model_version_by_alias.assert_called_once_with(
            "weather_forecasting_h72", "production"
        )

    def test_rejection_keeps_existing_production_version_id_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ao rejeitar candidato pior, o alias production não deve ser reatribuído."""
        current_mv = _make_model_version("42", "run-prod", mape=4.0)
        run = _make_run("run-bad", mape=9.0)
        client = self._make_client(run, current_mv=current_mv)

        monkeypatch.setattr(promote_module, "resolve_tracking_uri", lambda uri: None)

        with patch("src.ml_workstation.promotion.promote.mlflow.MlflowClient", return_value=client):
            with pytest.raises(PromotionRejectedError):
                promote_run(
                    run_id="run-bad",
                    experiment_name="weather_forecasting_h72",
                )

        client.set_registered_model_alias.assert_not_called()
        client.set_model_version_tag.assert_not_called()
        client.get_model_version_by_alias.assert_called_once_with(
            "weather_forecasting_h72", "production"
        )

    def test_force_bypasses_regression_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        current_mv = _make_model_version("2", "run-old", mape=5.0)
        run = _make_run("run-worse", mape=15.0)
        client = self._make_client(run, current_mv=current_mv)

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
        client.set_registered_model_alias.assert_called_once()

    def test_uses_custom_model_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _make_run("run-xyz", mape=3.0)
        client = self._make_client(run, current_mv=None)

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
        tag_calls = {
            c.args[2]: c.args[3]
            for c in client.set_model_version_tag.call_args_list
        }
        assert tag_calls["promoted_by"] == "manual"

    def test_first_promotion_no_current_alias_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sem alias 'production' existente, a promoção deve prosseguir livremente."""
        run = _make_run("run-first", mape=12.0)
        client = self._make_client(run, current_mv=None)

        monkeypatch.setattr(promote_module, "resolve_tracking_uri", lambda uri: None)

        with (
            patch("src.ml_workstation.promotion.promote.mlflow.MlflowClient", return_value=client),
            patch(
                "src.ml_workstation.promotion.promote.mlflow.register_model",
                return_value=SimpleNamespace(version="1"),
            ),
        ):
            version = promote_run(
                run_id="run-first",
                experiment_name="weather_forecasting_h72",
            )

        assert version == "1"
        client.set_registered_model_alias.assert_called_once()

    def test_promotes_even_without_mape_metric(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run sem mape não deve bloquear a promoção (guarda incompleta é inerte)."""
        run = _make_run("run-no-mape", mape=None)
        current_mv = _make_model_version("2", "run-old", mape=5.0)
        client = self._make_client(run, current_mv=current_mv)

        monkeypatch.setattr(promote_module, "resolve_tracking_uri", lambda uri: None)

        with (
            patch("src.ml_workstation.promotion.promote.mlflow.MlflowClient", return_value=client),
            patch(
                "src.ml_workstation.promotion.promote.mlflow.register_model",
                return_value=SimpleNamespace(version="5"),
            ),
        ):
            version = promote_run(
                run_id="run-no-mape",
                experiment_name="weather_forecasting_h72",
            )

        assert version == "5"

    def test_uses_val_mape_alias_when_canonical_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _make_run("run-val-mape", mape=None)
        run.data.metrics["val_MAPE"] = 7.25
        client = self._make_client(run, current_mv=None)

        monkeypatch.setattr(promote_module, "resolve_tracking_uri", lambda uri: None)

        with (
            patch("src.ml_workstation.promotion.promote.mlflow.MlflowClient", return_value=client),
            patch(
                "src.ml_workstation.promotion.promote.mlflow.register_model",
                return_value=SimpleNamespace(version="3"),
            ),
        ):
            version = promote_run(
                run_id="run-val-mape",
                experiment_name="weather_forecasting_h72",
            )

        assert version == "3"
        mape_tag_calls = [
            c for c in client.set_model_version_tag.call_args_list if c.args[2] == "mape"
        ]
        assert mape_tag_calls
        assert mape_tag_calls[-1].args[3] == "7.25"

    def test_horizons_are_independent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Promoção para h72 não deve consultar ou alterar o Registry de h168."""
        run = _make_run("run-h72", mape=6.0)
        client = self._make_client(run, current_mv=None)

        monkeypatch.setattr(promote_module, "resolve_tracking_uri", lambda uri: None)

        with (
            patch("src.ml_workstation.promotion.promote.mlflow.MlflowClient", return_value=client),
            patch(
                "src.ml_workstation.promotion.promote.mlflow.register_model",
                return_value=SimpleNamespace(version="1"),
            ),
        ):
            promote_run(run_id="run-h72", experiment_name="weather_forecasting_h72")

        alias_calls = client.set_registered_model_alias.call_args_list
        assert all(c.kwargs["name"] == "weather_forecasting_h72" for c in alias_calls)
        tag_calls = client.set_model_version_tag.call_args_list
        assert all(c.args[0] == "weather_forecasting_h72" for c in tag_calls)

    def test_raises_when_strict_export_enabled_and_export_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _make_run("run-export-fail", mape=4.0)
        client = self._make_client(run, current_mv=None)
        monkeypatch.setattr(promote_module, "resolve_tracking_uri", lambda uri: None)
        monkeypatch.setattr(
            promote_module,
            "export_promoted_model_to_disk",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("broken export")),
        )

        with (
            patch("src.ml_workstation.promotion.promote.mlflow.MlflowClient", return_value=client),
            patch(
                "src.ml_workstation.promotion.promote.mlflow.register_model",
                return_value=SimpleNamespace(version="9"),
            ),
            pytest.raises(RuntimeError, match="Exportação local falhou"),
        ):
            promote_run(
                run_id="run-export-fail",
                experiment_name="weather_forecasting_h72",
                export_dir="src/api/ml_models",
                strict_export=True,
            )

    def test_keeps_previous_behavior_when_strict_export_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _make_run("run-export-warn", mape=4.0)
        client = self._make_client(run, current_mv=None)
        monkeypatch.setattr(promote_module, "resolve_tracking_uri", lambda uri: None)
        monkeypatch.setattr(
            promote_module,
            "export_promoted_model_to_disk",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("broken export")),
        )

        with (
            patch("src.ml_workstation.promotion.promote.mlflow.MlflowClient", return_value=client),
            patch(
                "src.ml_workstation.promotion.promote.mlflow.register_model",
                return_value=SimpleNamespace(version="10"),
            ),
        ):
            version = promote_run(
                run_id="run-export-warn",
                experiment_name="weather_forecasting_h72",
                export_dir="src/api/ml_models",
                strict_export=False,
            )

        assert version == "10"


# ---------------------------------------------------------------------------
# load_production_model
# ---------------------------------------------------------------------------

class TestLoadProductionModel:
    def test_loads_via_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dummy_model = torch.nn.Linear(4, 2)
        mv = _make_model_version("3", "run-ok", mape=7.0)

        client = MagicMock()
        client.get_model_version_by_alias.return_value = mv

        monkeypatch.setattr(loader_module, "resolve_tracking_uri", lambda uri: None)
        monkeypatch.setattr(
            loader_module.mlflow.pytorch,
            "load_model",
            lambda uri, map_location=None: dummy_model,
        )

        with patch("src.ml_workstation.promotion.loader.mlflow.MlflowClient", return_value=client):
            model = load_production_model("weather_forecasting_h72")

        assert model is dummy_model
        client.get_model_version_by_alias.assert_called_once_with(
            "weather_forecasting_h72", "production"
        )

    def test_raises_when_no_production_alias(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = MagicMock()
        client.get_model_version_by_alias.side_effect = Exception("alias not found")

        monkeypatch.setattr(loader_module, "resolve_tracking_uri", lambda uri: None)

        with patch("src.ml_workstation.promotion.loader.mlflow.MlflowClient", return_value=client):
            with pytest.raises(ModelNotInProductionError, match="Nenhum modelo em produção"):
                load_production_model("weather_forecasting_h72")

    def test_raises_when_load_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mv = _make_model_version("3", "run-broken", mape=7.0)
        client = MagicMock()
        client.get_model_version_by_alias.return_value = mv

        monkeypatch.setattr(loader_module, "resolve_tracking_uri", lambda uri: None)
        monkeypatch.setattr(
            loader_module.mlflow.pytorch,
            "load_model",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("load failed")),
        )

        with patch("src.ml_workstation.promotion.loader.mlflow.MlflowClient", return_value=client):
            with pytest.raises(ModelNotInProductionError, match="Falha ao carregar"):
                load_production_model("weather_forecasting_h72")

    def test_uses_custom_model_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dummy_model = torch.nn.Linear(4, 2)
        mv = _make_model_version("1", "run-custom", mape=3.0)
        client = MagicMock()
        client.get_model_version_by_alias.return_value = mv

        monkeypatch.setattr(loader_module, "resolve_tracking_uri", lambda uri: None)
        monkeypatch.setattr(
            loader_module.mlflow.pytorch,
            "load_model",
            lambda uri, map_location=None: dummy_model,
        )

        with patch("src.ml_workstation.promotion.loader.mlflow.MlflowClient", return_value=client):
            model = load_production_model(
                "weather_forecasting_h72", model_name="my_custom_model"
            )

        assert model is dummy_model
        client.get_model_version_by_alias.assert_called_once_with("my_custom_model", "production")


# ---------------------------------------------------------------------------
# get_production_info
# ---------------------------------------------------------------------------

class TestGetProductionInfo:
    def test_returns_info_from_registry_tags(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mv = _make_model_version(
            "5",
            "run-info",
            mape=9.5,
            promoted_at="2026-04-14",
            promoted_by="auto",
            experiment_name="weather_forecasting_h72",
        )
        client = MagicMock()
        client.get_model_version_by_alias.return_value = mv

        monkeypatch.setattr(loader_module, "resolve_tracking_uri", lambda uri: None)

        with patch("src.ml_workstation.promotion.loader.mlflow.MlflowClient", return_value=client):
            info = get_production_info("weather_forecasting_h72")

        assert info["run_id"] == "run-info"
        assert info["mape"] == pytest.approx(9.5)
        assert info["version"] == "5"
        assert info["promoted_by"] == "auto"
        assert info["experiment_name"] == "weather_forecasting_h72"

    def test_raises_when_no_production_alias(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = MagicMock()
        client.get_model_version_by_alias.side_effect = Exception("alias not found")

        monkeypatch.setattr(loader_module, "resolve_tracking_uri", lambda uri: None)

        with patch("src.ml_workstation.promotion.loader.mlflow.MlflowClient", return_value=client):
            with pytest.raises(ModelNotInProductionError):
                get_production_info("weather_forecasting_h72")

    def test_handles_missing_mape_tag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Versão promovida sem tag 'mape' deve retornar None no campo mape."""
        mv = _make_model_version("1", "run-no-mape", mape=None)
        client = MagicMock()
        client.get_model_version_by_alias.return_value = mv

        monkeypatch.setattr(loader_module, "resolve_tracking_uri", lambda uri: None)

        with patch("src.ml_workstation.promotion.loader.mlflow.MlflowClient", return_value=client):
            info = get_production_info("weather_forecasting_h72")

        assert info["mape"] is None


# ---------------------------------------------------------------------------
# load_production_scaler
# ---------------------------------------------------------------------------

class TestLoadProductionScaler:
    def test_loads_scaler_from_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sklearn_pre = pytest.importorskip("sklearn.preprocessing", exc_type=ImportError)
        dummy_scaler = sklearn_pre.StandardScaler()

        mv = _make_model_version("3", "run-with-scaler", mape=5.0)
        client = MagicMock()
        client.get_model_version_by_alias.return_value = mv

        monkeypatch.setattr(loader_module, "resolve_tracking_uri", lambda uri: None)
        monkeypatch.setattr(
            loader_module.mlflow.artifacts,
            "download_artifacts",
            lambda run_id, artifact_path: "/tmp/scaler.pkl",
        )
        monkeypatch.setattr(loader_module.joblib, "load", lambda path: dummy_scaler)

        with patch("src.ml_workstation.promotion.loader.mlflow.MlflowClient", return_value=client):
            scaler = load_production_scaler("weather_forecasting_h72")

        assert scaler is dummy_scaler

    def test_raises_when_no_production_alias(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = MagicMock()
        client.get_model_version_by_alias.side_effect = Exception("alias not found")

        monkeypatch.setattr(loader_module, "resolve_tracking_uri", lambda uri: None)

        with patch("src.ml_workstation.promotion.loader.mlflow.MlflowClient", return_value=client):
            with pytest.raises(ModelNotInProductionError):
                load_production_scaler("weather_forecasting_h72")

    def test_raises_when_scaler_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mv = _make_model_version("3", "run-no-scaler", mape=5.0)
        client = MagicMock()
        client.get_model_version_by_alias.return_value = mv

        monkeypatch.setattr(loader_module, "resolve_tracking_uri", lambda uri: None)
        monkeypatch.setattr(
            loader_module.mlflow.artifacts,
            "download_artifacts",
            lambda **kw: (_ for _ in ()).throw(Exception("artifact not found")),
        )

        with patch("src.ml_workstation.promotion.loader.mlflow.MlflowClient", return_value=client):
            with pytest.raises(ModelNotInProductionError, match="scaler.pkl"):
                load_production_scaler("weather_forecasting_h72")


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
            "--strict-export",
        ])
        assert args.experiment_name == "weather_forecasting_h72"
        assert args.run_id == "abc123"
        assert args.force is True
        assert args.strict_export is True

    def test_main_promotes_best_when_no_run_id(
        self, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        import src.ml_workstation.promotion.run_promote as cli_module

        monkeypatch.setattr(
            cli_module,
            "_build_parser",
            lambda: _fixed_parser(run_id=None),
        )
        monkeypatch.setattr(cli_module, "_resolve_export_dir", lambda path: path)
        monkeypatch.setattr(cli_module, "promote_best", lambda **kw: "2")
        monkeypatch.setattr(
            cli_module,
            "promote_run",
            lambda **kw: (_ for _ in ()).throw(AssertionError("should not call promote_run")),
        )

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
        monkeypatch.setattr(cli_module, "_resolve_export_dir", lambda path: path)
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
        monkeypatch.setattr(cli_module, "_resolve_export_dir", lambda path: path)
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
        self.export_dir = None
        self.strict_export = True
        self.update_knowledge_base = False
        self.knowledge_base_path = None


class _FixedParser:
    def __init__(self, run_id):
        self._run_id = run_id

    def parse_args(self):
        return _FixedArgs(self._run_id)


def _fixed_parser(run_id):
    return _FixedParser(run_id)


class TestResolveExportDir:
    def test_returns_none_for_empty(self) -> None:
        assert _resolve_export_dir(None) is None

    def test_resolves_relative_path_from_workspace_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            run_promote_module,
            "workspace_root",
            lambda: tmp_path,
        )
        resolved = _resolve_export_dir("src/api/ml_models")
        assert resolved == str((tmp_path / "src/api/ml_models").resolve())

    def test_keeps_windows_absolute_path_without_prefixing_workspace_root(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            run_promote_module,
            "workspace_root",
            lambda: Path("/home/runner/work/WeatherOps"),
        )
        resolved = _resolve_export_dir("C:/repo/src/api/ml_models")
        assert resolved == os.path.normpath("C:/repo/src/api/ml_models")


class TestExportPromotedModelToDisk:
    def test_fails_when_required_artifacts_are_missing(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        downloaded = tmp_path / "downloaded_model"
        downloaded.mkdir(parents=True)
        (downloaded / "MLmodel").write_text("flavors: {}", encoding="utf-8")
        monkeypatch.setattr(export_local_module, "resolve_tracking_uri", lambda uri: None)
        monkeypatch.setattr(export_local_module, "workspace_root", lambda: tmp_path)
        monkeypatch.setattr(
            export_local_module.mlflow.artifacts,
            "download_artifacts",
            lambda **kwargs: str(downloaded),
        )

        with pytest.raises(RuntimeError, match="faltando data/"):
            export_promoted_model_to_disk(
                run_id="run-1",
                effective_model_name="weather_forecasting_h72",
                registry_version="1",
                experiment_name="weather_forecasting_h72",
                tracking_uri=None,
                export_dir=tmp_path / "out",
                candidate_mape=1.0,
            )

    def test_exports_complete_layout_and_writes_manifest(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        downloaded = tmp_path / "downloaded_complete"
        (downloaded / "data").mkdir(parents=True)
        (downloaded / "MLmodel").write_text("flavors: {}", encoding="utf-8")
        (downloaded / "data" / "weights.bin").write_text("ok", encoding="utf-8")
        serving_spec = tmp_path / "data" / "spec" / "salvador"
        serving_spec.mkdir(parents=True)
        (serving_spec / "dados.parquet").write_text("fake parquet", encoding="utf-8")
        monkeypatch.setattr(export_local_module, "resolve_tracking_uri", lambda uri: None)
        monkeypatch.setattr(export_local_module, "workspace_root", lambda: tmp_path)
        monkeypatch.setattr(
            export_local_module.mlflow.artifacts,
            "download_artifacts",
            lambda **kwargs: str(downloaded),
        )

        result = export_promoted_model_to_disk(
            run_id="run-2",
            effective_model_name="weather_forecasting_h72",
            registry_version="2",
            experiment_name="weather_forecasting_h72",
            tracking_uri=None,
            export_dir=tmp_path / "out",
            candidate_mape=2.0,
        )

        assert result == (tmp_path / "out" / "weather_forecasting_h72")
        assert (result / "MLmodel").is_file()
        assert (result / "data").is_dir()
        assert (result / "manifest.json").is_file()
        assert (result / "serving_data").is_dir()
        assert (result / "serving_data" / "dados.parquet").is_file()
        assert (result / "serving_data" / "serving_data_metadata.json").is_file()
