from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import mlflow
import mlflow.pytorch
import yaml

from src.ml_workstation.evaluation.mlflow_helpers import resolve_tracking_uri

logger = logging.getLogger(__name__)

_PRODUCTION_FILE = Path(__file__).resolve().parents[1] / "models_production.yaml"


class PromotionRejectedError(Exception):
    """Levantada quando o candidato tem métrica pior que o modelo atual em produção."""


def _load_production_registry() -> dict:
    if not _PRODUCTION_FILE.exists():
        return {"experiments": {}}
    with _PRODUCTION_FILE.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {"experiments": {}}


def _save_production_registry(data: dict) -> None:
    with _PRODUCTION_FILE.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _get_experiment_id(client: mlflow.MlflowClient, experiment_name: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento MLflow não encontrado: '{experiment_name}'")
    return experiment.experiment_id


def select_best_run(
    experiment_name: str,
    metric: str = "mape",
    tracking_uri: str | None = None,
) -> mlflow.entities.Run:
    """Retorna o run com o menor valor de `metric` no experimento informado."""
    resolved = resolve_tracking_uri(tracking_uri)
    if resolved:
        mlflow.set_tracking_uri(resolved)

    client = mlflow.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[_get_experiment_id(client, experiment_name)],
        filter_string="status = 'FINISHED'",
        order_by=[f"metrics.{metric} ASC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(
            f"Nenhum run finalizado encontrado no experimento '{experiment_name}'."
        )

    best = runs[0]
    logger.info(
        "Melhor run selecionado: %s | %s=%.4f",
        best.info.run_id,
        metric,
        best.data.metrics.get(metric, float("nan")),
    )
    return best


def promote_run(
    run_id: str,
    experiment_name: str,
    model_name: str | None = None,
    tracking_uri: str | None = None,
    force: bool = False,
) -> str:
    """Registra o run no MLflow Model Registry com alias 'production' e atualiza
    models_production.yaml.

    Rejeita a promoção se o MAPE do candidato for pior que o atual em produção,
    a menos que `force=True`.

    Returns:
        Número da versão registrada no Model Registry.

    Raises:
        PromotionRejectedError: quando candidato tem MAPE pior e force=False.
    """
    resolved = resolve_tracking_uri(tracking_uri)
    if resolved:
        mlflow.set_tracking_uri(resolved)

    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    candidate_mape = run.data.metrics.get("mape")

    effective_model_name = model_name or experiment_name

    # Verifica regressão de métricas
    registry = _load_production_registry()
    current = registry.get("experiments", {}).get(experiment_name, {})
    current_mape = current.get("mape")

    if (
        not force
        and candidate_mape is not None
        and current_mape is not None
        and candidate_mape > current_mape
    ):
        delta = candidate_mape - current_mape
        raise PromotionRejectedError(
            f"Promoção rejeitada: candidato MAPE={candidate_mape:.4f} é pior que o atual "
            f"em produção MAPE={current_mape:.4f} (delta=+{delta:.4f}). "
            f"Use --force para sobrescrever."
        )

    # Registra no Model Registry
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=effective_model_name)
    logger.info("Modelo registrado: %s versão %s", effective_model_name, mv.version)

    # Seta alias 'production'
    client.set_registered_model_alias(
        name=effective_model_name,
        alias="production",
        version=mv.version,
    )
    logger.info("Alias 'production' atribuído a %s v%s", effective_model_name, mv.version)

    # Atualiza models_production.yaml
    experiments = registry.setdefault("experiments", {})
    experiments.setdefault(experiment_name, {}).update(
        {
            "model_name": effective_model_name,
            "run_id": run_id,
            "mape": float(candidate_mape) if candidate_mape is not None else None,
            "promoted_at": date.today().isoformat(),
            "promoted_by": "manual" if model_name else "auto",
        }
    )
    _save_production_registry(registry)
    logger.info("models_production.yaml atualizado para experimento '%s'.", experiment_name)

    return mv.version


def promote_best(
    experiment_name: str,
    metric: str = "mape",
    model_name: str | None = None,
    tracking_uri: str | None = None,
    force: bool = False,
) -> str:
    """Seleciona o melhor run e o promove para produção.

    Returns:
        Número da versão registrada no Model Registry.
    """
    best = select_best_run(
        experiment_name=experiment_name,
        metric=metric,
        tracking_uri=tracking_uri,
    )
    return promote_run(
        run_id=best.info.run_id,
        experiment_name=experiment_name,
        model_name=model_name,
        tracking_uri=tracking_uri,
        force=force,
    )
