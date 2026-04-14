from __future__ import annotations

import logging
from datetime import date

import mlflow
import mlflow.pytorch

from src.ml_workstation.evaluation.mlflow_helpers import resolve_tracking_uri

logger = logging.getLogger(__name__)


class PromotionRejectedError(Exception):
    """Levantada quando o candidato tem métrica pior que o modelo atual em produção."""


def _get_experiment_id(client: mlflow.MlflowClient, experiment_name: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento MLflow não encontrado: '{experiment_name}'")
    return experiment.experiment_id


def _get_current_production_mape(
    client: mlflow.MlflowClient, model_name: str
) -> float | None:
    """Retorna o MAPE da model version atualmente em produção via Registry tags.

    Retorna None se não houver alias 'production' ou se a tag 'mape' estiver ausente.
    """
    try:
        mv = client.get_model_version_by_alias(model_name, "production")
        mape_str = mv.tags.get("mape")
        return float(mape_str) if mape_str else None
    except Exception:
        return None


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
    """Registra o run no MLflow Model Registry com alias 'production'.

    O MLflow Model Registry é a única fonte de verdade sobre o estado de produção.
    Metadados de promoção (mape, promoted_at, promoted_by, experiment_name, run_id)
    são armazenados como tags na model version registrada.

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

    # Verifica regressão de métricas consultando o Registry
    current_mape = _get_current_production_mape(client, effective_model_name)

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

    # Grava metadados como tags na model version — fonte de verdade sobre produção
    tags = {
        "mape": str(float(candidate_mape)) if candidate_mape is not None else "",
        "promoted_at": date.today().isoformat(),
        "promoted_by": "manual" if model_name else "auto",
        "experiment_name": experiment_name,
        "run_id": run_id,
    }
    for key, value in tags.items():
        client.set_model_version_tag(effective_model_name, mv.version, key, value)
    logger.info("Tags de promoção gravadas na versão %s de '%s'.", mv.version, effective_model_name)

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
