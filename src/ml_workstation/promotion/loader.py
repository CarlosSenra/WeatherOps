from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import yaml

from src.ml_workstation.evaluation.mlflow_helpers import resolve_tracking_uri

logger = logging.getLogger(__name__)

_PRODUCTION_FILE = Path(__file__).resolve().parents[1] / "models_production.yaml"


class ModelNotInProductionError(Exception):
    """Levantada quando nenhum modelo está promovido para o experimento solicitado."""


def load_production_model(
    experiment_name: str,
    tracking_uri: str | None = None,
    device: str = "cpu",
) -> torch.nn.Module:
    """Carrega o modelo em produção para o experimento informado.

    Tenta o MLflow Model Registry primeiro (via alias 'production').
    Em caso de falha (servidor indisponível ou artefatos antigos), cai
    no URI runs:/ registrado no models_production.yaml.

    Args:
        experiment_name: Nome do experimento MLflow (ex.: weather_forecasting_h72).
        tracking_uri: URI do MLflow tracking. Auto-detectado se None.
        device: Dispositivo de inferência ('cpu' ou 'cuda').

    Returns:
        Modelo PyTorch carregado e pronto para inferência.

    Raises:
        ModelNotInProductionError: se não houver modelo promovido ou o carregamento falhar.
    """
    resolved = resolve_tracking_uri(tracking_uri)
    if resolved:
        mlflow.set_tracking_uri(resolved)

    entry = _get_production_entry(experiment_name)
    model_name = entry["model_name"]
    run_id = entry.get("run_id")
    map_device = torch.device(device)

    # Tenta Model Registry com alias 'production'
    registry_uri = f"models:/{model_name}@production"
    try:
        model = mlflow.pytorch.load_model(registry_uri, map_location=map_device)
        logger.info("Modelo carregado do Registry: %s", registry_uri)
        return model
    except Exception as exc:
        logger.warning(
            "Falha ao carregar via Registry (%s): %s. Tentando runs:/ URI.",
            registry_uri,
            exc,
        )

    # Fallback via runs:/ URI
    if not run_id:
        raise ModelNotInProductionError(
            f"Nenhum run_id em produção para '{experiment_name}' e Registry indisponível."
        )

    runs_uri = f"runs:/{run_id}/model"
    try:
        model = mlflow.pytorch.load_model(runs_uri, map_location=map_device)
        logger.info("Modelo carregado via runs URI: %s", runs_uri)
        return model
    except Exception as exc:
        raise ModelNotInProductionError(
            f"Falha ao carregar modelo em produção para '{experiment_name}'. "
            f"Registry URI: {registry_uri}. Runs URI: {runs_uri}. Erro: {exc}"
        ) from exc


def get_production_info(experiment_name: str) -> dict:
    """Retorna os metadados do modelo em produção para o experimento informado.

    Returns:
        Dicionário com model_name, run_id, mape, promoted_at, promoted_by.

    Raises:
        ModelNotInProductionError: se não houver modelo promovido.
    """
    return _get_production_entry(experiment_name)


def _get_production_entry(experiment_name: str) -> dict:
    if not _PRODUCTION_FILE.exists():
        raise ModelNotInProductionError(
            f"models_production.yaml não encontrado. "
            f"Nenhum modelo promovido para '{experiment_name}'."
        )

    with _PRODUCTION_FILE.open(encoding="utf-8") as f:
        registry = yaml.safe_load(f) or {}

    entry = registry.get("experiments", {}).get(experiment_name)
    if not entry or not entry.get("run_id"):
        raise ModelNotInProductionError(
            f"Nenhum modelo em produção para o experimento '{experiment_name}'."
        )

    return entry
