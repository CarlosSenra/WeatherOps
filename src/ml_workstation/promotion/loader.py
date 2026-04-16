from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import mlflow
import mlflow.pytorch
import torch

if TYPE_CHECKING:
    from sklearn.preprocessing import StandardScaler

from src.ml_workstation.evaluation.mlflow_helpers import resolve_tracking_uri
from src.ml_workstation.promotion.export_local import (
    is_mlflow_pytorch_model_dir,
    read_export_manifest,
)

logger = logging.getLogger(__name__)


class ModelNotInProductionError(Exception):
    """Levantada quando nenhum modelo está promovido para o experimento solicitado."""


def _resolve_model_name(experiment_name: str, model_name: str | None) -> str:
    return model_name or experiment_name


def _get_production_version(
    client: mlflow.MlflowClient,
    model_name: str,
    experiment_name: str,
) -> mlflow.entities.model_registry.ModelVersion:
    """Retorna o ModelVersion com alias 'production' para o model_name.

    Raises:
        ModelNotInProductionError: se o alias 'production' não existir no Registry.
    """
    try:
        return client.get_model_version_by_alias(model_name, "production")
    except Exception as exc:
        raise ModelNotInProductionError(
            f"Nenhum modelo em produção para '{experiment_name}' "
            f"(Registry name: '{model_name}'). Erro: {exc}"
        ) from exc


def load_production_model(
    experiment_name: str,
    tracking_uri: str | None = None,
    device: str = "cpu",
    model_name: str | None = None,
    *,
    local_model_root: str | Path | None = None,
) -> torch.nn.Module:
    """Carrega o modelo em produção via MLflow Model Registry (alias 'production').

    Se ``local_model_root`` estiver definido e existir
    ``<local_model_root>/<model_name>/`` com um modelo PyTorch MLflow (pasta com
    ``MLmodel``), carrega a partir do disco; caso contrário usa o Registry.

    Args:
        experiment_name: Nome do experimento MLflow (ex.: weather_forecasting_h72).
        tracking_uri: URI do MLflow tracking. Auto-detectado se None.
        device: Dispositivo de inferência ('cpu' ou 'cuda').
        model_name: Nome no Registry (padrão: experiment_name).
        local_model_root: Diretório base com exportações (ex.: ``/app/ml_models``).

    Returns:
        Modelo PyTorch carregado e pronto para inferência.

    Raises:
        ModelNotInProductionError: se não houver alias 'production' ou o carregamento falhar.
    """
    effective_model_name = _resolve_model_name(experiment_name, model_name)
    map_device = torch.device(device)

    if local_model_root:
        local_dir = Path(local_model_root) / effective_model_name
        if is_mlflow_pytorch_model_dir(local_dir):
            try:
                model = mlflow.pytorch.load_model(
                    str(local_dir), map_location=map_device
                )
                logger.info("Modelo carregado de disco: %s", local_dir)
                return model
            except Exception as exc:
                logger.warning(
                    "Falha ao carregar de %s (%s); a tentar Registry.",
                    local_dir,
                    exc,
                )

    resolved = resolve_tracking_uri(tracking_uri)
    if resolved:
        mlflow.set_tracking_uri(resolved)

    client = mlflow.MlflowClient()
    _get_production_version(client, effective_model_name, experiment_name)

    registry_uri = f"models:/{effective_model_name}@production"

    try:
        model = mlflow.pytorch.load_model(registry_uri, map_location=map_device)
        logger.info("Modelo carregado do Registry: %s", registry_uri)
        return model
    except Exception as exc:
        raise ModelNotInProductionError(
            f"Falha ao carregar modelo em produção para '{experiment_name}'. "
            f"Registry URI: {registry_uri}. Erro: {exc}"
        ) from exc


def get_production_info(
    experiment_name: str,
    tracking_uri: str | None = None,
    model_name: str | None = None,
    *,
    local_model_root: str | Path | None = None,
) -> dict:
    """Retorna os metadados do modelo em produção lidos das tags da model version.

    Se existir ``manifest.json`` numa exportação local (``local_model_root``), usa-o;
    caso contrário consulta o MLflow Model Registry.

    Returns:
        Dicionário com model_name, version, run_id, mape, promoted_at, promoted_by,
        experiment_name.

    Raises:
        ModelNotInProductionError: se não houver modelo promovido.
    """
    effective_model_name = _resolve_model_name(experiment_name, model_name)

    if local_model_root:
        local_dir = Path(local_model_root) / effective_model_name
        manifest = read_export_manifest(local_dir) if local_dir.is_dir() else None
        if manifest and is_mlflow_pytorch_model_dir(local_dir):
            return {
                "model_name": manifest.get("model_name", effective_model_name),
                "version": manifest.get("registry_version", ""),
                "run_id": manifest.get("run_id", ""),
                "mape": manifest.get("mape"),
                "promoted_at": manifest.get("promoted_at"),
                "promoted_by": "export",
                "experiment_name": manifest.get("experiment_name", experiment_name),
            }

    resolved = resolve_tracking_uri(tracking_uri)
    if resolved:
        mlflow.set_tracking_uri(resolved)

    client = mlflow.MlflowClient()
    mv = _get_production_version(client, effective_model_name, experiment_name)

    mape_str = mv.tags.get("mape")
    return {
        "model_name": effective_model_name,
        "version": mv.version,
        "run_id": mv.run_id,
        "mape": float(mape_str) if mape_str else None,
        "promoted_at": mv.tags.get("promoted_at"),
        "promoted_by": mv.tags.get("promoted_by"),
        "experiment_name": mv.tags.get("experiment_name", experiment_name),
    }


def load_production_scaler(
    experiment_name: str,
    tracking_uri: str | None = None,
    model_name: str | None = None,
) -> StandardScaler:
    """Carrega o StandardScaler do run promovido em produção.

    O scaler é baixado do MLflow como artefato 'scaler.pkl' do run associado à
    model version em produção. Aplicável a modelos LSTM e Transformer.

    Modelos TFT e N-BEATS incorporam normalização internamente via GroupNormalizer
    do pytorch-forecasting e não requerem scaler externo.

    Args:
        experiment_name: Nome do experimento MLflow (ex.: weather_forecasting_h72).
        tracking_uri: URI do MLflow tracking. Auto-detectado se None.
        model_name: Nome no Registry (padrão: experiment_name).

    Returns:
        StandardScaler ajustado nos dados de treino do run promovido.

    Raises:
        ModelNotInProductionError: se não houver modelo promovido ou o artefato não existir.
    """
    resolved = resolve_tracking_uri(tracking_uri)
    if resolved:
        mlflow.set_tracking_uri(resolved)

    effective_model_name = _resolve_model_name(experiment_name, model_name)
    client = mlflow.MlflowClient()
    mv = _get_production_version(client, effective_model_name, experiment_name)

    run_id = mv.run_id
    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="scaler.pkl"
        )
        scaler: StandardScaler = joblib.load(local_path)
        logger.info("Scaler carregado do run %s para '%s'.", run_id, experiment_name)
        return scaler
    except Exception as exc:
        raise ModelNotInProductionError(
            f"Falha ao carregar scaler.pkl do run '{run_id}' "
            f"(experimento: '{experiment_name}'). "
            f"Nota: modelos TFT/N-BEATS não requerem scaler externo. Erro: {exc}"
        ) from exc
