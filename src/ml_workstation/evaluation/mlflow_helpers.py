from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import torch

from src.ml_workstation.config.training_config import DataConfig, ModelConfig
from src.ml_workstation.models import build_model

logger = logging.getLogger(__name__)


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_tracking_uri(tracking_uri: str | None) -> str | None:
    if tracking_uri:
        return tracking_uri

    local_mlruns = workspace_root() / "src" / "ml_workstation" / "mlruns"
    if local_mlruns.exists():
        return local_mlruns.resolve().as_uri()
    return None


def resolve_parquet_path(raw_path: str) -> str:
    """Resolve path from MLflow params to a valid local path when possible."""
    candidate = Path(raw_path)
    if candidate.exists():
        return str(candidate)

    normalized = raw_path.replace("\\", "/")
    if normalized.startswith("/app/"):
        relative = normalized[len("/app/"):]
        local_candidate = workspace_root() / relative
        if local_candidate.exists():
            return str(local_candidate)

    local_candidate = workspace_root() / raw_path
    if local_candidate.exists():
        return str(local_candidate)

    return raw_path


def parse_mlflow_param(raw: str) -> Any:
    """Converte strings de params do MLflow para tipos Python quando possivel."""
    text = raw.strip()
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass

    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False

    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def require_param(params: dict[str, str], key: str) -> Any:
    if key not in params:
        raise KeyError(f"Parametro ausente no run MLflow: {key}")
    return parse_mlflow_param(params[key])


def build_data_config(params: dict[str, str]) -> DataConfig:
    feature_columns = require_param(params, "data.feature_columns")
    target_columns = require_param(params, "data.target_columns")

    if not isinstance(feature_columns, list) or not all(isinstance(x, str) for x in feature_columns):
        raise ValueError("data.feature_columns invalido no run MLflow")

    if not isinstance(target_columns, list) or not all(isinstance(x, str) for x in target_columns):
        raise ValueError("data.target_columns invalido no run MLflow")

    return DataConfig(
        parquet_path=resolve_parquet_path(str(require_param(params, "data.parquet_path"))),
        feature_columns=feature_columns,
        target_columns=target_columns,
        sequence_length=int(require_param(params, "data.sequence_length")),
        horizon=int(require_param(params, "data.horizon")),
        train_ratio=float(require_param(params, "data.train_ratio")),
        val_ratio=float(require_param(params, "data.val_ratio")),
    )


def build_model_config(params: dict[str, str]) -> ModelConfig:
    return ModelConfig(
        model_type=str(require_param(params, "model.model_type")),
        hidden_size=int(require_param(params, "model.hidden_size")),
        num_layers=int(require_param(params, "model.num_layers")),
        dropout=float(require_param(params, "model.dropout")),
        num_heads=int(parse_mlflow_param(params.get("model.num_heads", "4"))),
        ffn_dim=int(parse_mlflow_param(params.get("model.ffn_dim", "256"))),
    )


def load_model_with_fallback(
    *,
    run,
    params: dict[str, str],
    data_config: DataConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    model_uri = f"runs:/{run.info.run_id}/model"
    try:
        logger.info("Carregando modelo de %s", model_uri)
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
        model.to(device)
        return model, model_uri
    except Exception as exc:
        logger.warning(
            "Falha ao carregar modelo via runs URI (%s). Tentando fallback best_model.pt. Erro: %s",
            model_uri,
            exc,
        )

    model_config = build_model_config(params)
    model = build_model(
        n_features=len(data_config.feature_columns),
        n_targets=len(data_config.target_columns),
        horizon=data_config.horizon,
        config=model_config,
    )

    checkpoint_path = (
        workspace_root()
        / "src"
        / "ml_workstation"
        / "mlruns"
        / run.info.experiment_id
        / run.info.run_id
        / "artifacts"
        / "best_model.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Nao foi possivel carregar o modelo. O URI runs:/ falhou e o checkpoint local nao existe em "
            f"{checkpoint_path}"
        )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, str(checkpoint_path)
