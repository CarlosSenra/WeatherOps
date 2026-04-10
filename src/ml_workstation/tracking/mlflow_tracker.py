import logging
import os
import re
import subprocess
import uuid
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch.nn as nn

from src.ml_workstation.config.training_config import TrainingConfig

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Wrapper fino sobre a API MLflow para rastreamento de experimentos.

    Centraliza todas as chamadas MLflow neste módulo para facilitar
    substituição ou extensão futura.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._run = None
        mlflow.set_experiment(config.experiment_name)
        logger.info("Experimento MLflow: %s", config.experiment_name)

    def start_run(self) -> None:
        """
        Inicia um run MLflow e registra todos os parâmetros do TrainingConfig.

        O config completo é serializado via Pydantic e logado de uma vez.
        """
        run_name = self.config.run_name or f"run_{uuid.uuid4().hex[:8]}"
        self._run = mlflow.start_run(run_name=run_name)
        logger.info("MLflow run iniciado: %s (id=%s)", run_name, self._run.info.run_id)

        params = self.config.model_dump(mode="json", exclude_none=True)
        flat_params = {
            k: v
            for k, v in _flatten_dict(params).items()
            if not k.startswith("governance.")
        }
        governance = self._build_governance_schema()

        mlflow.log_params(flat_params)

        mlflow.set_tags(governance)
        mlflow.log_params({f"governance.{k}": str(v) for k, v in governance.items()})

    def log_governance_metrics(self, metrics: dict[str, float]) -> None:
        """Loga métricas de governança (snapshot final) no run atual."""
        if not metrics:
            return

        clean_metrics = {
            f"governance_metric.{k}": float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        if clean_metrics:
            mlflow.log_metrics(clean_metrics)

    def _build_governance_schema(self) -> dict[str, str]:
        """Monta o schema mínimo obrigatório para governança de modelo."""
        owner = self.config.governance.owner or _read_email_from_environment() or "unknown"
        git_sha = _resolve_git_sha(self.config.governance.git_sha)
        data_version = _resolve_training_data_version(self.config.governance.training_data_version)

        schema = {
            "model_name": self.config.governance.model_name,
            "model_version": self.config.governance.model_version,
            "model_type": self.config.governance.model_type,
            "training_data_version": data_version,
            "owner": owner,
            "risk_level": self.config.governance.risk_level,
            "fairness_checked": str(self.config.governance.fairness_checked).lower(),
            "git_sha": git_sha,
        }
        return schema

    def log_epoch_metrics(self, metrics: dict[str, float], epoch: int) -> None:
        """
        Loga métricas de uma época.

        Args:
            metrics: Dicionário com métricas (train_loss, val_loss, mae, rmse, mape).
            epoch: Número da época (usado como step no MLflow).
        """
        mlflow.log_metrics(metrics, step=epoch)

    def log_artifact(self, path: str) -> None:
        """Loga um arquivo como artefato do run."""
        mlflow.log_artifact(path)
        logger.info("Artefato logado: %s", path)

    def log_model(self, model: nn.Module, artifact_path: str = "model") -> None:
        """Registra o modelo PyTorch no MLflow."""
        mlflow.pytorch.log_model(model, artifact_path, serialization_format='pt2')
        logger.info("Modelo PyTorch registrado no MLflow em: %s", artifact_path)

    def end_run(self) -> None:
        """Finaliza o run MLflow."""
        mlflow.end_run()
        logger.info("MLflow run finalizado: %s", self.run_id)

    @property
    def run_id(self) -> str | None:
        if self._run is None:
            return None
        return self._run.info.run_id


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, str]:
    """Achata um dicionário aninhado para logar como parâmetros MLflow."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = str(v)
    return items


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_git_sha() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_workspace_root(),
            text=True,
        )
        return output.strip()
    except Exception:
        return "unknown"


def _resolve_git_sha(config_value: str | None) -> str:
    """Resolve git SHA com precedência: config -> env -> runtime -> unknown."""
    if config_value:
        return config_value.strip()

    env_value = os.getenv("GIT_SHA")
    if env_value:
        clean = env_value.strip("\"'").strip()
        if clean:
            return clean

    runtime_value = _read_git_sha()
    return runtime_value or "unknown"


def _read_data_dvc_hash() -> str | None:
    dvc_file = _workspace_root() / "data.dvc"
    if not dvc_file.exists():
        return None

    text = dvc_file.read_text(encoding="utf-8")
    patterns = [
        r"-\s*md5:\s*([^\s]+)",
        r"\bmd5:\s*([^\s]+)",
        r"\bhash:\s*([^\s]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def _resolve_training_data_version(config_value: str | None) -> str:
    """Resolve versão dos dados com precedência: config -> env -> data.dvc -> unknown."""
    if config_value:
        return config_value.strip()

    env_value = os.getenv("TRAINING_DATA_VERSION")
    if env_value:
        clean = env_value.strip("\"'").strip()
        if clean:
            return clean

    runtime_value = _read_data_dvc_hash()
    return runtime_value or "unknown"


def _read_email_from_environment() -> str | None:
    email = os.getenv("EMAIL")
    if email:
        return email.strip("\"'").strip() or None

    env_path = _workspace_root() / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "EMAIL":
            clean = value.strip().strip("\"'").strip()
            return clean or None
    return None
