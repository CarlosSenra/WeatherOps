import logging
import uuid

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

        params = self.config.model_dump(mode="json")
        flat_params = _flatten_dict(params)
        mlflow.log_params(flat_params)

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
        mlflow.pytorch.log_model(model, artifact_path)
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
