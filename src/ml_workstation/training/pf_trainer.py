"""
Trainer para modelos pytorch-forecasting (TFT, NBEATS) baseado em PyTorch Lightning.

Usa `pl.Trainer` com callbacks `EarlyStopping` e `ModelCheckpoint`, e integra
com o `MLflowTracker` existente via um callback Lightning personalizado que
replica a interface de logging por época do `Trainer` original.

Retorna `TrainerOutput` — o mesmo Pydantic model do `Trainer` convencional —
garantindo compatibilidade com o restante do pipeline (avaliação, promoção).
"""
import logging
import math
from pathlib import Path
from typing import Mapping

import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks.callback import Callback
from torch.utils.data import DataLoader

from src.ml_workstation.config.training_config import TrainingConfig
from src.ml_workstation.training.trainer import TrainerOutput

logger = logging.getLogger(__name__)


def _canonicalize_tft_mape(metrics: Mapping[str, float]) -> dict[str, float]:
    """Normaliza a métrica MAPE do TFT para a chave canônica `mape`.

    O ecossistema Lightning/Pytorch Forecasting pode emitir MAPE com chaves
    diferentes (ex.: `val_MAPE`, `val_mape` ou `MAPE`). A promoção usa
    explicitamente `mape`, então consolidamos aqui sem remover métricas
    auxiliares do callback.
    """
    canonicalized = dict(metrics)

    # Preserva um `mape` já calculado explicitamente pelo pipeline.
    if "mape" in canonicalized:
        return canonicalized

    candidate_keys = ("val_MAPE", "val_mape", "MAPE", "mape_scaled")
    for key in candidate_keys:
        value = canonicalized.get(key)
        if value is not None and not math.isnan(value):
            canonicalized["mape"] = float(value)
            break

    return canonicalized


class _MLflowEpochCallback(Callback):
    """
    Callback Lightning que encaminha métricas por época ao `MLflowTracker`.

    Monitora `train_loss` e `val_loss` disponibilizadas pelo pytorch-forecasting
    e chama `tracker.log_epoch_metrics()` ao fim de cada época de validação.
    """

    def __init__(self, tracker) -> None:
        super().__init__()
        self.tracker = tracker
        self.best_epoch: int = 0
        self.best_val_loss: float = float("inf")
        self._last_metrics: dict[str, float] = {}

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        epoch = trainer.current_epoch
        metrics: dict[str, float] = {}

        for key, value in trainer.callback_metrics.items():
            try:
                scalar = float(value)
                if not math.isnan(scalar):
                    metrics[key] = scalar
            except (TypeError, ValueError):
                pass

        if not metrics:
            return

        metrics = _canonicalize_tft_mape(metrics)
        self._last_metrics = metrics
        self.tracker.log_epoch_metrics(metrics, epoch=epoch)

        val_loss = metrics.get("val_loss", metrics.get("val_MAE", float("inf")))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

        logger.info(
            "Época %d — %s",
            epoch,
            " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
        )


class PytorchForecastingTrainer:
    """
    Orquestra o treinamento de modelos pytorch-forecasting com PyTorch Lightning.

    Responsabilidades:
    - Configura `pl.Trainer` com `EarlyStopping`, `ModelCheckpoint` e o callback
      de logging MLflow.
    - Executa `trainer.fit()` e coleta as métricas finais.
    - Salva o checkpoint do melhor modelo e loga no `MLflowTracker`.
    - Retorna `TrainerOutput` compatível com o restante do pipeline.

    Não é responsável por construir o modelo ou os dataloaders.
    """

    def __init__(
        self,
        model: pl.LightningModule,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        tracker,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tracker = tracker

        run_name = config.run_name or "run"
        self.checkpoint_dir = Path(config.checkpoint_dir) / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def fit(self) -> TrainerOutput:
        """Executa o treinamento completo e retorna o resultado."""
        mlflow_cb = _MLflowEpochCallback(self.tracker)

        early_stop_cb = EarlyStopping(
            monitor="val_loss",
            patience=self.config.early_stopping_patience,
            mode="min",
            verbose=True,
        )

        checkpoint_cb = ModelCheckpoint(
            dirpath=str(self.checkpoint_dir),
            filename="best_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=False,
        )

        accelerator, devices = self._resolve_accelerator()

        pl_trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=[mlflow_cb, early_stop_cb, checkpoint_cb],
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=False,
        )

        logger.info(
            "Iniciando pl.Trainer — max_epochs=%d, accelerator=%s, devices=%s",
            self.config.epochs,
            accelerator,
            devices,
        )

        pl_trainer.fit(
            self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
        )

        best_checkpoint = checkpoint_cb.best_model_path or str(
            self.checkpoint_dir / "best_model.ckpt"
        )
        best_val_loss = float(checkpoint_cb.best_model_score or mlflow_cb.best_val_loss)

        total_epochs = pl_trainer.current_epoch
        best_epoch = mlflow_cb.best_epoch
        final_metrics = mlflow_cb._last_metrics

        logger.info(
            "Treinamento PF concluído — best_val_loss=%.4f na época %d/%d",
            best_val_loss,
            best_epoch,
            total_epochs,
        )

        return TrainerOutput(
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            checkpoint_path=best_checkpoint,
            total_epochs_run=total_epochs,
            final_metrics=final_metrics,
        )

    def _resolve_accelerator(self) -> tuple[str, int | str]:
        """Mapeia o campo `config.device` para os parâmetros do `pl.Trainer`."""
        device = self.config.device.lower()
        if device.startswith("cuda") and torch.cuda.is_available():
            return "gpu", 1
        if device == "mps" and torch.backends.mps.is_available():
            return "mps", 1
        return "cpu", "auto"
