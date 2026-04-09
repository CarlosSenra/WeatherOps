import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.utils.data import DataLoader

from src.ml_workstation.config.training_config import TrainingConfig
from src.ml_workstation.models.interface import ITimeSeriesModel
from src.ml_workstation.training.metrics import compute_mape, compute_metrics

logger = logging.getLogger(__name__)


class TrainerOutput(BaseModel):
    """Resultado do treinamento."""

    best_epoch: int
    best_val_loss: float
    checkpoint_path: str
    total_epochs_run: int
    final_metrics: dict[str, float]


class Trainer:
    """
    Orquestra o loop de treinamento PyTorch com early stopping e checkpointing.

    Recebe objetos já construídos (modelo, dataloaders, tracker) e
    não é responsável por construí-los.
    """

    def __init__(
        self,
        model: ITimeSeriesModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        tracker,
        target_mean: np.ndarray | None = None,
        target_scale: np.ndarray | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tracker = tracker
        self.target_mean = target_mean
        self.target_scale = target_scale

        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        run_name = config.run_name or "run"
        self.checkpoint_path = Path(config.checkpoint_dir) / run_name / "best_model.pt"
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def fit(self) -> TrainerOutput:
        """Executa o loop de treinamento completo e retorna o resultado."""
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        final_metrics: dict[str, float] = {}

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch()
            val_metrics = self._validate()
            val_loss = val_metrics["val_loss"]

            metrics = {"train_loss": train_loss, **val_metrics}
            final_metrics = metrics
            self.tracker.log_epoch_metrics(metrics, epoch)

            logger.info(
                "Epoch %d/%d — train_loss=%.4f val_loss=%.4f mae=%.4f rmse=%.4f",
                epoch,
                self.config.epochs,
                train_loss,
                val_loss,
                val_metrics.get("mae", float("nan")),
                val_metrics.get("rmse", float("nan")),
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint()
                logger.info("Checkpoint salvo (val_loss=%.4f)", best_val_loss)
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(
                        "Early stopping na época %d (patience=%d)",
                        epoch,
                        self.config.early_stopping_patience,
                    )
                    break

        return TrainerOutput(
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            checkpoint_path=str(self.checkpoint_path),
            total_epochs_run=epoch,
            final_metrics=final_metrics,
        )

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(x)

        return total_loss / len(self.train_loader.dataset)

    def _validate(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item() * len(x)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        val_loss = total_loss / len(self.val_loader.dataset)
        preds_arr = np.concatenate(all_preds, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)
        metrics = compute_metrics(preds_arr, targets_arr)

        # O MAPE principal deve ser calculado na escala original do alvo.
        if self.target_mean is not None and self.target_scale is not None:
            mean = self.target_mean.reshape(1, 1, -1)
            scale = self.target_scale.reshape(1, 1, -1)
            preds_original = preds_arr * scale + mean
            targets_original = targets_arr * scale + mean
            metrics["mape_scaled"] = metrics["mape"]
            metrics["mape"] = compute_mape(preds_original, targets_original)

        metrics["val_loss"] = val_loss
        return metrics

    def _save_checkpoint(self) -> None:
        torch.save(self.model.state_dict(), self.checkpoint_path)
