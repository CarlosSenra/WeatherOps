from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.ml_workstation.config.training_config import TrainingConfig
from src.ml_workstation.training.trainer import Trainer


class TinyRegressor(nn.Module):
    def __init__(self, seq_len: int, n_features: int, horizon: int, n_targets: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.horizon = horizon
        self.n_targets = n_targets
        self.linear = nn.Linear(seq_len * n_features, horizon * n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        out = self.linear(x.reshape(batch, -1))
        return out.reshape(batch, self.horizon, self.n_targets)


class TrackerStub:
    def __init__(self) -> None:
        self.logged = []

    def log_epoch_metrics(self, metrics: dict[str, float], epoch: int) -> None:
        self.logged.append((epoch, metrics))


def _make_loader(num_samples: int, seq_len: int, n_features: int, horizon: int, n_targets: int) -> DataLoader:
    x = torch.randn(num_samples, seq_len, n_features)
    y = torch.randn(num_samples, horizon, n_targets)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=4, shuffle=False)


def test_trainer_fit_saves_checkpoint_and_logs_metrics(tmp_path) -> None:
    seq_len, n_features, horizon, n_targets = 5, 3, 2, 1
    train_loader = _make_loader(16, seq_len, n_features, horizon, n_targets)
    val_loader = _make_loader(8, seq_len, n_features, horizon, n_targets)

    model = TinyRegressor(seq_len, n_features, horizon, n_targets)
    tracker = TrackerStub()
    config = TrainingConfig(
        run_name="unit_trainer",
        epochs=3,
        early_stopping_patience=2,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu",
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        tracker=tracker,
    )

    output = trainer.fit()

    assert output.best_epoch >= 1
    assert output.total_epochs_run >= 1
    assert Path(output.checkpoint_path).exists()
    assert len(tracker.logged) >= 1
    assert "val_loss" in output.final_metrics


def test_validate_computes_mape_on_original_scale(tmp_path) -> None:
    seq_len, n_features, horizon, n_targets = 4, 2, 2, 1
    train_loader = _make_loader(8, seq_len, n_features, horizon, n_targets)
    val_loader = _make_loader(8, seq_len, n_features, horizon, n_targets)

    model = TinyRegressor(seq_len, n_features, horizon, n_targets)
    tracker = TrackerStub()
    config = TrainingConfig(
        run_name="scale_metrics",
        epochs=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu",
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        tracker=tracker,
        target_mean=np.array([10.0]),
        target_scale=np.array([2.0]),
    )

    metrics = trainer._validate()

    assert "mape" in metrics
    assert "mape_scaled" in metrics
    assert "val_loss" in metrics
