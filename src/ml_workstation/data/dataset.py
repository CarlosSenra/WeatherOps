import numpy as np
import torch
from torch.utils.data import Dataset

from src.ml_workstation.config.training_config import DataConfig


class WeatherSequenceDataset(Dataset):
    """
    Dataset de sequências temporais para previsão meteorológica.

    Dado um array 2D (T, n_features), gera pares (X, y) onde:
      - X: janela de look-back  → shape (sequence_length, n_features)
      - y: horizonte de previsão → shape (horizon, n_targets)

    O índice 'i' corresponde à janela [i : i+sequence_length] como entrada
    e [i+sequence_length : i+sequence_length+horizon] como alvo.
    """

    def __init__(self, data: np.ndarray, config: DataConfig, target_indices: list[int]):
        """
        Args:
            data: Array normalizado (T, n_features).
            config: DataConfig com sequence_length, horizon.
            target_indices: Índices das colunas alvo dentro de `data`.
        """
        self.data = torch.from_numpy(data.astype(np.float32))
        self.sequence_length = config.sequence_length
        self.horizon = config.horizon
        self.target_indices = target_indices

    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - self.horizon + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.sequence_length]
        y = self.data[
            idx + self.sequence_length : idx + self.sequence_length + self.horizon,
            self.target_indices,
        ]
        return x, y
