from abc import ABC, abstractmethod

import torch


class ITimeSeriesModel(ABC):
    """
    Interface base para modelos de séries temporais.

    Todos os modelos devem receber tensores com shape:
      x: (batch, seq_len, n_features)

    E retornar:
      output: (batch, horizon, n_targets)
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de entrada (batch, seq_len, n_features)

        Returns:
            Tensor de saída (batch, horizon, n_targets)
        """
        ...
