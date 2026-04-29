import torch
import torch.nn as nn

from src.ml_workstation.config.training_config import ModelConfig
from src.ml_workstation.models.interface import ITimeSeriesModel


class WeatherLSTM(nn.Module, ITimeSeriesModel):
    """
    LSTM empilhado para previsão de séries temporais meteorológicas.

    Arquitetura:
        LSTM(n_features → hidden_size, num_layers) → Linear(hidden_size → horizon * n_targets)

    O último estado oculto do LSTM é projetado para a previsão.
    """

    def __init__(self, n_features: int, n_targets: int, horizon: int, config: ModelConfig):
        """
        Args:
            n_features: Número de features de entrada.
            n_targets: Número de variáveis alvo.
            horizon: Número de passos à frente para prever.
            config: Configuração do modelo (hidden_size, num_layers, dropout).
        """
        super().__init__()
        self.horizon = horizon
        self.n_targets = n_targets

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(config.hidden_size, horizon * n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)

        Returns:
            (batch, horizon, n_targets)
        """
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.head(last_hidden)
        return out.view(-1, self.horizon, self.n_targets)
