import math

import torch
import torch.nn as nn

from src.ml_workstation.config.training_config import ModelConfig
from src.ml_workstation.models.interface import ITimeSeriesModel


class _PositionalEncoding(nn.Module):
    """Codificação posicional sinusoidal fixa (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, dropout: float, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class WeatherTransformer(nn.Module, ITimeSeriesModel):
    """
    Transformer encoder para previsão de séries temporais meteorológicas.

    Arquitetura:
        Linear(n_features → hidden_size)
        → PositionalEncoding
        → TransformerEncoder(num_layers, num_heads)
        → mean-pool sobre a dimensão temporal
        → Linear(hidden_size → horizon * n_targets)

    Mean-pooling é preferido ao last-token para sequências curtas (24h),
    pois agrega informação de todos os passos de forma estável.
    """

    def __init__(self, n_features: int, n_targets: int, horizon: int, config: ModelConfig):
        """
        Args:
            n_features: Número de features de entrada.
            n_targets: Número de variáveis alvo.
            horizon: Número de passos à frente para prever.
            config: Configuração do modelo (hidden_size, num_layers, num_heads, ffn_dim, dropout).
        """
        super().__init__()
        self.horizon = horizon
        self.n_targets = n_targets

        self.input_proj = nn.Linear(n_features, config.hidden_size)
        self.pos_enc = _PositionalEncoding(config.hidden_size, config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Linear(config.hidden_size, horizon * n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)

        Returns:
            (batch, horizon, n_targets)
        """
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        out = self.head(x)
        return out.view(-1, self.horizon, self.n_targets)
