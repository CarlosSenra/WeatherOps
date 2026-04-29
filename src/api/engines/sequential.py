"""Engine de inferência para modelos sequenciais (tensor puro): LSTM e Transformer.

Esses modelos operam sobre tensores PyTorch simples, normalizados por um
``StandardScaler`` armazenado como artefato MLflow ``scaler.pkl`` na run
promovida para produção.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from src.api.engines.base import BaseInferenceEngine, InferenceContext
from src.ml_workstation.promotion.loader import load_production_scaler

if TYPE_CHECKING:
    from src.api.config import ServingModelConfig
    from src.api.services.data_service import DataService

logger = logging.getLogger(__name__)


class SequentialEngine(BaseInferenceEngine):
    """Engine para modelos LSTM e Transformer.

    Formato de entrada
    ------------------
    Tensor de shape ``(1, sequence_length, n_features)`` produzido aplicando
    o ``StandardScaler`` à janela de contexto e selecionando as últimas
    ``sequence_length`` linhas.

    Formato de saída
    ----------------
    O modelo retorna ``(1, horizon, n_targets)``; o resultado é comprimido
    para ``(horizon,)`` antes do pós-processamento.

    Scaler
    ------
    O scaler é carregado do artefato MLflow ``scaler.pkl`` anexado à run
    de produção. É armazenado em ``ModelEntry.metadata["scaler"]`` e
    acessado via ``InferenceContext.scaler``.
    """

    async def prepare_metadata(
        self,
        cfg: ServingModelConfig,
        data_service: DataService,
    ) -> dict:
        """Baixa o ``scaler.pkl`` da run MLflow promovida para produção."""
        scaler = await asyncio.to_thread(load_production_scaler, cfg.experiment_name)
        logger.info(
            "SequentialEngine: scaler carregado para '%s'",
            cfg.key,
        )
        return {"scaler": scaler}

    def prepare_input(
        self,
        window_df: pd.DataFrame,
        ctx: InferenceContext,
    ) -> torch.Tensor:
        """Normaliza a janela de contexto e retorna um tensor PyTorch 3-D."""
        arr = ctx.scaler.transform(window_df[ctx.feature_columns].values)
        # Seleciona as últimas sequence_length linhas caso window_df seja maior
        arr = arr[-ctx.sequence_length :]
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

    def run_inference(self, model: Any, x: torch.Tensor) -> np.ndarray:
        """Executa o modelo e retorna as previsões como array numpy 1-D."""
        model.eval()
        with torch.no_grad():
            out = model(x)  # (1, horizon, n_targets)
        return out.squeeze().cpu().numpy()
