"""Engine de inferência para modelos pytorch-forecasting (TFT e N-BEATS).

Ambas as arquiteturas compartilham este engine pois, em tempo de inferência,
utilizam ``TimeSeriesDataSet.from_dataset(training_dataset, window_df, predict=True)``
seguido de ``model.predict(dataloader)``. A única diferença de treinamento
(N-BEATS univariado vs. TFT multivariado) já está incorporada no objeto
``training_dataset`` armazenado no ``ModelEntry``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from src.api.engines.base import BaseInferenceEngine, InferenceContext
from src.ml_workstation.config.training_config import DataConfig
from src.ml_workstation.data.pf_loader import PytorchForecastingDataLoader

if TYPE_CHECKING:
    from src.api.config import ServingModelConfig
    from src.api.services.data_service import DataService

logger = logging.getLogger(__name__)


class PytorchForecastingEngine(BaseInferenceEngine):
    """Engine para TemporalFusionTransformer e NBeats via pytorch-forecasting.

    Metadados preparados na inicialização
    --------------------------------------
    ``training_dataset`` — objeto ``TimeSeriesDataSet`` reconstruído a partir
    do split de treinamento do Parquet. O ``GroupNormalizer`` embutido carrega
    as estatísticas de normalização ajustadas durante o treinamento e as aplica
    em ``from_dataset`` durante a inferência, garantindo consistência.

    Por que reconstruir em vez de carregar do MLflow?
    --------------------------------------------------
    O pytorch-forecasting não serializa objetos ``TimeSeriesDataSet`` como
    artefatos MLflow. Reconstruir a partir dos mesmos dados com o mesmo ratio
    de split é a abordagem canônica e acrescenta ~1-2 s ao tempo de startup.
    """

    async def prepare_metadata(
        self,
        cfg: ServingModelConfig,
        data_service: DataService,
    ) -> dict:
        """Reconstrói o ``TimeSeriesDataSet`` a partir do split de treinamento.

        Delega a construção do dataset (CPU-bound) a uma thread pool para não
        bloquear a corrotina de lifespan.
        """
        training_dataset = await asyncio.to_thread(
            self._build_training_dataset_sync, cfg, data_service
        )
        logger.info(
            "PytorchForecastingEngine: training_dataset reconstruído para '%s' "
            "(model_type=%s, horizon=%d, sequence_length=%d)",
            cfg.key,
            cfg.model_type,
            cfg.horizon,
            cfg.sequence_length,
        )
        return {"training_dataset": training_dataset}

    # ------------------------------------------------------------------
    # Implementação de BaseInferenceEngine
    # ------------------------------------------------------------------

    def prepare_input(
        self,
        window_df: pd.DataFrame,
        ctx: InferenceContext,
    ) -> Any:
        """Constrói um DataLoader de amostra única a partir da janela de contexto.

        ``from_dataset`` aplica o ``GroupNormalizer`` ajustado durante o
        treinamento — por isso o objeto ``training_dataset`` é necessário.
        Com ``predict=True``, o pytorch-forecasting cria exatamente uma amostra
        de previsão por grupo usando os últimos passos disponíveis do encoder.
        """
        from pytorch_forecasting import TimeSeriesDataSet  # importação lazy

        pred_dataset = TimeSeriesDataSet.from_dataset(
            ctx.training_dataset,
            window_df,
            predict=True,
            stop_randomization=True,
        )
        return pred_dataset.to_dataloader(
            train=False,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

    def run_inference(self, model: Any, dataloader: Any) -> np.ndarray:
        """Executa o modelo e retorna as previsões como array numpy 1-D."""
        model.eval()
        with torch.no_grad():
            # model.predict retorna um Tensor de shape (n_samples, horizon)
            preds = model.predict(dataloader, return_index=False)

        arr = preds.cpu().numpy() if hasattr(preds, "cpu") else np.asarray(preds)
        return arr.squeeze()

    # ------------------------------------------------------------------
    # Métodos auxiliares privados
    # ------------------------------------------------------------------

    def _build_training_dataset_sync(
        self,
        cfg: ServingModelConfig,
        data_service: DataService,
    ) -> Any:
        """CPU-bound: reconstrói o ``training_dataset`` a partir do split de
        treinamento do Parquet carregado. Espelha exatamente o que o
        ``PytorchForecastingDataLoader`` faz no treinamento, garantindo que
        as estatísticas do normalizador sejam idênticas."""
        train_df = data_service.get_training_slice(cfg.train_ratio)

        data_config = DataConfig(
            parquet_path="",  # não usado — os dados já estão carregados pelo DataService
            feature_columns=cfg.feature_columns,
            target_columns=cfg.target_columns,
            sequence_length=cfg.sequence_length,
            horizon=cfg.horizon,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
        )
        loader = PytorchForecastingDataLoader(data_config, cfg.model_type)
        known_reals, unknown_reals = loader._classify_features()
        return loader._build_training_dataset(train_df, known_reals, unknown_reals)
