from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.api.schemas.common import ForecastPoint

if TYPE_CHECKING:
    from src.api.config import ServingModelConfig
    from src.api.services.data_service import DataService


@dataclass
class InferenceContext:
    """Contêiner tipado com dados de runtime específicos do engine.

    Transportado do ``ModelRegistry`` pelo ``Predictor`` até os métodos do
    engine. Cada engine lê apenas os campos de que precisa e ignora os demais,
    mantendo o Predictor agnóstico aos internos de cada engine.
    """

    training_dataset: Any | None
    """``TimeSeriesDataSet`` do pytorch-forecasting (apenas para ``PytorchForecastingEngine``)."""

    scaler: Any | None
    """``StandardScaler`` ajustado nos dados de treinamento (apenas para ``SequentialEngine``)."""

    horizon: int
    sequence_length: int
    group_id: str
    target_columns: list[str]
    feature_columns: list[str]


class BaseInferenceEngine(abc.ABC):
    """Interface estratégia para pipelines de inferência específicos por família de modelo.

    Cada engine concreto lida com uma família de modelos (pytorch-forecasting,
    tensores sequenciais, ONNX, etc.). O Predictor delega todas as etapas de
    preparação de dados e inferência ao engine, permanecendo agnóstico ao
    framework subjacente.

    Ciclo de vida
    -------------
    ``prepare_metadata``  — chamado uma vez na inicialização; retorna um ``dict``
                            armazenado em ``ModelEntry.metadata``.
    ``prepare_input``     — bloqueante; converte uma janela DataFrame no formato
                            esperado pelo modelo.
    ``run_inference``     — bloqueante; executa o modelo e retorna um array
                            float 1-D de comprimento ``horizon``.
    ``postprocess``       — converte o array bruto em uma lista de ``ForecastPoint``.
                            A implementação padrão adiciona timestamps horários;
                            sobrescreva para modelos com cadência diferente ou
                            múltiplos targets.
    """

    @abc.abstractmethod
    async def prepare_metadata(
        self,
        cfg: ServingModelConfig,
        data_service: DataService,
    ) -> dict:
        """Chamado UMA VEZ na inicialização. Retorna o dicionário de metadados
        específico do engine, armazenado em ``ModelEntry.metadata``
        (ex.: ``training_dataset``, ``scaler``).

        Operações de I/O ou CPU intensivo devem ser delegadas a
        ``asyncio.to_thread`` dentro deste método para não bloquear o
        event loop.
        """

    @abc.abstractmethod
    def prepare_input(
        self,
        window_df: pd.DataFrame,
        ctx: InferenceContext,
    ) -> Any:
        """Bloqueante. Converte a janela DataFrame no formato que ``run_inference``
        espera (DataLoader, Tensor, etc.)."""

    @abc.abstractmethod
    def run_inference(self, model: Any, model_input: Any) -> np.ndarray:
        """Bloqueante. Executa o modelo e retorna um array float de shape
        ``(horizon,)`` com as previsões."""

    def postprocess(
        self,
        raw: np.ndarray,
        reference_date: datetime,
        horizon: int,
    ) -> list[ForecastPoint]:
        """Converte previsões brutas em ``ForecastPoint`` com timestamps.

        Padrão: timestamps horários iniciando em ``reference_date + 1h``.
        Sobrescreva este método para cadência não horária ou modelos
        com múltiplos targets.
        """
        return [
            ForecastPoint(
                timestamp=reference_date + timedelta(hours=i + 1),
                temp_ar_c=float(raw[i]),
            )
            for i in range(min(horizon, len(raw)))
        ]
