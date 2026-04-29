"""Predictor — orquestrador entre a camada de API e a camada de engines.

O Predictor sabe *o que* fazer (resolver o modelo, buscar a janela de contexto,
chamar o engine), mas não *como* cada engine funciona internamente. Toda a
lógica específica do engine vive em ``engines/``.
"""
from __future__ import annotations

import asyncio
import logging
import time

from fastapi import HTTPException

from src.api.engines.base import InferenceContext
from src.api.schemas.forecast import ForecastRequest, ForecastResponse
from src.api.services.data_service import DataService
from src.api.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class Predictor:
    """Orquestra uma requisição de previsão de ponta a ponta.

    Fluxo
    -----
    1. Resolve ``model_key`` a partir de ``(model_type, horizon)``.
    2. Busca a janela de contexto no ``DataService``.
    3. Constrói o ``InferenceContext`` com os metadados específicos do engine.
    4. Delega ``engine.prepare_input`` e ``engine.run_inference`` a uma thread
       pool (ambos são operações bloqueantes de CPU/IO).
    5. Chama ``engine.postprocess`` (barato; executa no event loop).
    6. Retorna um ``ForecastResponse``.
    """

    def __init__(self, registry: ModelRegistry, data_service: DataService) -> None:
        self._registry = registry
        self._data_service = data_service

    async def predict(
        self,
        horizon: int,
        request: ForecastRequest,
    ) -> ForecastResponse:
        """Executa a inferência e retorna a resposta de previsão.

        Argumentos:
            horizon: Horizonte de previsão em horas.
            request: Corpo da requisição validado pelo schema ``ForecastRequest``.

        Retorna:
            ``ForecastResponse`` com previsões horárias e metadados de governança.

        Levanta:
            HTTPException 404: Se o modelo não estiver disponível no registry.
            HTTPException 422: Se não houver dados suficientes antes de ``reference_date``.
            HTTPException 500: Se a inferência falhar por qualquer outro motivo.
        """
        model_key = f"{request.model_type}_{horizon}"

        try:
            entry = self._registry.get(model_key)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        try:
            window_df = self._data_service.get_context_window(
                reference_date=request.reference_date,
                sequence_length=entry.config.sequence_length,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        ctx = InferenceContext(
            training_dataset=entry.metadata.get("training_dataset"),
            scaler=entry.metadata.get("scaler"),
            horizon=horizon,
            sequence_length=entry.config.sequence_length,
            group_id=request.group_id,
            target_columns=entry.config.target_columns,
            feature_columns=entry.config.feature_columns,
        )

        t0 = time.perf_counter()

        try:
            model_input = await asyncio.to_thread(
                entry.engine.prepare_input, window_df, ctx
            )
            raw_output = await asyncio.to_thread(
                entry.engine.run_inference, entry.model, model_input
            )
        except Exception as exc:
            logger.error(
                "Falha na inferência do modelo '%s' em %s: %s",
                model_key,
                request.reference_date,
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Erro de inferência para o modelo '{model_key}': {exc}",
            ) from exc

        latency_ms = (time.perf_counter() - t0) * 1000

        points = entry.engine.postprocess(raw_output, request.reference_date, horizon)

        logger.info(
            "Previsão concluída: modelo='%s' reference_date='%s' horizon=%d latency_ms=%.1f",
            model_key,
            request.reference_date.isoformat(),
            horizon,
            latency_ms,
        )

        return ForecastResponse(
            reference_date=request.reference_date,
            horizon=horizon,
            model_type=request.model_type,
            model_version=entry.version,
            mape=entry.mape,
            predictions=points,
            latency_ms=round(latency_ms, 2),
        )
