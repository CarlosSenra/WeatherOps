"""Roteador de previsão — POST /v1/forecast/{horizon}.

O roteador é intencionalmente simples:
- Valida o parâmetro de path ``horizon``.
- Verifica o cache (ignora inferência se a mesma requisição foi atendida recentemente).
- Delega ao ``Predictor.predict`` e armazena o resultado no cache.

Adicionando um novo horizonte
------------------------------
Adicione o inteiro em ``VALID_HORIZONS`` em ``config.py``. Nenhuma alteração
no roteador é necessária — o path usa um parâmetro ``int`` genérico.

Adicionando um novo tipo de modelo
------------------------------------
Adicione o literal de string em ``ModelType`` em ``schemas/forecast.py``.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.cache import get_cache
from src.api.config import VALID_HORIZONS
from src.api.dependencies import get_predictor
from src.api.metrics import cache_hits, cache_misses, inference_duration
from src.api.schemas.forecast import ForecastRequest, ForecastResponse
from src.api.services.predictor import Predictor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/forecast", tags=["forecast"])


@router.post(
    "/{horizon}",
    response_model=ForecastResponse,
    summary="Previsão horária de temperatura",
    description=(
        "**Path — `horizon`**\n\n"
        "Número de horas a prever (ex.: `72` = 3 dias). O valor deve estar entre os "
        "horizontes suportados pela API (configurados em `VALID_HORIZONS`).\n\n"
        "**Corpo — `reference_date`**\n\n"
        "Último instante do histórico a usar (inclusivo). A primeira previsão corresponde "
        "à hora seguinte: `reference_date + 1h`. Pode enviar só a data (`YYYY-MM-DD`, "
        "interpretada como 00:00 desse dia) ou data e hora em ISO 8601.\n\n"
        "**Resposta**\n\n"
        "Lista `predictions` com `horizon` pontos horários e temperatura prevista (`temp_ar_c`)."
    ),
    responses={
        404: {"description": "Modelo não carregado (não promovido para produção no MLflow)."},
        422: {
            "description": (
                "Pedido inválido: horizonte não suportado, `reference_date` em formato não aceite, "
                "ou dados insuficientes no Parquet antes do instante de referência."
            )
        },
        500: {"description": "Erro interno durante a inferência."},
    },
)
async def forecast(
    horizon: int,
    request_body: ForecastRequest,
    http_request: Request,
    predictor: Predictor = Depends(get_predictor),
) -> ForecastResponse:
    if horizon not in VALID_HORIZONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"horizon={horizon} não é suportado. "
                f"Horizontes válidos: {sorted(VALID_HORIZONS)}."
            ),
        )

    cache = get_cache()
    cache_key = cache.make_key(
        request_body.model_type,
        str(horizon),
        request_body.reference_date.isoformat(),
        request_body.group_id,
    )

    cached = cache.get(cache_key)
    if cached is not None:
        cache_hits.inc()
        trace_id = getattr(http_request.state, "trace_id", "")
        logger.debug("Cache hit para chave %s (trace_id=%s)", cache_key, trace_id)
        return ForecastResponse.model_validate(cached)

    cache_misses.inc()
    response = await predictor.predict(horizon=horizon, request=request_body)
    model_key = f"{request_body.model_type}_{horizon}"
    inference_duration.labels(model_key=model_key).observe(response.latency_ms / 1000)
    cache.set(cache_key, response.model_dump(mode="json"))
    return response
