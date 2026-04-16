"""Endpoints de verificação de saúde da API.

GET /health       — probe de liveness. Sempre retorna 200 se o processo estiver
                    em execução. Usado por orquestradores de contêineres para
                    decidir se o pod deve ser reiniciado.

GET /health/ready — probe de readiness. Retorna 200 apenas quando todos os
                    modelos terminaram de carregar e o DataService está populado.
                    Usado por balanceadores de carga para decidir se devem rotear
                    tráfego para o pod. Retorna 503 com um corpo JSON listando
                    modelos carregados e falhos quando não estiver pronto.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get("/health", summary="Probe de liveness")
async def liveness() -> JSONResponse:
    """Verifica se o processo está em execução.

    Retorna sempre HTTP 200 enquanto o processo estiver ativo. Não verifica
    o estado dos modelos ou dados — use ``/health/ready`` para isso.
    """
    return JSONResponse({"status": "ok"})


@router.get("/health/ready", summary="Probe de readiness")
async def readiness(request: Request) -> JSONResponse:
    """Verifica se a API está pronta para receber requisições de inferência.

    Retorna HTTP 200 apenas quando o DataService está carregado e todos os
    modelos registrados foram carregados com sucesso. Retorna HTTP 207 se
    algum modelo falhou ao carregar (degradado) e HTTP 503 enquanto o
    carregamento ainda está em progresso.
    """
    registry = request.app.state.registry
    data_service = request.app.state.data_service

    data_ready = data_service.is_ready
    models_ready = registry.is_ready()
    loaded = registry.list_models()
    failed = registry.failed_models()

    payload = {
        "data_service": "pronto" if data_ready else "carregando",
        "models_loaded": [m["key"] for m in loaded],
        "models_failed": failed,
        "row_count": data_service.row_count,
    }

    if data_ready and models_ready and not failed:
        return JSONResponse({"status": "pronto", **payload})

    if failed:
        payload["status"] = "degradado"
        return JSONResponse(payload, status_code=207)

    payload["status"] = "carregando"
    return JSONResponse(payload, status_code=503)
