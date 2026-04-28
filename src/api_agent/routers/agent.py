"""Router FastAPI — agente meteorológico."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.config import get_settings
from src.api.dependencies import get_data_service, get_predictor
from src.api.metrics import agent_chat_duration_seconds, agent_chat_requests_total
from src.api.services.data_service import DataService
from src.api.services.predictor import Predictor
from src.api_agent.schemas import AgentChatRequest, AgentChatResponse
from src.api_agent.service import AgentRuntimeError, run_agent_chat

router = APIRouter(prefix="/v1/agent", tags=["agent"])


@router.post(
    "/chat",
    response_model=AgentChatResponse,
    summary="Agente meteorológico simplificado (roteamento de tools sem LLM)",
    responses={
        422: {"description": "Mensagem inválida ou acima do limite configurado."},
        400: {"description": "Mensagem sem correspondência para tools suportadas."},
    },
)
async def agent_chat(
    http_request: Request,
    body: AgentChatRequest,
    predictor: Predictor = Depends(get_predictor),
    data_service: DataService = Depends(get_data_service),
) -> AgentChatResponse:
    settings = get_settings()

    msg = body.message.strip()
    if not msg:
        raise HTTPException(status_code=422, detail="message não pode ser vazio.")
    if len(msg) > settings.agent_max_message_chars:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Mensagem excede agent_max_message_chars={settings.agent_max_message_chars}. "
                "Encurte o pedido ou ajuste a configuração."
            ),
        )

    t0 = time.perf_counter()
    try:
        out = await run_agent_chat(msg, settings, predictor, data_service)
    except AgentRuntimeError as e:
        agent_chat_requests_total.labels(status="error").inc()
        raise HTTPException(status_code=400, detail=e.public_message) from e
    except HTTPException:
        raise
    except Exception as e:
        agent_chat_requests_total.labels(status="error").inc()
        raise HTTPException(
            status_code=500,
            detail="Falha interna ao executar o agente. Verifique logs da API para diagnostico detalhado.",
        ) from e
    finally:
        dt = time.perf_counter() - t0
        agent_chat_duration_seconds.observe(dt)
    agent_chat_requests_total.labels(status="success").inc()
    return out
