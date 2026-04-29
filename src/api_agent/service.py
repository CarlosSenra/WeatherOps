"""Orquestração do agente WeatherOps via Gemini 2.5 Flash (LangChain function calling)."""
from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.api.metrics import (
    agent_rag_queries_total,
    agent_tool_calls_total,
    agent_tool_duration_seconds,
)
from src.api.services.data_service import DataService
from src.api.services.predictor import Predictor
from src.api_agent.guardrails import check_output
from src.api_agent.schemas import AgentChatResponse, ToolCallRecord
from src.api_agent.tools import (
    list_available_models,
    make_dataset_window_tool,
    make_forecast_tool,
    make_historical_context_tool,
    make_period_forecast_tool,
)

_logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 5

_SYSTEM_PROMPT = """Você é um assistente meteorológico da API WeatherOps. Responda sempre em português.

Regras para escolha de tool:
- Quando o usuário perguntar sobre temperatura em um período do dia (manhã, tarde, noite, madrugada),
  use SEMPRE get_forecast_by_period — ela calcula o horizonte automaticamente e filtra as horas certas.
  Nunca pergunte o horizonte ao usuário neste caso.
- Use run_weather_forecast apenas quando o usuário especificar explicitamente um horizonte (72h, 168h ou 336h).
- Use list_available_models para explicar opções disponíveis.
- Use summarize_dataset_window para dúvidas sobre dados históricos observados.
- Use get_historical_weather_context quando o usuário perguntar sobre o clima típico de um mês ou
  período do ano, ou para enriquecer uma resposta de previsão com contexto histórico sazonal.

Períodos do dia:
- manhã: 07h–12h
- tarde: 12h–17h
- noite: 17h–00h
- madrugada: 00h–07h
"""


class AgentRuntimeError(RuntimeError):
    """Erro operacional do agente com mensagem segura para retorno HTTP."""

    def __init__(self, public_message: str) -> None:
        super().__init__(public_message)
        self.public_message = public_message


async def run_agent_chat(
    message: str,
    settings: object,
    predictor: Predictor,
    data_service: DataService,
    retriever=None,
) -> AgentChatResponse:
    api_key = getattr(settings, "google_api_key", None)
    if not api_key:
        raise AgentRuntimeError(
            "GOOGLE_API_KEY não configurada. Defina a variável de ambiente antes de usar o agente."
        )

    tools = [
        make_period_forecast_tool(predictor),
        make_forecast_tool(predictor),
        make_dataset_window_tool(data_service),
        list_available_models,
    ]
    if retriever is not None:
        tools.append(make_historical_context_tool(retriever))

    tool_map = {t.name: t for t in tools}

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    llm_with_tools = llm.bind_tools(tools)

    messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=message)]
    tool_calls_made: list[ToolCallRecord] = []
    rag_snippets: list[str] = []

    response = None
    for _ in range(_MAX_ITERATIONS):
        try:
            response = await llm_with_tools.ainvoke(messages)
        except Exception as exc:
            raise AgentRuntimeError(f"Falha na chamada ao LLM: {exc}") from exc

        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]

            if tool_name not in tool_map:
                if tool_name == "get_historical_weather_context":
                    agent_rag_queries_total.labels(status="unavailable").inc()
                messages.append(
                    ToolMessage(
                        content=f"Tool '{tool_name}' não encontrada.",
                        tool_call_id=tool_id,
                    )
                )
                continue

            _t_tool = time.perf_counter()
            try:
                result = await tool_map[tool_name].ainvoke(tool_args)
            except Exception as exc:
                result = f'{{"error": "execucao_tool", "detail": "{exc}"}}'
            finally:
                agent_tool_calls_total.labels(tool_name=tool_name).inc()
                agent_tool_duration_seconds.labels(tool_name=tool_name).observe(
                    time.perf_counter() - _t_tool
                )

            tool_calls_made.append(ToolCallRecord(name=tool_name, arguments=tool_args))
            if tool_name == "get_historical_weather_context" and isinstance(result, str):
                snippets = [s for s in result.split("\n\n") if s.strip()]
                if snippets and "não disponível" not in result and "não encontrado" not in result:
                    rag_snippets.extend(snippets)
                    agent_rag_queries_total.labels(status="hit").inc()
                else:
                    agent_rag_queries_total.labels(status="miss").inc()
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

    if response is None:
        raise AgentRuntimeError("O agente não produziu resposta.")

    if response.content:
        out_guard = check_output(response.content)
        if not out_guard.passed:
            _logger.warning(
                "Output guardrail triggered: threat=%s reason=%s",
                out_guard.threat_type,
                out_guard.reason,
            )

    return AgentChatResponse(
        answer=response.content,
        tool_calls=tool_calls_made,
        rag_context_snippets=rag_snippets,
    )
