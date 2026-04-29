"""Tool LangChain para busca de contexto histórico de clima via RAG."""

from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def make_historical_context_tool(retriever):
    """Cria a tool de contexto histórico injetando o retriever via closure."""

    @tool
    async def get_historical_weather_context(query: str) -> str:
        """Busca padrões históricos de temperatura e clima do município monitorado.

        Use antes de fazer análises sazonais, comparações com histórico ou quando
        o usuário perguntar sobre o clima típico de um mês ou estação do ano.
        Não use para previsões futuras — use run_weather_forecast para isso.

        Args:
            query: Descrição do contexto desejado (ex: "temperatura em julho", "chuva no verão").
        """
        try:
            chunks: list[str] = await asyncio.to_thread(retriever.search, query)
            if not chunks:
                return "Nenhum dado histórico encontrado para esse contexto."
            return "\n\n".join(chunks)
        except Exception as exc:
            logger.warning("get_historical_weather_context falhou: %s", exc)
            return "Contexto histórico não disponível no momento."

    return get_historical_weather_context
