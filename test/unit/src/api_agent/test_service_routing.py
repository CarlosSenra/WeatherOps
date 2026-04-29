"""Testes do serviço agêntico (Gemini 2.5 Flash + function calling)."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.api_agent.service import AgentRuntimeError, run_agent_chat


def _settings(api_key: str | None = "test-key") -> MagicMock:
    s = MagicMock()
    s.google_api_key = api_key
    return s


def _mock_llm_cls(side_effects: list) -> MagicMock:
    """Retorna instância mockada de ChatGoogleGenerativeAI com ainvoke sequencial."""
    llm_instance = MagicMock()
    llm_with_tools = MagicMock()
    llm_instance.bind_tools.return_value = llm_with_tools
    llm_with_tools.ainvoke = AsyncMock(side_effect=side_effects)
    return llm_instance


@pytest.mark.asyncio
async def test_run_agent_chat_no_tool_calls_returns_llm_answer() -> None:
    predictor = MagicMock()
    data_service = MagicMock()

    mock_llm = _mock_llm_cls([AIMessage(content="Resposta direta do LLM.")])

    with patch("src.api_agent.service.ChatGoogleGenerativeAI", return_value=mock_llm):
        out = await run_agent_chat("Olá", _settings(), predictor, data_service)

    assert out.answer == "Resposta direta do LLM."
    assert out.tool_calls == []


@pytest.mark.asyncio
async def test_run_agent_chat_routes_forecast() -> None:
    predictor = MagicMock()
    data_service = MagicMock()

    fc_args = {"horizon": 168, "reference_date": "2024-06-01", "model_type": "tft"}
    ai_tool = AIMessage(
        content="",
        tool_calls=[{"name": "run_weather_forecast", "args": fc_args, "id": "c1"}],
    )
    ai_final = AIMessage(content="Previsão de 168h pronta.")

    mock_llm = _mock_llm_cls([ai_tool, ai_final])

    forecast_tool_mock = MagicMock()
    forecast_tool_mock.name = "run_weather_forecast"
    forecast_tool_mock.ainvoke = AsyncMock(return_value='{"horizon":168}')

    with (
        patch("src.api_agent.service.ChatGoogleGenerativeAI", return_value=mock_llm),
        patch("src.api_agent.service.make_forecast_tool", return_value=forecast_tool_mock),
    ):
        out = await run_agent_chat("preciso de previsao 168h para 2024-06-01", _settings(), predictor, data_service)

    assert out.answer == "Previsão de 168h pronta."
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "run_weather_forecast"
    assert out.tool_calls[0].arguments["horizon"] == 168


@pytest.mark.asyncio
async def test_run_agent_chat_routes_dataset() -> None:
    predictor = MagicMock()
    data_service = MagicMock()

    ds_args = {"reference_date": "2024-06-01", "sequence_length": 24, "max_rows": 10}
    ai_tool = AIMessage(
        content="",
        tool_calls=[{"name": "summarize_dataset_window", "args": ds_args, "id": "c2"}],
    )
    ai_final = AIMessage(content="Resumo do dataset carregado.")

    mock_llm = _mock_llm_cls([ai_tool, ai_final])

    dataset_tool_mock = MagicMock()
    dataset_tool_mock.name = "summarize_dataset_window"
    dataset_tool_mock.ainvoke = AsyncMock(return_value='{"rows":24}')

    with (
        patch("src.api_agent.service.ChatGoogleGenerativeAI", return_value=mock_llm),
        patch("src.api_agent.service.make_dataset_window_tool", return_value=dataset_tool_mock),
    ):
        out = await run_agent_chat(
            "quero resumo da janela 2024-06-01 sequence_length=24 max_rows=10",
            _settings(),
            predictor,
            data_service,
        )

    assert out.answer == "Resumo do dataset carregado."
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "summarize_dataset_window"
    assert out.tool_calls[0].arguments["sequence_length"] == 24
    assert out.tool_calls[0].arguments["max_rows"] == 10


@pytest.mark.asyncio
async def test_run_agent_chat_routes_list_models() -> None:
    predictor = MagicMock()
    data_service = MagicMock()

    ai_tool = AIMessage(
        content="",
        tool_calls=[{"name": "list_available_models", "args": {}, "id": "c3"}],
    )
    ai_final = AIMessage(content="Modelos disponíveis: tft com horizontes 72, 168, 336.")

    mock_llm = _mock_llm_cls([ai_tool, ai_final])

    models_tool_mock = MagicMock()
    models_tool_mock.name = "list_available_models"
    models_tool_mock.ainvoke = AsyncMock(return_value='{"valid_horizons":[72,168,336]}')

    with (
        patch("src.api_agent.service.ChatGoogleGenerativeAI", return_value=mock_llm),
        patch("src.api_agent.service.list_available_models", models_tool_mock),
    ):
        out = await run_agent_chat("quais modelos estão disponíveis?", _settings(), predictor, data_service)

    assert out.answer == "Modelos disponíveis: tft com horizontes 72, 168, 336."
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "list_available_models"


@pytest.mark.asyncio
async def test_run_agent_chat_raises_when_api_key_missing() -> None:
    predictor = MagicMock()
    data_service = MagicMock()

    with pytest.raises(AgentRuntimeError, match="GOOGLE_API_KEY"):
        await run_agent_chat("qualquer mensagem", _settings(api_key=None), predictor, data_service)


@pytest.mark.asyncio
async def test_run_agent_chat_returns_json_in_answer() -> None:
    predictor = MagicMock()
    data_service = MagicMock()

    fc_args = {"horizon": 72, "reference_date": "2024-06-01"}
    ai_tool = AIMessage(
        content="",
        tool_calls=[{"name": "run_weather_forecast", "args": fc_args, "id": "c4"}],
    )
    ai_final = AIMessage(content='{"horizon":72}')

    mock_llm = _mock_llm_cls([ai_tool, ai_final])

    forecast_tool_mock = MagicMock()
    forecast_tool_mock.name = "run_weather_forecast"
    forecast_tool_mock.ainvoke = AsyncMock(return_value='{"horizon":72}')

    with (
        patch("src.api_agent.service.ChatGoogleGenerativeAI", return_value=mock_llm),
        patch("src.api_agent.service.make_forecast_tool", return_value=forecast_tool_mock),
    ):
        out = await run_agent_chat("forecast 72h", _settings(), predictor, data_service)

    parsed = json.loads(out.answer)
    assert parsed["horizon"] == 72
