"""Testes do router /v1/agent/chat."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("langgraph")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api_agent.routers.agent import router
from src.api_agent.schemas import AgentChatResponse


def _app_with_state() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.predictor = MagicMock()
    app.state.data_service = MagicMock()
    return app


def test_agent_chat_503_when_no_api_key() -> None:
    client = TestClient(_app_with_state())
    fake = MagicMock()
    fake.gemini_api_key = None
    fake.agent_max_message_chars = 8000
    with patch("src.api_agent.routers.agent.get_settings", return_value=fake):
        r = client.post("/v1/agent/chat", json={"message": "Olá"})
    assert r.status_code == 503


def test_agent_chat_503_when_api_key_blank() -> None:
    client = TestClient(_app_with_state())
    fake = MagicMock()
    fake.gemini_api_key = "   "
    fake.agent_max_message_chars = 8000
    with patch("src.api_agent.routers.agent.get_settings", return_value=fake):
        r = client.post("/v1/agent/chat", json={"message": "Olá"})
    assert r.status_code == 503


def test_agent_chat_422_empty_message() -> None:
    client = TestClient(_app_with_state())
    fake = MagicMock()
    fake.gemini_api_key = "x"
    fake.agent_max_message_chars = 8000
    with patch("src.api_agent.routers.agent.get_settings", return_value=fake):
        r = client.post("/v1/agent/chat", json={"message": ""})
    assert r.status_code == 422


def test_agent_chat_422_whitespace_only_message() -> None:
    client = TestClient(_app_with_state())
    fake = MagicMock()
    fake.gemini_api_key = "x"
    fake.agent_max_message_chars = 8000
    with patch("src.api_agent.routers.agent.get_settings", return_value=fake):
        r = client.post("/v1/agent/chat", json={"message": "   \t  "})
    assert r.status_code == 422


def test_agent_chat_422_message_too_long() -> None:
    client = TestClient(_app_with_state())
    fake = MagicMock()
    fake.gemini_api_key = "x"
    fake.agent_max_message_chars = 5
    with patch("src.api_agent.routers.agent.get_settings", return_value=fake):
        r = client.post("/v1/agent/chat", json={"message": "123456"})
    assert r.status_code == 422


def test_agent_chat_success_mocked() -> None:
    client = TestClient(_app_with_state())
    fake = MagicMock()
    fake.gemini_api_key = "fake-key"
    fake.agent_max_message_chars = 8000

    async def _fake_run(*_a, **_k):
        return AgentChatResponse(
            answer="Resposta simulada.",
            tool_calls=[],
            rag_context_snippets=[],
        )

    with (
        patch("src.api_agent.routers.agent.get_settings", return_value=fake),
        patch("src.api_agent.routers.agent.run_agent_chat", new=AsyncMock(side_effect=_fake_run)),
    ):
        r = client.post("/v1/agent/chat", json={"message": "Qual o horizonte máximo?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Resposta simulada."
