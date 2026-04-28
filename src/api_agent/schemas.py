"""Esquemas Pydantic do endpoint do agente."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"message": "Olá, como estará a temperatura pela tarde do dia 2024-06-01?"}]}
    )

    message: str = Field(
        ...,
        min_length=1,
        description="Pergunta ou instrução em linguagem natural (PT-BR).",
    )


class AgentChatResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    rag_context_snippets: list[str] = Field(
        default_factory=list,
        description="Campo legado; mantido vazio no agente simplificado sem RAG.",
    )
