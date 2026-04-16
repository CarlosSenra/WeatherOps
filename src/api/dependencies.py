"""Injetores de dependência do FastAPI.

Todos os injetores leem o estado armazenado em ``app.state`` pelo evento de
lifespan em ``main.py``. Esse padrão facilita a substituição de dependências
em testes, bastando atribuir diretamente em ``app.state``.
"""
from __future__ import annotations

from fastapi import Request

from src.api.services.data_service import DataService
from src.api.services.model_registry import ModelRegistry
from src.api.services.predictor import Predictor


def get_registry(request: Request) -> ModelRegistry:
    """Retorna o ``ModelRegistry`` armazenado no estado da aplicação."""
    return request.app.state.registry


def get_data_service(request: Request) -> DataService:
    """Retorna o ``DataService`` armazenado no estado da aplicação."""
    return request.app.state.data_service


def get_predictor(request: Request) -> Predictor:
    """Retorna o ``Predictor`` armazenado no estado da aplicação."""
    return request.app.state.predictor
