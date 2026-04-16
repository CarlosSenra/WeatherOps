"""Serviços da API WeatherOps.

Exporta as classes de serviço que compõem a camada de negócio da API:

- ``DataService``:    Carrega e serve janelas de contexto a partir do Parquet.
- ``ModelRegistry``: Carrega e gerencia os modelos em produção do MLflow.
- ``ModelEntry``:    Contêiner de estado de runtime para um modelo servido.
- ``Predictor``:     Orquestra uma requisição de previsão de ponta a ponta.
"""
from src.api.services.data_service import DataService
from src.api.services.model_registry import ModelEntry, ModelRegistry
from src.api.services.predictor import Predictor

__all__ = [
    "DataService",
    "ModelRegistry",
    "ModelEntry",
    "Predictor",
]
