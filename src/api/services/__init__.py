"""Serviços da API WeatherOps.

Exporta as classes de serviço que compõem a camada de negócio da API:

- ``DataService``:    Carrega e serve janelas de contexto a partir do Parquet.
- ``ModelRegistry``: Carrega e gerencia os modelos em produção do MLflow.
- ``ModelEntry``:    Contêiner de estado de runtime para um modelo servido.
- ``Predictor``:     Orquestra uma requisição de previsão de ponta a ponta.
"""
from src.api.services.data_service import DataService

__all__ = [
    "DataService",
    "ModelEntry",
    "ModelRegistry",
    "Predictor",
]

_LAZY: dict[str, tuple[str, str]] = {
    "ModelEntry": ("src.api.services.model_registry", "ModelEntry"),
    "ModelRegistry": ("src.api.services.model_registry", "ModelRegistry"),
    "Predictor": ("src.api.services.predictor", "Predictor"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib  

        mod_path, attr = _LAZY[name]
        return getattr(importlib.import_module(mod_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
