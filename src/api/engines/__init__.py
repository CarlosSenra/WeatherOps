"""Registro de engines — mapeia strings ``engine_class`` para suas classes de implementação.

Para adicionar uma nova família de engines:
1. Crie ``engines/seu_engine.py`` implementando ``BaseInferenceEngine``.
2. Adicione uma linha ao ``ENGINE_REGISTRY`` abaixo.
3. Referencie ``engine_class="seu_engine"`` nas entradas de ``ServingModelConfig``.
"""
from __future__ import annotations

from src.api.engines.base import BaseInferenceEngine, InferenceContext

__all__ = [
    "BaseInferenceEngine",
    "InferenceContext",
    "PytorchForecastingEngine",
    "SequentialEngine",
    "ENGINE_REGISTRY",
]

_LAZY: dict[str, tuple[str, str]] = {
    "PytorchForecastingEngine": ("src.api.engines.pytorch_forecasting", "PytorchForecastingEngine"),
    "SequentialEngine": ("src.api.engines.sequential", "SequentialEngine"),
}


def __getattr__(name: str):
    import importlib  # noqa: PLC0415

    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        return getattr(importlib.import_module(mod_path), attr)
    if name == "ENGINE_REGISTRY":
        PytorchForecastingEngine = __getattr__("PytorchForecastingEngine")
        SequentialEngine = __getattr__("SequentialEngine")
        return {
            "pytorch_forecasting": PytorchForecastingEngine,
            "sequential": SequentialEngine,
        }
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseInferenceEngine",
    "InferenceContext",
    "PytorchForecastingEngine",
    "SequentialEngine",
    "ENGINE_REGISTRY",
]
