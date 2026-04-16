"""Registro de engines — mapeia strings ``engine_class`` para suas classes de implementação.

Para adicionar uma nova família de engines:
1. Crie ``engines/seu_engine.py`` implementando ``BaseInferenceEngine``.
2. Adicione uma linha ao ``ENGINE_REGISTRY`` abaixo.
3. Referencie ``engine_class="seu_engine"`` nas entradas de ``ServingModelConfig``.
"""
from __future__ import annotations

from src.api.engines.base import BaseInferenceEngine, InferenceContext
from src.api.engines.pytorch_forecasting import PytorchForecastingEngine
from src.api.engines.sequential import SequentialEngine

ENGINE_REGISTRY: dict[str, type[BaseInferenceEngine]] = {
    "pytorch_forecasting": PytorchForecastingEngine,
    "sequential": SequentialEngine,
    # Famílias de engines futuras:
    # "onnx": OnnxEngine,
    # "triton": TritonEngine,
}

__all__ = [
    "BaseInferenceEngine",
    "InferenceContext",
    "PytorchForecastingEngine",
    "SequentialEngine",
    "ENGINE_REGISTRY",
]
