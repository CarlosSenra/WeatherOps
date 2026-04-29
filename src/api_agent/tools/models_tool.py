"""Tool que lista modelos e horizontes disponíveis na API WeatherOps."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from src.api.config import REGISTERED_MODELS, VALID_HORIZONS, VALID_MODEL_TYPES

_HORIZON_LABELS = {72: "3 dias", 168: "7 dias", 336: "14 dias"}


@tool
def list_available_models() -> str:
    """Lista os horizontes de previsão e tipos de modelo disponíveis na API WeatherOps."""
    models = [
        {
            "key": cfg.key,
            "model_type": cfg.model_type,
            "horizon_hours": cfg.horizon,
            "horizon_label": _HORIZON_LABELS.get(cfg.horizon, f"{cfg.horizon}h"),
            "sequence_length_hours": cfg.sequence_length,
        }
        for cfg in REGISTERED_MODELS
    ]
    return json.dumps(
        {
            "valid_horizons": sorted(VALID_HORIZONS),
            "valid_model_types": sorted(VALID_MODEL_TYPES),
            "models": models,
        },
        ensure_ascii=False,
    )
