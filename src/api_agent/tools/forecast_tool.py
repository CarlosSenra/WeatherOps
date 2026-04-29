"""Tool de previsão via Predictor (LangChain @tool com closure)."""
from __future__ import annotations

import json
import logging
from typing import Literal

from langchain_core.tools import tool

from src.api.config import VALID_HORIZONS
from src.api.schemas.forecast import ForecastRequest
from src.api.services.predictor import Predictor

logger = logging.getLogger(__name__)


def make_forecast_tool(predictor: Predictor):
    @tool
    async def run_weather_forecast(
        horizon: int,
        reference_date: str,
        model_type: Literal["tft", "nbeats"] = "tft",
    ) -> str:
        """Executa previsão horária de temperatura (WeatherOps) para o horizonte pedido.

        Args:
            horizon: Horas a prever (72, 168 ou 336).
            reference_date: Último instante do histórico no formato ISO ou YYYY-MM-DD.
            model_type: Família de modelo (tft ou nbeats).
        """
        if horizon not in VALID_HORIZONS:
            return json.dumps(
                {"error": "horizonte_invalido", "detail": f"Use um destes horizontes: {sorted(VALID_HORIZONS)}"},
                ensure_ascii=False,
            )
        try:
            req = ForecastRequest.model_validate(
                {"reference_date": reference_date, "model_type": model_type}
            )
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": "pedido_invalido", "detail": str(e)}, ensure_ascii=False)

        try:
            resp = await predictor.predict(horizon, req)
        except Exception as e:  # noqa: BLE001
            logger.warning("run_weather_forecast: %s", e)
            return json.dumps({"error": "inferencia", "detail": str(e)}, ensure_ascii=False)

        payload = {
            "reference_date": resp.reference_date.isoformat(),
            "horizon": resp.horizon,
            "model_type": resp.model_type,
            "model_version": resp.model_version,
            "mape": resp.mape,
            "latency_ms": resp.latency_ms,
            "predictions_sample": [p.model_dump(mode="json") for p in resp.predictions[:12]],
            "predictions_total": len(resp.predictions),
        }
        return json.dumps(payload, ensure_ascii=False)

    return run_weather_forecast
