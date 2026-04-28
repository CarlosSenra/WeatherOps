"""Tool de previsão por período do dia — seleciona horizonte e filtra horários automaticamente."""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import Literal

from langchain_core.tools import tool

from src.api.schemas.forecast import ForecastRequest
from src.api.services.predictor import Predictor

logger = logging.getLogger(__name__)

_PERIOD_HOURS: dict[str, tuple[int, int]] = {
    "manha":     (7,  12),   # 07:00–11:59
    "tarde":     (12, 17),   # 12:00–16:59
    "noite":     (17, 24),   # 17:00–23:59
    "madrugada": (0,   7),   # 00:00–06:59
}

_PERIOD_LABELS = {
    "manha":     "Manhã (07h–12h)",
    "tarde":     "Tarde (12h–17h)",
    "noite":     "Noite (17h–00h)",
    "madrugada": "Madrugada (00h–07h)",
}


def make_period_forecast_tool(predictor: Predictor):
    @tool
    async def get_forecast_by_period(
        target_date: str,
        period: Literal["manha", "tarde", "noite", "madrugada"],
    ) -> str:
        """Retorna previsão de temperatura para um período específico do dia.

        Use esta tool quando o usuário perguntar sobre a temperatura em um período
        do dia (manhã, tarde, noite ou madrugada) sem precisar informar o horizonte —
        ele é calculado automaticamente.

        Args:
            target_date: Data alvo no formato YYYY-MM-DD (ex.: 2024-06-01).
            period: Período do dia — manha (07–12h), tarde (12–17h), noite (17–00h),
                    madrugada (00–07h).
        """
        try:
            target = date.fromisoformat(target_date)
        except ValueError as e:
            return json.dumps({"error": "data_invalida", "detail": str(e)}, ensure_ascii=False)

        # reference_date = início do dia alvo; 72h cobre o dia inteiro com folga
        reference_date_str = f"{target_date}T00:00:00"
        horizon = 72

        try:
            req = ForecastRequest.model_validate(
                {"reference_date": reference_date_str, "model_type": "tft"}
            )
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": "pedido_invalido", "detail": str(e)}, ensure_ascii=False)

        try:
            resp = await predictor.predict(horizon, req)
        except Exception as e:  # noqa: BLE001
            logger.warning("get_forecast_by_period: %s", e)
            return json.dumps({"error": "inferencia", "detail": str(e)}, ensure_ascii=False)

        h_start, h_end = _PERIOD_HOURS[period]
        window = [
            p for p in resp.predictions
            if p.timestamp.date() == target and h_start <= p.timestamp.hour < h_end
        ]

        if not window:
            return json.dumps(
                {"error": "sem_dados", "detail": f"Sem previsões para {_PERIOD_LABELS[period]} em {target_date}."},
                ensure_ascii=False,
            )

        temps = [p.temp_ar_c for p in window]
        peak = max(window, key=lambda p: p.temp_ar_c)

        result = {
            "target_date": target_date,
            "period": _PERIOD_LABELS[period],
            "temperature_min_c": round(min(temps), 1),
            "temperature_max_c": round(max(temps), 1),
            "temperature_avg_c": round(sum(temps) / len(temps), 1),
            "peak_hour": f"{peak.timestamp.hour:02d}h",
            "mape_model": resp.mape,
            "hourly": [
                {"hour": f"{p.timestamp.hour:02d}h", "temp_c": round(p.temp_ar_c, 1)}
                for p in window
            ],
        }
        return json.dumps(result, ensure_ascii=False)

    return get_forecast_by_period
