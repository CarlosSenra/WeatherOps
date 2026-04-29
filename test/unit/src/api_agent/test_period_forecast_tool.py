"""Testes da tool get_forecast_by_period."""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api_agent.tools.period_forecast_tool import make_period_forecast_tool


def _make_predictions(base_date: str, hours: list[int], temps: list[float]):
    """Cria lista de ForecastPoint mocks para os horários indicados."""
    points = []
    for h, t in zip(hours, temps):
        p = MagicMock()
        p.timestamp = datetime.fromisoformat(f"{base_date}T{h:02d}:00:00")
        p.temp_ar_c = t
    points.append(p)
    return points


def _make_predictor(predictions, mape: float = 0.04):
    resp = MagicMock()
    resp.predictions = predictions
    resp.model_type = "tft"
    resp.mape = mape
    predictor = MagicMock()
    predictor.predict = AsyncMock(return_value=resp)
    return predictor


@pytest.mark.asyncio
async def test_tarde_returns_filtered_predictions() -> None:
    hours = list(range(1, 25))
    temps = [20.0 + h * 0.5 for h in hours]
    predictions = []
    for h, t in zip(hours, temps):
        p = MagicMock()
        p.timestamp = datetime.fromisoformat(f"2024-06-01T{h % 24:02d}:00:00")
        p.temp_ar_c = t
        predictions.append(p)

    tool = make_period_forecast_tool(_make_predictor(predictions))
    result = json.loads(await tool.ainvoke({"target_date": "2024-06-01", "period": "tarde"}))

    assert result["period"] == "Tarde (12h–17h)"
    assert "temperature_min_c" in result
    assert "temperature_max_c" in result
    assert "hourly" in result
    assert all(12 <= int(h["hour"].replace("h", "")) < 17 for h in result["hourly"])


@pytest.mark.asyncio
async def test_manha_filters_correct_hours() -> None:
    predictions = []
    for h in range(1, 25):
        p = MagicMock()
        p.timestamp = datetime.fromisoformat(f"2024-06-01T{h % 24:02d}:00:00")
        p.temp_ar_c = 22.0 + h * 0.3
        predictions.append(p)

    tool = make_period_forecast_tool(_make_predictor(predictions))
    result = json.loads(await tool.ainvoke({"target_date": "2024-06-01", "period": "manha"}))

    assert result["period"] == "Manhã (07h–12h)"
    hours_returned = [int(h["hour"].replace("h", "")) for h in result["hourly"]]
    assert all(7 <= h < 12 for h in hours_returned)


@pytest.mark.asyncio
async def test_invalid_date_returns_error() -> None:
    tool = make_period_forecast_tool(MagicMock())
    result = json.loads(await tool.ainvoke({"target_date": "not-a-date", "period": "tarde"}))
    assert result["error"] == "data_invalida"


@pytest.mark.asyncio
async def test_predictor_error_returns_error_json() -> None:
    from fastapi import HTTPException

    predictor = MagicMock()
    predictor.predict = AsyncMock(side_effect=HTTPException(status_code=422, detail="dados insuficientes"))

    tool = make_period_forecast_tool(predictor)
    result = json.loads(await tool.ainvoke({"target_date": "2024-06-01", "period": "tarde"}))
    assert result["error"] == "inferencia"


@pytest.mark.asyncio
async def test_peak_hour_is_hottest() -> None:
    predictions = []
    temps = {12: 26.0, 13: 28.5, 14: 30.0, 15: 29.5, 16: 27.0}
    for h, t in temps.items():
        p = MagicMock()
        p.timestamp = datetime.fromisoformat(f"2024-06-01T{h:02d}:00:00")
        p.temp_ar_c = t
        predictions.append(p)
    # add unrelated hours
    for h in [1, 6, 20]:
        p = MagicMock()
        p.timestamp = datetime.fromisoformat(f"2024-06-01T{h:02d}:00:00")
        p.temp_ar_c = 22.0
        predictions.append(p)

    tool = make_period_forecast_tool(_make_predictor(predictions))
    result = json.loads(await tool.ainvoke({"target_date": "2024-06-01", "period": "tarde"}))

    assert result["peak_hour"] == "14h"
    assert result["temperature_max_c"] == 30.0
    assert result["temperature_min_c"] == 26.0
