"""HTTP tests for src/api/routers/forecast.py (POST /v1/forecast/{horizon})."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.api.cache import InMemoryResponseCache
from src.api.config import VALID_HORIZONS
from src.api.routers.forecast import router
from src.api.schemas.common import ForecastPoint
from src.api.schemas.forecast import ForecastResponse


def _sample_forecast_response(*, horizon: int) -> ForecastResponse:
    ref = datetime(2024, 6, 1, 12, 0, 0)
    predictions = [
        ForecastPoint(timestamp=ref + timedelta(hours=h + 1), temp_ar_c=20.0 + h * 0.01)
        for h in range(horizon)
    ]
    return ForecastResponse(
        reference_date=ref,
        horizon=horizon,
        model_type="tft",
        model_version="1",
        mape=None,
        predictions=predictions,
        latency_ms=12.34,
    )


def _make_app(predictor: MagicMock | AsyncMock) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.predictor = predictor
    return app


def _valid_body() -> dict:
    return {
        "reference_date": "2024-06-01T12:00:00",
        "model_type": "tft",
        "group_id": "station_1",
    }


@pytest.fixture()
def valid_horizon() -> int:
    return min(VALID_HORIZONS)


@pytest.fixture()
def fresh_cache() -> InMemoryResponseCache:
    return InMemoryResponseCache(ttl_seconds=3600)


def test_forecast_invalid_horizon_returns_422(valid_horizon: int) -> None:
    predictor = MagicMock()
    app = _make_app(predictor)
    invalid = max(VALID_HORIZONS) + 999
    assert invalid not in VALID_HORIZONS
    with patch("src.api.routers.forecast.get_cache", return_value=InMemoryResponseCache()):
        client = TestClient(app)
        r = client.post(f"/v1/forecast/{invalid}", json=_valid_body())
    assert r.status_code == 422
    assert "horizontes válidos" in r.json()["detail"].lower() or "não é suportado" in r.json()["detail"]


def test_forecast_get_method_returns_405(valid_horizon: int) -> None:
    predictor = MagicMock()
    app = _make_app(predictor)
    with patch("src.api.routers.forecast.get_cache", return_value=InMemoryResponseCache()):
        client = TestClient(app)
        r = client.get(f"/v1/forecast/{valid_horizon}")
    assert r.status_code == 405


def test_forecast_missing_reference_date_returns_422(valid_horizon: int) -> None:
    predictor = MagicMock()
    app = _make_app(predictor)
    with patch("src.api.routers.forecast.get_cache", return_value=InMemoryResponseCache()):
        client = TestClient(app)
        r = client.post(
            f"/v1/forecast/{valid_horizon}",
            json={"model_type": "tft", "group_id": "station_1"},
        )
    assert r.status_code == 422


def test_forecast_invalid_model_type_returns_422(valid_horizon: int) -> None:
    predictor = MagicMock()
    app = _make_app(predictor)
    body = {**_valid_body(), "model_type": "not_a_model"}
    with patch("src.api.routers.forecast.get_cache", return_value=InMemoryResponseCache()):
        client = TestClient(app)
        r = client.post(f"/v1/forecast/{valid_horizon}", json=body)
    assert r.status_code == 422


def test_forecast_malformed_json_returns_error(valid_horizon: int) -> None:
    predictor = MagicMock()
    app = _make_app(predictor)
    with patch("src.api.routers.forecast.get_cache", return_value=InMemoryResponseCache()):
        client = TestClient(app)
        r = client.post(
            f"/v1/forecast/{valid_horizon}",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
    assert r.status_code in (400, 422)


def test_forecast_success_returns_200(valid_horizon: int, fresh_cache: InMemoryResponseCache) -> None:
    resp = _sample_forecast_response(horizon=valid_horizon)
    predictor = MagicMock()
    predictor.predict = AsyncMock(return_value=resp)
    app = _make_app(predictor)
    with patch("src.api.routers.forecast.get_cache", return_value=fresh_cache):
        client = TestClient(app)
        r = client.post(f"/v1/forecast/{valid_horizon}", json=_valid_body())
    assert r.status_code == 200
    data = r.json()
    assert data["horizon"] == valid_horizon
    assert len(data["predictions"]) == valid_horizon
    assert data["model_type"] == "tft"
    predictor.predict.assert_awaited_once()


def test_forecast_predictor_404_propagates(valid_horizon: int) -> None:
    predictor = MagicMock()
    predictor.predict = AsyncMock(side_effect=HTTPException(status_code=404, detail="model missing"))
    app = _make_app(predictor)
    with patch("src.api.routers.forecast.get_cache", return_value=InMemoryResponseCache()):
        client = TestClient(app)
        r = client.post(f"/v1/forecast/{valid_horizon}", json=_valid_body())
    assert r.status_code == 404
    assert r.json()["detail"] == "model missing"


def test_forecast_predictor_422_propagates(valid_horizon: int) -> None:
    predictor = MagicMock()
    predictor.predict = AsyncMock(
        side_effect=HTTPException(status_code=422, detail="insufficient history")
    )
    app = _make_app(predictor)
    with patch("src.api.routers.forecast.get_cache", return_value=InMemoryResponseCache()):
        client = TestClient(app)
        r = client.post(f"/v1/forecast/{valid_horizon}", json=_valid_body())
    assert r.status_code == 422


def test_forecast_predictor_generic_error_returns_500(valid_horizon: int) -> None:
    predictor = MagicMock()
    predictor.predict = AsyncMock(side_effect=RuntimeError("engine failure"))
    app = _make_app(predictor)
    with patch("src.api.routers.forecast.get_cache", return_value=InMemoryResponseCache()):
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post(f"/v1/forecast/{valid_horizon}", json=_valid_body())
    assert r.status_code == 500


def test_forecast_cache_second_request_skips_predictor(valid_horizon: int, fresh_cache: InMemoryResponseCache) -> None:
    resp = _sample_forecast_response(horizon=valid_horizon)
    predictor = MagicMock()
    predictor.predict = AsyncMock(return_value=resp)
    app = _make_app(predictor)
    with patch("src.api.routers.forecast.get_cache", return_value=fresh_cache):
        client = TestClient(app)
        r1 = client.post(f"/v1/forecast/{valid_horizon}", json=_valid_body())
        r2 = client.post(f"/v1/forecast/{valid_horizon}", json=_valid_body())
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert predictor.predict.await_count == 1


def test_forecast_with_prediction_logger_does_not_break(valid_horizon: int, fresh_cache: InMemoryResponseCache) -> None:
    resp = _sample_forecast_response(horizon=valid_horizon)
    predictor = MagicMock()
    predictor.predict = AsyncMock(return_value=resp)
    app = _make_app(predictor)
    app.state.prediction_logger = MagicMock(log=AsyncMock())
    with patch("src.api.routers.forecast.get_cache", return_value=fresh_cache):
        client = TestClient(app)
        r = client.post(f"/v1/forecast/{valid_horizon}", json=_valid_body())
    assert r.status_code == 200
