"""Unit tests for src/api/services/accuracy_evaluator.py.

Estrutura
---------
1. Testes das funções puras ``compute_metrics`` e ``slice_forecast`` —
   sem async, sem mocks, determinísticos.
2. Testes de ``AccuracyEvaluator.run_once`` com mocks de ``PredictionLogger``
   e ``DataService`` para verificar o fluxo de integração.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.api.services.accuracy_evaluator import (
    AccuracyEvaluator,
    MetricsResult,
    compute_metrics,
    slice_forecast,
)
from src.api.services.prediction_logger import BucketResult, PredictionRecord


# ---------------------------------------------------------------------------
# compute_metrics — funções puras
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_metrics_perfect_prediction() -> None:
    predicted = [20.0, 21.0, 22.0]
    actual = pd.Series([20.0, 21.0, 22.0])
    result = compute_metrics(predicted, actual)
    assert result.mae == pytest.approx(0.0)
    assert result.rmse == pytest.approx(0.0)
    assert result.mape == pytest.approx(0.0)
    assert result.n_points == 3


@pytest.mark.unit
def test_compute_metrics_known_mae() -> None:
    # Erros absolutos: |10-8|=2, |10-12|=2 → MAE = 2.0
    predicted = [10.0, 10.0]
    actual = pd.Series([8.0, 12.0])
    result = compute_metrics(predicted, actual)
    assert result.mae == pytest.approx(2.0)


@pytest.mark.unit
def test_compute_metrics_known_rmse() -> None:
    # Erros ao quadrado: 4, 4 → RMSE = sqrt(4) = 2.0
    predicted = [10.0, 10.0]
    actual = pd.Series([8.0, 12.0])
    result = compute_metrics(predicted, actual)
    assert result.rmse == pytest.approx(2.0)


@pytest.mark.unit
def test_compute_metrics_known_mape() -> None:
    # |10-8|/|8| = 0.25, |10-12|/|12| = 0.1667 → MAPE ≈ 0.2083
    predicted = [10.0, 10.0]
    actual = pd.Series([8.0, 12.0])
    result = compute_metrics(predicted, actual)
    expected_mape = (0.25 + (2.0 / 12.0)) / 2
    assert result.mape == pytest.approx(expected_mape, rel=1e-4)


@pytest.mark.unit
def test_compute_metrics_ignores_zero_actual_in_mape() -> None:
    # Quando actual=0, o ponto é ignorado no MAPE
    predicted = [5.0, 10.0]
    actual = pd.Series([0.0, 10.0])
    result = compute_metrics(predicted, actual)
    assert result.mape == pytest.approx(0.0)  # só o segundo ponto conta


@pytest.mark.unit
def test_compute_metrics_all_zero_actuals_returns_nan_mape() -> None:
    predicted = [5.0, 5.0]
    actual = pd.Series([0.0, 0.0])
    result = compute_metrics(predicted, actual)
    assert math.isnan(result.mape)


@pytest.mark.unit
def test_compute_metrics_empty_returns_nan() -> None:
    result = compute_metrics([], pd.Series([], dtype=float))
    assert result.n_points == 0
    assert math.isnan(result.mae)
    assert math.isnan(result.rmse)
    assert math.isnan(result.mape)


@pytest.mark.unit
def test_compute_metrics_uses_min_length() -> None:
    # predicted tem 5 pontos, actual tem 3 → deve usar 3
    predicted = [1.0, 2.0, 3.0, 4.0, 5.0]
    actual = pd.Series([1.0, 2.0, 3.0])
    result = compute_metrics(predicted, actual)
    assert result.n_points == 3
    assert result.mae == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# slice_forecast — funções puras
# ---------------------------------------------------------------------------


def _make_forecast_json(n: int = 72) -> str:
    points = [{"timestamp": f"2024-01-01T{i:02d}:00:00", "temp_ar_c": float(i)} for i in range(n)]
    return json.dumps(points)


@pytest.mark.unit
def test_slice_forecast_near_bucket() -> None:
    fc = _make_forecast_json(72)
    result = slice_forecast(fc, start_offset_h=1, end_offset_h=24)
    assert len(result) == 24
    # hora 1 → índice 0 → temp_ar_c = 0.0
    assert result[0] == pytest.approx(0.0)
    # hora 24 → índice 23 → temp_ar_c = 23.0
    assert result[-1] == pytest.approx(23.0)


@pytest.mark.unit
def test_slice_forecast_mid_bucket() -> None:
    fc = _make_forecast_json(72)
    result = slice_forecast(fc, start_offset_h=25, end_offset_h=72)
    assert len(result) == 48
    assert result[0] == pytest.approx(24.0)  # hora 25 → índice 24


@pytest.mark.unit
def test_slice_forecast_single_hour() -> None:
    fc = _make_forecast_json(10)
    result = slice_forecast(fc, start_offset_h=5, end_offset_h=5)
    assert len(result) == 1
    assert result[0] == pytest.approx(4.0)  # índice 4


@pytest.mark.unit
def test_slice_forecast_beyond_available_hours() -> None:
    fc = _make_forecast_json(10)
    result = slice_forecast(fc, start_offset_h=1, end_offset_h=100)
    assert len(result) == 10  # limitado pelo tamanho real


# ---------------------------------------------------------------------------
# AccuracyEvaluator.run_once — testes com mocks
# ---------------------------------------------------------------------------


def _make_prediction_record(
    horizon: int = 72,
    reference_date: datetime | None = None,
    n_points: int = 72,
) -> PredictionRecord:
    ref = reference_date or datetime(2024, 1, 1, 0, 0, 0)
    forecast = [
        {"timestamp": (ref + timedelta(hours=i + 1)).isoformat(), "temp_ar_c": 20.0}
        for i in range(n_points)
    ]
    return PredictionRecord(
        id="test-uuid-1234",
        model_key=f"tft_{horizon}",
        model_version="1",
        group_id="station_1",
        reference_date=ref,
        horizon=horizon,
        forecast_json=json.dumps(forecast),
    )


@pytest.fixture()
def mock_logger_svc() -> MagicMock:
    svc = MagicMock()
    svc.get_pending.return_value = []
    svc.save_accuracy = MagicMock()
    return svc


@pytest.fixture()
def mock_data_service() -> MagicMock:
    svc = MagicMock()
    svc.get_actual_values.return_value = None
    return svc


@pytest.mark.unit
async def test_run_once_does_nothing_when_no_pending(
    mock_logger_svc: MagicMock, mock_data_service: MagicMock
) -> None:
    mock_logger_svc.get_pending.return_value = []
    evaluator = AccuracyEvaluator(mock_logger_svc, mock_data_service)
    await evaluator.run_once()
    mock_data_service.get_actual_values.assert_not_called()
    mock_logger_svc.save_accuracy.assert_not_called()


@pytest.mark.unit
async def test_run_once_skips_when_no_actual_data(
    mock_logger_svc: MagicMock, mock_data_service: MagicMock
) -> None:
    mock_logger_svc.get_pending.return_value = [_make_prediction_record(horizon=72)]
    mock_data_service.get_actual_values.return_value = None

    evaluator = AccuracyEvaluator(mock_logger_svc, mock_data_service)
    await evaluator.run_once()
    mock_logger_svc.save_accuracy.assert_not_called()


@pytest.mark.unit
async def test_run_once_saves_metrics_when_actuals_available(
    mock_logger_svc: MagicMock, mock_data_service: MagicMock
) -> None:
    record = _make_prediction_record(horizon=72, n_points=72)
    mock_logger_svc.get_pending.return_value = [record]
    mock_data_service.get_actual_values.return_value = pd.Series([20.0] * 24)

    evaluator = AccuracyEvaluator(mock_logger_svc, mock_data_service)

    with patch("src.api.metrics.model_mae") as m_mae, \
         patch("src.api.metrics.model_rmse") as m_rmse, \
         patch("src.api.metrics.model_mape") as m_mape:

        for gauge in (m_mae, m_rmse, m_mape):
            gauge.labels.return_value = MagicMock()

        await evaluator.run_once()

    mock_logger_svc.save_accuracy.assert_called_once()
    saved_id, saved_results = mock_logger_svc.save_accuracy.call_args[0]
    assert saved_id == "test-uuid-1234"
    assert len(saved_results) > 0
    assert all(isinstance(r, BucketResult) for r in saved_results)


@pytest.mark.unit
async def test_run_once_only_evaluates_buckets_within_horizon(
    mock_logger_svc: MagicMock, mock_data_service: MagicMock
) -> None:
    """Para horizon=72, o bucket 'far' (h73+) não deve ser avaliado."""
    record = _make_prediction_record(horizon=72, n_points=72)
    mock_logger_svc.get_pending.return_value = [record]
    mock_data_service.get_actual_values.return_value = pd.Series([20.0] * 24)

    evaluator = AccuracyEvaluator(mock_logger_svc, mock_data_service)

    with patch("src.api.metrics.model_mae") as m_mae, \
         patch("src.api.metrics.model_rmse") as m_rmse, \
         patch("src.api.metrics.model_mape") as m_mape:
        for gauge in (m_mae, m_rmse, m_mape):
            gauge.labels.return_value = MagicMock()

        await evaluator.run_once()

    _, saved_results = mock_logger_svc.save_accuracy.call_args[0]
    bucket_names = {r.bucket for r in saved_results}
    assert "far" not in bucket_names


@pytest.mark.unit
async def test_run_once_continues_on_error(
    mock_logger_svc: MagicMock, mock_data_service: MagicMock
) -> None:
    """O worker não deve propagar exceções — deve logar e continuar."""
    mock_logger_svc.get_pending.side_effect = RuntimeError("db error")
    evaluator = AccuracyEvaluator(mock_logger_svc, mock_data_service)
    # start() captura erros; run_once() pode propagar — verificamos que start não quebra
    import asyncio

    task = asyncio.create_task(evaluator.start())
    await asyncio.sleep(0)  # deixa o event loop processar
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
