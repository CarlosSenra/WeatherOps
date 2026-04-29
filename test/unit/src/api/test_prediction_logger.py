"""Unit tests for src/api/services/prediction_logger.py.

Usa SQLite em memória (":memory:") para isolamento total — sem I/O de disco,
sem side effects entre testes.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.api.schemas.common import ForecastPoint
from src.api.schemas.forecast import ForecastResponse
from src.api.services.prediction_logger import BucketResult, PredictionLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    model_type: str = "tft",
    horizon: int = 72,
    reference_date: datetime | None = None,
    n_points: int = 3,
) -> ForecastResponse:
    ref = reference_date or datetime(2024, 1, 1, 0, 0, 0)
    predictions = [
        ForecastPoint(
            timestamp=ref + timedelta(hours=i + 1),
            temp_ar_c=20.0 + i,
        )
        for i in range(n_points)
    ]
    return ForecastResponse(
        reference_date=ref,
        horizon=horizon,
        model_type=model_type,
        model_version="1",
        mape=0.1,
        predictions=predictions,
        latency_ms=50.0,
    )


@pytest.fixture()
async def logger() -> PredictionLogger:
    lg = PredictionLogger(db_path=":memory:")
    await lg.init()
    return lg


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_init_creates_connection(logger: PredictionLogger) -> None:
    assert logger._conn is not None


@pytest.mark.unit
async def test_init_is_idempotent() -> None:
    lg = PredictionLogger(db_path=":memory:")
    await lg.init()
    await lg.init()
    assert lg._conn is not None


# ---------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_log_returns_uuid_string(logger: PredictionLogger) -> None:
    pred_id = await logger.log(_make_response(), group_id="station_1")
    assert isinstance(pred_id, str)
    assert len(pred_id) == 36  # UUID4 format


@pytest.mark.unit
async def test_log_persists_prediction(logger: PredictionLogger) -> None:
    await logger.log(_make_response(), group_id="station_1")
    rows = logger._conn.execute("SELECT * FROM prediction_log").fetchall()  # type: ignore[union-attr]
    assert len(rows) == 1


@pytest.mark.unit
async def test_log_stores_correct_model_key(logger: PredictionLogger) -> None:
    await logger.log(_make_response(model_type="tft", horizon=168), group_id="s1")
    row = logger._conn.execute("SELECT model_key FROM prediction_log").fetchone()  # type: ignore[union-attr]
    assert row["model_key"] == "tft_168"


@pytest.mark.unit
async def test_log_stores_forecast_json(logger: PredictionLogger) -> None:
    await logger.log(_make_response(n_points=3), group_id="s1")
    row = logger._conn.execute("SELECT forecast_json FROM prediction_log").fetchone()  # type: ignore[union-attr]
    points = json.loads(row["forecast_json"])
    assert len(points) == 3
    assert "temp_ar_c" in points[0]
    assert "timestamp" in points[0]


@pytest.mark.unit
async def test_log_multiple_predictions(logger: PredictionLogger) -> None:
    await logger.log(_make_response(), group_id="s1")
    await logger.log(_make_response(), group_id="s1")
    count = logger._conn.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0]  # type: ignore[union-attr]
    assert count == 2


# ---------------------------------------------------------------------------
# get_pending
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_get_pending_returns_expired_prediction(logger: PredictionLogger) -> None:
    ref = datetime(2024, 1, 1, 0, 0, 0)
    await logger.log(_make_response(horizon=72, reference_date=ref), group_id="s1")

    # agora = reference_date + 73h → prediction deve aparecer
    now = ref + timedelta(hours=73)
    pending = logger.get_pending(now=now)
    assert len(pending) == 1


@pytest.mark.unit
async def test_get_pending_excludes_not_yet_expired(logger: PredictionLogger) -> None:
    ref = datetime(2024, 1, 1, 0, 0, 0)
    await logger.log(_make_response(horizon=72, reference_date=ref), group_id="s1")

    # agora = reference_date + 71h → prediction NÃO deve aparecer
    now = ref + timedelta(hours=71)
    pending = logger.get_pending(now=now)
    assert len(pending) == 0


@pytest.mark.unit
async def test_get_pending_excludes_already_evaluated(logger: PredictionLogger) -> None:
    ref = datetime(2024, 1, 1, 0, 0, 0)
    pred_id = await logger.log(_make_response(horizon=72, reference_date=ref), group_id="s1")
    logger.save_accuracy(pred_id, [BucketResult("near", 1.0, 1.4, 0.05, 24)])

    now = ref + timedelta(hours=73)
    pending = logger.get_pending(now=now)
    assert len(pending) == 0


@pytest.mark.unit
async def test_get_pending_returns_correct_record_fields(logger: PredictionLogger) -> None:
    ref = datetime(2024, 6, 15, 12, 0, 0)
    await logger.log(
        _make_response(model_type="tft", horizon=72, reference_date=ref), group_id="station_1"
    )
    now = ref + timedelta(hours=73)
    pending = logger.get_pending(now=now)

    assert pending[0].model_key == "tft_72"
    assert pending[0].horizon == 72
    assert pending[0].reference_date == ref


# ---------------------------------------------------------------------------
# save_accuracy
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_save_accuracy_stores_all_buckets(logger: PredictionLogger) -> None:
    pred_id = await logger.log(_make_response(), group_id="s1")
    results = [
        BucketResult("near", 1.0, 1.2, 0.04, 24),
        BucketResult("mid", 2.0, 2.5, 0.09, 48),
        BucketResult("far", 3.5, 4.0, 0.15, 264),
    ]
    logger.save_accuracy(pred_id, results)

    rows = logger._conn.execute(  # type: ignore[union-attr]
        "SELECT bucket FROM accuracy_log WHERE prediction_id = ?", (pred_id,)
    ).fetchall()
    buckets = {r["bucket"] for r in rows}
    assert buckets == {"near", "mid", "far"}


@pytest.mark.unit
async def test_save_accuracy_is_idempotent(logger: PredictionLogger) -> None:
    pred_id = await logger.log(_make_response(), group_id="s1")
    result = [BucketResult("near", 1.0, 1.2, 0.04, 24)]
    logger.save_accuracy(pred_id, result)
    logger.save_accuracy(pred_id, result)

    count = logger._conn.execute(  # type: ignore[union-attr]
        "SELECT COUNT(*) FROM accuracy_log WHERE prediction_id = ?", (pred_id,)
    ).fetchone()[0]
    assert count == 1


@pytest.mark.unit
def test_save_accuracy_noop_when_results_empty() -> None:
    lg = PredictionLogger(db_path=":memory:")
    # Sem init — deve ser silencioso
    lg.save_accuracy("some-id", [])


@pytest.mark.unit
def test_get_pending_returns_empty_before_init() -> None:
    lg = PredictionLogger(db_path=":memory:")
    assert lg.get_pending(now=datetime.now()) == []
