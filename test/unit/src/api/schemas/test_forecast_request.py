"""Testes do schema ForecastRequest (parse de reference_date)."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from src.api.schemas.forecast import ForecastRequest


@pytest.mark.unit
def test_reference_date_date_only_string_normalizes_to_midnight() -> None:
    body = ForecastRequest.model_validate(
        {
            "reference_date": "2024-06-01",
            "model_type": "tft",
            "group_id": "station_1",
        }
    )
    assert body.reference_date == datetime(2024, 6, 1, 0, 0, 0)


@pytest.mark.unit
def test_reference_date_iso_datetime_naive() -> None:
    body = ForecastRequest.model_validate(
        {
            "reference_date": "2024-06-01T12:00:00",
            "model_type": "tft",
        }
    )
    assert body.reference_date == datetime(2024, 6, 1, 12, 0, 0)


@pytest.mark.unit
def test_reference_date_z_suffix_utc() -> None:
    body = ForecastRequest.model_validate(
        {
            "reference_date": "2024-06-01T12:00:00Z",
            "model_type": "tft",
        }
    )
    assert body.reference_date == datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.mark.unit
def test_reference_date_pass_through_datetime() -> None:
    dt = datetime(2024, 6, 1, 9, 30, 0)
    body = ForecastRequest.model_validate(
        {
            "reference_date": dt,
            "model_type": "tft",
        }
    )
    assert body.reference_date is dt


@pytest.mark.unit
def test_reference_date_from_date_object() -> None:
    body = ForecastRequest.model_validate(
        {
            "reference_date": date(2024, 6, 1),
            "model_type": "tft",
        }
    )
    assert body.reference_date == datetime(2024, 6, 1, 0, 0, 0)
