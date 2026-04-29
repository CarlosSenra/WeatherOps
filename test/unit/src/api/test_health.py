"""Unit tests for src/api/routers/health.py."""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers.health import router


def _make_app(
    *,
    data_ready: bool = True,
    row_count: int = 5000,
    models_ready: bool = True,
    loaded_keys: list[str] | None = None,
    failed_keys: list[str] | None = None,
) -> FastAPI:
    if loaded_keys is None:
        loaded_keys = ["tft_72"]
    if failed_keys is None:
        failed_keys = []

    app = FastAPI()
    app.include_router(router)

    registry = MagicMock()
    registry.is_ready.return_value = models_ready
    registry.list_models.return_value = [{"key": k} for k in loaded_keys]
    registry.failed_models.return_value = failed_keys

    data_service = MagicMock()
    data_service.is_ready = data_ready
    data_service.row_count = row_count

    app.state.registry = registry
    app.state.data_service = data_service
    return app


# ---------------------------------------------------------------------------
# Liveness probe
# ---------------------------------------------------------------------------

def test_liveness_always_returns_200() -> None:
    client = TestClient(_make_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_liveness_200_even_when_data_not_ready() -> None:
    client = TestClient(_make_app(data_ready=False, row_count=0, models_ready=False, loaded_keys=[]))
    response = client.get("/health")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Readiness probe
# ---------------------------------------------------------------------------

def test_readiness_200_when_fully_ready() -> None:
    client = TestClient(_make_app())
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pronto"
    assert "tft_72" in body["models_loaded"]
    assert body["row_count"] == 5000


def test_readiness_503_when_data_not_loaded() -> None:
    client = TestClient(_make_app(data_ready=False, row_count=0, models_ready=False, loaded_keys=[]))
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "carregando"


def test_readiness_207_when_models_failed() -> None:
    client = TestClient(_make_app(
        data_ready=True,
        models_ready=False,
        loaded_keys=[],
        failed_keys=["tft_72"],
    ))
    response = client.get("/health/ready")
    assert response.status_code == 207
    body = response.json()
    assert body["status"] == "degradado"
    assert "tft_72" in body["models_failed"]
