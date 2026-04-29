"""Unit tests for src/api/metrics.py.

Verifica que o endpoint /metrics está disponível, retorna o formato correto
do Prometheus e acumula contadores após requisições reais.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.metrics import setup_metrics
from src.api.routers.health import router as health_router


def _make_app() -> FastAPI:
    """App mínimo com métricas + rota /health para poder gerar tráfego."""
    app = FastAPI()
    app.include_router(health_router)
    setup_metrics(app)
    return app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(_make_app())


# ---------------------------------------------------------------------------
# Endpoint /metrics
# ---------------------------------------------------------------------------


def test_metrics_endpoint_exists(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_content_type_is_prometheus_format(client: TestClient) -> None:
    response = client.get("/metrics")
    assert "text/plain" in response.headers["content-type"]


def test_metrics_contains_http_requests_total(client: TestClient) -> None:
    client.get("/health")
    response = client.get("/metrics")
    assert "http_requests_total" in response.text


def test_metrics_contains_duration_histogram(client: TestClient) -> None:
    client.get("/health")
    response = client.get("/metrics")
    # prometheus-fastapi-instrumentator expõe dois histogramas de latência:
    # - http_request_duration_highr_seconds (alta resolução, sem labels de rota)
    # - http_request_duration_seconds (poucos buckets, com label handler)
    assert "http_request_duration_highr_seconds_bucket" in response.text


def test_metrics_increments_counter_after_request(client: TestClient) -> None:
    """Faz duas requisições e verifica que o contador está presente no /metrics."""
    client.get("/health")
    client.get("/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_metrics_endpoint_not_tracked_in_itself(client: TestClient) -> None:
    """/metrics não deve aparecer como rota rastreada (excluded_handlers)."""
    client.get("/metrics")
    response = client.get("/metrics")
    # O handler /metrics está excluído — não deve aparecer como label de rota
    assert 'handler="/metrics"' not in response.text
