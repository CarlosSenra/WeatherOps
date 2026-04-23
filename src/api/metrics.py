"""Configuração de métricas Prometheus para a API WeatherOps.

Utiliza ``prometheus-fastapi-instrumentator`` para instrumentar automaticamente
todas as rotas FastAPI e expor um endpoint ``/metrics`` no formato Prometheus.

Métricas expostas automaticamente
-----------------------------------
- ``http_requests_total``            — contador de requisições por rota/método/status
- ``http_request_duration_seconds``  — histograma de latência (inclui buckets p50/p95/p99)
- ``http_requests_in_progress``      — gauge de requisições em andamento

Uso
---
Chamar ``setup_metrics(app)`` dentro de ``create_app()`` após registrar os routers.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

logger = logging.getLogger(__name__)


def setup_metrics(app: FastAPI) -> None:
    """Instrumenta o app FastAPI e expõe o endpoint ``/metrics``."""
    Instrumentator(
        should_group_status_codes=False,
        excluded_handlers=["/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    logger.info("Métricas Prometheus disponíveis em /metrics")
