"""Configuração de métricas Prometheus para a API WeatherOps.

Utiliza ``prometheus-fastapi-instrumentator`` para instrumentar automaticamente
todas as rotas FastAPI e expor um endpoint ``/metrics`` no formato Prometheus.

Métricas expostas automaticamente
-----------------------------------
- ``http_requests_total``                       — contador de requisições por rota/método/status
- ``http_request_duration_seconds``             — histograma de latência HTTP (p50/p95/p99)
- ``http_requests_in_progress``                 — gauge de requisições em andamento

Métricas customizadas de ML
-----------------------------
- ``weatherops_inference_duration_seconds``     — histograma de latência de inferência por modelo
- ``weatherops_cache_hits_total``               — contador de respostas servidas do cache
- ``weatherops_cache_misses_total``             — contador de requisições que exigiram inferência

Uso
---
Chamar ``setup_metrics(app)`` dentro de ``create_app()`` após registrar os routers.
Importar ``inference_duration``, ``cache_hits`` e ``cache_misses`` nos módulos que
precisam registrar eventos (ex.: ``routers/forecast.py``).
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Métricas customizadas de ML — instâncias globais, compartilhadas por módulo
# ---------------------------------------------------------------------------

inference_duration = Histogram(
    "weatherops_inference_duration_seconds",
    "Tempo de inferência do modelo ML (prepare_input + run_inference), por model_key",
    labelnames=["model_key"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

cache_hits = Counter(
    "weatherops_cache_hits_total",
    "Número de respostas de previsão servidas diretamente do cache",
)

cache_misses = Counter(
    "weatherops_cache_misses_total",
    "Número de requisições de previsão que exigiram inferência (cache miss)",
)

# ---------------------------------------------------------------------------
# Métricas de qualidade do modelo — atualizadas pelo AccuracyEvaluator
# ---------------------------------------------------------------------------
# Labels:
#   model_key — ex.: "tft_72", "tft_168", "tft_336"
#   bucket    — "near" (h1-24) | "mid" (h25-72) | "far" (h73+)

model_mae = Gauge(
    "weatherops_model_mae",
    "Erro Absoluto Médio (MAE) por modelo e bucket de horizonte",
    labelnames=["model_key", "bucket"],
)

model_rmse = Gauge(
    "weatherops_model_rmse",
    "Raiz do Erro Quadrático Médio (RMSE) por modelo e bucket de horizonte",
    labelnames=["model_key", "bucket"],
)

model_mape = Gauge(
    "weatherops_model_mape",
    "Erro Percentual Absoluto Médio (MAPE) por modelo e bucket de horizonte",
    labelnames=["model_key", "bucket"],
)


def setup_metrics(app: FastAPI) -> None:
    """Instrumenta o app FastAPI e expõe o endpoint ``/metrics``."""
    Instrumentator(
        should_group_status_codes=False,
        excluded_handlers=["/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    logger.info("Métricas Prometheus disponíveis em /metrics")
