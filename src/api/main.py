"""Fábrica da aplicação FastAPI para a API de Serving de ML do WeatherOps.

Sequência de inicialização (startup)
-------------------------------------
1. Carregar configurações do ambiente / arquivo ``.env``.
2. Iniciar o DataService: ler arquivos Parquet em memória.
3. Iniciar o ModelRegistry: carregar modelos em produção do MLflow Registry
   e preparar metadados do engine (``training_dataset`` / ``scaler``) para
   cada modelo.
4. Instanciar o Predictor e armazenar tudo em ``app.state``.

A inicialização é totalmente assíncrona. Operações de I/O bloqueantes
(leitura de Parquet, downloads do MLflow, construção do TimeSeriesDataSet)
são delegadas a thread pools dentro de cada serviço, para que o event loop
nunca seja bloqueado.

Sequência de encerramento (shutdown)
--------------------------------------
Nenhuma limpeza necessária — os modelos são liberados da memória e os dados
do Parquet são descartados quando o processo termina.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import ALL_SERVING_COLUMNS, get_settings
from src.api.metrics import setup_metrics
from src.api.routers import forecast, health
from src.api.services.accuracy_evaluator import AccuracyEvaluator
from src.api.services.data_service import DataService
from src.api.services.model_registry import ModelRegistry
from src.api.services.prediction_logger import PredictionLogger
from src.api.services.predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    # ── 1. Definir URI de rastreamento do MLflow ───────────────────────────
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        logger.info("MLflow tracking URI: %s", settings.mlflow_tracking_uri)

    # ── 2. DataService ─────────────────────────────────────────────────────
    logger.info("Carregando dataset Parquet de %s …", settings.parquet_path)
    data_service = DataService()
    await data_service.load(
        settings.parquet_path,
        ALL_SERVING_COLUMNS,
        model_root=settings.weatherops_model_root,
    )

    # ── 3. ModelRegistry ───────────────────────────────────────────────────
    logger.info("Carregando modelos em produção do MLflow Registry …")
    registry = ModelRegistry()
    await registry.load_all(
        configs=settings.registered_models,
        data_service=data_service,
        tracking_uri=settings.mlflow_tracking_uri,
        device=settings.device,
        local_model_root=settings.weatherops_model_root,
    )

    # ── 4. PredictionLogger + AccuracyEvaluator ────────────────────────────
    logger.info("Inicializando PredictionLogger em %s …", settings.accuracy_db_path)
    prediction_logger = PredictionLogger(settings.accuracy_db_path)
    await prediction_logger.init()

    evaluator = AccuracyEvaluator(
        logger_svc=prediction_logger,
        data_service=data_service,
        interval_seconds=settings.accuracy_eval_interval_seconds,
    )
    evaluator_task = asyncio.create_task(evaluator.start())

    # ── 5. Configurar app.state ────────────────────────────────────────────
    app.state.data_service = data_service
    app.state.registry = registry
    app.state.predictor = Predictor(registry, data_service)
    app.state.prediction_logger = prediction_logger

    logger.info("Inicialização da API WeatherOps concluída.")

    yield

    evaluator_task.cancel()
    logger.info("Encerrando a API WeatherOps.")


# ---------------------------------------------------------------------------
# Fábrica da aplicação
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Cria e configura a instância da aplicação FastAPI."""
    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middlewares ────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Roteadores ─────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(forecast.router)
    try:
        from src.api_agent.routers.agent import router as agent_router

        app.include_router(agent_router)
    except ImportError as exc:
        logger.warning(
            "Router /v1/agent omitido — falha ao importar módulo do agente (%s). "
            "Garanta as dependências no ambiente da API (ex.: poetry install --only api,agent).",
            exc,
        )

    # ── Métricas Prometheus ────────────────────────────────────────────────
    setup_metrics(app)

    return app


# ---------------------------------------------------------------------------
# Instância ``app`` no nível do módulo, consumida pelo uvicorn.
# ---------------------------------------------------------------------------

app = create_app()
