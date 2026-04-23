"""Configuração central da API de Serving do WeatherOps.

Define os modelos registrados para servir, as colunas padrão de features e
targets, e as configurações de ambiente lidas via variáveis de ambiente ou
arquivo ``.env``.

Para adicionar um novo modelo à API, basta incluir uma entrada em
``REGISTERED_MODELS`` — nenhuma alteração em routers, serviços ou engines
é necessária.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Colunas padrão de features e targets — devem ser idênticas às usadas no
# pipeline de treinamento para garantir consistência na inferência.
# ---------------------------------------------------------------------------
DEFAULT_FEATURES: list[str] = [
    "temp_ar_c",
    "umidade_rel_ar_percent",
    "pressao_atm_estacao_mb",
    "precipitacao_total_mm",
    "hora_sin",
    "hora_cos",
    "temp_lag_1h",
    "temp_lag_24h",
    "temp_ma_6h",
    "temp_ma_12h",
    "pressao_ma_6h",
    "pressao_ma_12h",
    "pressao_tendencia_1h",
    "temp_tendencia_1h",
]

DEFAULT_TARGETS: list[str] = ["temp_ar_c"]


# ---------------------------------------------------------------------------
# ServingModelConfig
# ---------------------------------------------------------------------------


@dataclass
class ServingModelConfig:
    """Descrição declarativa de uma versão de modelo servida pela API.

    Adicionar um novo modelo à API requer apenas incluir uma entrada aqui —
    sem alterações em routers, serviços ou engines.
    """

    key: str
    """Identificador curto usado como chave no ModelRegistry, ex.: ``'tft_72'``."""

    experiment_name: str
    """Nome lógico do experimento — usado como sub-chave de exportação local
    e como fallback para ``registry_model_name`` quando este não for definido."""

    model_type: str
    """Família de arquitetura: ``'tft'`` | ``'nbeats'`` | ``'lstm'`` | ``'transformer'``."""

    horizon: int
    """Horizonte de previsão em horas: 72 | 168 | 336."""

    sequence_length: int
    """Tamanho da janela de look-back em horas (encoder length para modelos PF)."""

    engine_class: str
    """Chave no ``ENGINE_REGISTRY``: ``'pytorch_forecasting'`` | ``'sequential'``."""

    registry_model_name: str | None = None
    """Nome real no MLflow Model Registry e na pasta de exportação local.
    Quando ``None``, usa ``experiment_name``.  Permite desacoplar o nome
    público do modelo (ex.: ``'weather_tft_h72'``) do nome registrado
    (ex.: ``'weather_forecasting_h72'``)."""

    feature_columns: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    target_columns: list[str] = field(default_factory=lambda: list(DEFAULT_TARGETS))
    train_ratio: float = 0.8
    val_ratio: float = 0.1


# ---------------------------------------------------------------------------
# REGISTERED_MODELS — ponto único para adicionar ou remover modelos da API.
# ---------------------------------------------------------------------------

REGISTERED_MODELS: list[ServingModelConfig] = [
    # ── TFT ─────────────────────────────────────────────────────────────────
    # registry_model_name aponta para o nome real no MLflow Model Registry
    # (weather_forecasting_h*), desacoplado do nome lógico do experimento.
    ServingModelConfig(
        key="tft_72",
        experiment_name="weather_tft_h72",
        model_type="tft",
        horizon=72,
        sequence_length=168,
        engine_class="pytorch_forecasting",
        registry_model_name="weather_forecasting_h72",
    ),
    ServingModelConfig(
        key="tft_168",
        experiment_name="weather_tft_h168",
        model_type="tft",
        horizon=168,
        sequence_length=336,
        engine_class="pytorch_forecasting",
        registry_model_name="weather_forecasting_h168",
    ),
    ServingModelConfig(
        key="tft_336",
        experiment_name="weather_tft_h336",
        model_type="tft",
        horizon=336,
        sequence_length=504,
        engine_class="pytorch_forecasting",
        registry_model_name="weather_forecasting_h336",
    ),
    # ── N-BEATS ──────────────────────────────────────────────────────────────
    # Desativado: nenhum modelo N-BEATS foi promovido para produção no Registry.
    # Reative após promover com: poetry run python -m src.ml_workstation.promotion.run_promote
    #   --experiment-name weather_forecasting_h72 --model-name weather_nbeats_h72 ...
    # ServingModelConfig(
    #     key="nbeats_72",
    #     experiment_name="weather_nbeats_h72",
    #     model_type="nbeats",
    #     horizon=72,
    #     sequence_length=168,
    #     engine_class="pytorch_forecasting",
    #     registry_model_name="weather_nbeats_h72",
    # ),
    # ServingModelConfig(
    #     key="nbeats_168",
    #     experiment_name="weather_nbeats_h168",
    #     model_type="nbeats",
    #     horizon=168,
    #     sequence_length=336,
    #     engine_class="pytorch_forecasting",
    #     registry_model_name="weather_nbeats_h168",
    # ),
    # ServingModelConfig(
    #     key="nbeats_336",
    #     experiment_name="weather_nbeats_h336",
    #     model_type="nbeats",
    #     horizon=336,
    #     sequence_length=504,
    #     engine_class="pytorch_forecasting",
    #     registry_model_name="weather_nbeats_h336",
    # ),
    # ── Modelos futuros (descomente e preencha para ativar) ──────────────────
    # ServingModelConfig(
    #     key="lstm_72",
    #     experiment_name="weather_lstm_h72",
    #     model_type="lstm",
    #     horizon=72,
    #     sequence_length=168,
    #     engine_class="sequential",
    # ),
]

VALID_HORIZONS: frozenset[int] = frozenset(cfg.horizon for cfg in REGISTERED_MODELS)
VALID_MODEL_TYPES: frozenset[str] = frozenset(cfg.model_type for cfg in REGISTERED_MODELS)

# União de todas as colunas necessárias por qualquer modelo registrado —
# usada pelo DataService para saber quais colunas carregar do Parquet.
ALL_SERVING_COLUMNS: list[str] = list(
    dict.fromkeys(
        col
        for cfg in REGISTERED_MODELS
        for col in (cfg.feature_columns + cfg.target_columns)
    )
)


# ---------------------------------------------------------------------------
# Settings — configurações lidas de variáveis de ambiente ou arquivo .env.
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # MLflow
    mlflow_tracking_uri: str | None = None
    # Diretório com modelos exportados (promote + export); ex.: /app/ml_models no Docker.
    # Se definido e existir pasta por experiment_name, a API carrega pesos daqui antes do Registry.
    weatherops_model_root: str | None = None

    # Dados
    parquet_path: str = "/app/data/spec"

    # Inferência
    device: str = "cpu"

    # Cache
    cache_backend: Literal["memory", "redis"] = "memory"
    cache_ttl_seconds: int = 3600
    redis_url: str = "redis://localhost:6379"

    # Metadados da API
    api_title: str = "WeatherOps ML Serving API"
    api_version: str = "1.0.0"
    api_description: str = (
        "API REST assíncrona para servir modelos de previsão meteorológica TFT e N-BEATS "
        "carregados do MLflow Model Registry."
    )

    @property
    def registered_models(self) -> list[ServingModelConfig]:
        return REGISTERED_MODELS


@lru_cache
def get_settings() -> Settings:
    """Retorna a instância singleton de ``Settings``.

    Utiliza ``lru_cache`` para garantir que as variáveis de ambiente sejam
    lidas apenas uma vez durante o ciclo de vida da aplicação.
    """
    return Settings()
