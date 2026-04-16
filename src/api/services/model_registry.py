"""ModelRegistry — carrega e mantém todas as instâncias de modelos em produção.

Cada modelo é carregado exatamente uma vez durante o startup da aplicação via
o evento de lifespan. O registry mapeia strings ``model_key`` (ex.: ``"tft_72"``)
para objetos ``ModelEntry`` que agrupam o modelo PyTorch, seu engine de inferência,
metadados específicos do engine (``training_dataset`` ou ``scaler``) e informações
de governança.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import mlflow
import torch

from src.api.config import ServingModelConfig
from src.api.engines import ENGINE_REGISTRY, BaseInferenceEngine
from src.api.services.data_service import DataService
from src.ml_workstation.promotion.loader import (
    ModelNotInProductionError,
    get_production_info,
    load_production_model,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """Estado de runtime completo de uma versão de modelo servida."""

    model: torch.nn.Module
    engine: BaseInferenceEngine
    config: ServingModelConfig
    metadata: dict[str, Any]
    """Dados específicos do engine (ex.: ``{"training_dataset": …}`` ou ``{"scaler": …}``)."""
    version: str
    mape: float | None


class ModelRegistry:
    """Dicionário de ``ModelEntry`` com leitura segura após inicialização.

    Escrito uma única vez no startup (``load_all``), depois somente leitura
    pelo restante do ciclo de vida da aplicação. Nenhum bloqueio é necessário
    para leituras.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ModelEntry] = {}
        self._ready: bool = False
        self._failed: list[str] = []

    # ------------------------------------------------------------------
    # Inicialização
    # ------------------------------------------------------------------

    async def load_all(
        self,
        configs: list[ServingModelConfig],
        data_service: DataService,
        tracking_uri: str | None = None,
        device: str = "cpu",
        local_model_root: str | None = None,
    ) -> None:
        """Carrega todos os modelos registrados de forma concorrente.

        Uma falha ao carregar um modelo é registrada como erro, mas não
        interrompe o startup dos demais. A API pode continuar servindo
        modelos disponíveis mesmo que alguns estejam ausentes do Registry.

        Argumentos:
            configs:      Lista de ``ServingModelConfig`` definida em ``config.py``.
            data_service: ``DataService`` completamente carregado (deve estar pronto).
            tracking_uri: Override da URI de rastreamento do MLflow (None → auto-detectar).
            device:       Dispositivo de inferência (``'cpu'`` ou ``'cuda'``).
            local_model_root: Diretório base de exportações (``WEATHEROPS_MODEL_ROOT``).
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        tasks = [
            self._load_one(cfg, data_service, tracking_uri, device, local_model_root)
            for cfg in configs
        ]
        await asyncio.gather(*tasks, return_exceptions=False)
        self._ready = True

        loaded = list(self._entries.keys())
        logger.info(
            "ModelRegistry pronto — carregados: %s | falhas: %s",
            loaded,
            self._failed,
        )

    async def _load_one(
        self,
        cfg: ServingModelConfig,
        data_service: DataService,
        tracking_uri: str | None,
        device: str,
        local_model_root: str | None,
    ) -> None:
        """Carrega um único modelo e prepara seus metadados de engine."""
        registry_name = cfg.registry_model_name or cfg.experiment_name
        try:
            model = await asyncio.to_thread(
                load_production_model,
                cfg.experiment_name,
                tracking_uri,
                device,
                registry_name,
                local_model_root=local_model_root,
            )
            info = await asyncio.to_thread(
                get_production_info,
                cfg.experiment_name,
                tracking_uri,
                registry_name,
                local_model_root=local_model_root,
            )

            engine_cls = ENGINE_REGISTRY.get(cfg.engine_class)
            if engine_cls is None:
                raise ValueError(
                    f"engine_class desconhecido '{cfg.engine_class}' para o modelo '{cfg.key}'. "
                    f"Disponíveis: {list(ENGINE_REGISTRY)}"
                )
            engine = engine_cls()
            metadata = await engine.prepare_metadata(cfg, data_service)

            self._entries[cfg.key] = ModelEntry(
                model=model.eval(),
                engine=engine,
                config=cfg,
                metadata=metadata,
                version=str(info.get("version", "desconhecida")),
                mape=info.get("mape"),
            )
            logger.info(
                "Modelo '%s' carregado (versão=%s, mape=%s)",
                cfg.key,
                info.get("version"),
                info.get("mape"),
            )
        except ModelNotInProductionError as exc:
            logger.error("Modelo '%s' não está em produção: %s", cfg.key, exc)
            self._failed.append(cfg.key)
        except Exception as exc:
            logger.error("Falha ao carregar modelo '%s': %s", cfg.key, exc, exc_info=True)
            self._failed.append(cfg.key)

    # ------------------------------------------------------------------
    # API de consulta
    # ------------------------------------------------------------------

    def get(self, key: str) -> ModelEntry:
        """Recupera um ``ModelEntry`` pela chave.

        Argumentos:
            key: Ex.: ``"tft_72"``.

        Levanta:
            KeyError: Se o modelo não foi carregado (não está em produção
                      ou falhou durante o carregamento).
        """
        entry = self._entries.get(key)
        if entry is None:
            available = list(self._entries)
            raise KeyError(
                f"Modelo '{key}' não está disponível. "
                f"Modelos disponíveis: {available}. "
                f"Falhas no carregamento: {self._failed}."
            )
        return entry

    def is_ready(self) -> bool:
        """Retorna ``True`` se o processo de carregamento inicial foi concluído."""
        return self._ready

    def list_models(self) -> list[dict]:
        """Retorna metadados de todos os modelos carregados (usado por /health/ready)."""
        return [
            {
                "key": entry.config.key,
                "model_type": entry.config.model_type,
                "horizon": entry.config.horizon,
                "version": entry.version,
                "mape": entry.mape,
                "engine": entry.config.engine_class,
            }
            for entry in self._entries.values()
        ]

    def failed_models(self) -> list[str]:
        """Retorna a lista de chaves de modelos que falharam ao carregar."""
        return list(self._failed)
