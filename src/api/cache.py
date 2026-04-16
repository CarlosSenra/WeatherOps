"""Cache de resultados de previsão.

Fornece um cache em memória com TTL (Time-To-Live) e uma interface limpa que
pode ser substituída por uma implementação Redis sem alterar o código dos
roteadores.

Uso nos roteadores
------------------
::

    from src.api.cache import get_cache

    cache = get_cache()          # singleton
    hit = cache.get(key)
    if hit is None:
        result = await predictor.predict(...)
        cache.set(key, result)

Construção da chave de cache
-----------------------------
``ResponseCache.make_key(*parts)`` aplica SHA-256 sobre todas as partes,
garantindo que chaves longas (ex.: timestamps ISO) nunca excedam limites de
cabeçalhos HTTP e não possam ser revertidas.

Migrando para Redis
-------------------
Defina ``CACHE_BACKEND=redis`` e ``REDIS_URL=redis://host:6379`` no ambiente.
A classe ``RedisResponseCache`` abaixo requer o pacote ``redis``
(``pip install redis``). O código dos roteadores permanece idêntico em ambos os casos.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocolo de cache — qualquer backend deve satisfazer esta interface.
# ---------------------------------------------------------------------------


class CacheBackend(Protocol):
    def get(self, key: str) -> Any | None: ...

    def set(self, key: str, value: Any) -> None: ...

    def make_key(self, *parts: str) -> str: ...


# ---------------------------------------------------------------------------
# Backend em memória (padrão)
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class InMemoryResponseCache:
    """Cache TTL em dicionário em memória.

    Não é distribuído — suficiente para deploys de processo único. Substitua
    por ``RedisResponseCache`` para compartilhar estado entre múltiplos
    workers ou pods.
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._store: dict[str, _CacheEntry] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Any | None:
        """Retorna o valor em cache ou ``None`` se expirado/ausente."""
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            del self._store[key]
            logger.debug("Cache expirado: %s", key)
            return None
        logger.debug("Cache hit: %s", key)
        return entry.value

    def set(self, key: str, value: Any) -> None:
        """Armazena um valor com TTL configurado na inicialização."""
        self._store[key] = _CacheEntry(
            value=value,
            expires_at=time.monotonic() + self._ttl,
        )
        logger.debug("Cache gravado: %s (ttl=%ds)", key, self._ttl)

    def make_key(self, *parts: str) -> str:
        """Gera uma chave SHA-256 a partir das partes fornecidas."""
        raw = ":".join(str(p) for p in parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    def clear(self) -> None:
        """Remove todas as entradas (útil em testes)."""
        self._store.clear()

    @property
    def size(self) -> int:
        """Número de entradas atualmente no cache."""
        return len(self._store)


# ---------------------------------------------------------------------------
# Backend Redis (opcional — requer o pacote ``redis``)
# ---------------------------------------------------------------------------


class RedisResponseCache:
    """Cache com backend Redis.

    Requer a instalação do pacote ``redis``. Os valores são serializados para
    JSON via ``orjson`` (ou ``json`` da stdlib como fallback). Os chamadores
    devem garantir que os valores sejam serializáveis em JSON (modelos Pydantic
    satisfazem isso via ``.model_dump()``).
    """

    def __init__(self, redis_url: str, ttl_seconds: int = 3600) -> None:
        try:
            import redis  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Instale o pacote 'redis' para usar RedisResponseCache: "
                "pip install redis"
            ) from exc

        self._client = redis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl_seconds

    def get(self, key: str) -> Any | None:
        """Recupera um valor do Redis. Retorna ``None`` em caso de falha ou ausência."""
        try:
            raw = self._client.get(key)
        except Exception as exc:
            logger.warning("Falha no Redis GET (key=%s): %s", key, exc)
            return None

        if raw is None:
            return None

        try:
            import json

            return json.loads(raw)
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        """Armazena um valor no Redis com o TTL configurado."""
        try:
            import json

            self._client.setex(key, self._ttl, json.dumps(value))
        except Exception as exc:
            logger.warning("Falha no Redis SET (key=%s): %s", key, exc)

    def make_key(self, *parts: str) -> str:
        """Gera uma chave SHA-256 a partir das partes fornecidas."""
        raw = ":".join(str(p) for p in parts)
        return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Fábrica — retorna o backend adequado com base nas configurações.
# ---------------------------------------------------------------------------


@lru_cache
def get_cache() -> InMemoryResponseCache | RedisResponseCache:
    """Retorna o backend de cache singleton.

    Lê ``CACHE_BACKEND`` e ``CACHE_TTL_SECONDS`` de ``Settings``. Inicializado
    de forma lazy na primeira requisição, permitindo que testes sobrescrevam as
    configurações antes da criação do cache.
    """
    from src.api.config import get_settings

    settings = get_settings()
    if settings.cache_backend == "redis":
        logger.info("Backend de cache: Redis (%s)", settings.redis_url)
        return RedisResponseCache(
            redis_url=settings.redis_url,
            ttl_seconds=settings.cache_ttl_seconds,
        )

    logger.info(
        "Backend de cache: em memória (ttl=%ds)", settings.cache_ttl_seconds
    )
    return InMemoryResponseCache(ttl_seconds=settings.cache_ttl_seconds)
