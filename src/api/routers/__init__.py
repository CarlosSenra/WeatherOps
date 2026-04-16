"""Roteadores da API WeatherOps.

Exporta os roteadores disponíveis:

- ``forecast``: POST /v1/forecast/{horizon} — executa previsões meteorológicas.
- ``health``:   GET /health e GET /health/ready — probes de liveness e readiness.
"""

__all__ = ["forecast", "health"]


def __getattr__(name: str):
    if name in {"forecast", "health"}:
        import importlib  # noqa: PLC0415

        return importlib.import_module(f"src.api.routers.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
