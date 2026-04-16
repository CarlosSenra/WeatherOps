"""Roteadores da API WeatherOps.

Exporta os roteadores disponíveis:

- ``forecast``: POST /v1/forecast/{horizon} — executa previsões meteorológicas.
- ``health``:   GET /health e GET /health/ready — probes de liveness e readiness.
"""
from src.api.routers import forecast, health

__all__ = ["forecast", "health"]
