"""Schemas Pydantic da API WeatherOps.

Exporta os modelos de dados utilizados nas requisições e respostas:

- ``ForecastPoint``:    Um único ponto de previsão com timestamp e valor.
- ``ErrorResponse``:   Envelope de erro padronizado para respostas 4xx/5xx.
- ``ForecastRequest``: Corpo da requisição do endpoint de previsão.
- ``ForecastResponse``: Corpo da resposta do endpoint de previsão.
"""
from src.api.schemas.common import ErrorResponse, ForecastPoint
from src.api.schemas.forecast import ForecastRequest, ForecastResponse

__all__ = [
    "ForecastPoint",
    "ErrorResponse",
    "ForecastRequest",
    "ForecastResponse",
]
