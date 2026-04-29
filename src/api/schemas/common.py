from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ForecastPoint(BaseModel):
    """Um único valor de previsão com timestamp associado."""

    timestamp: datetime
    temp_ar_c: float = Field(description="Temperatura do ar prevista em graus Celsius.")


class ErrorResponse(BaseModel):
    """Envelope de erro padronizado retornado em respostas 4xx/5xx."""

    detail: str
    error_type: str | None = None
