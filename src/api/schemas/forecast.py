from __future__ import annotations

from datetime import date, datetime, time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.api.schemas.common import ForecastPoint

# Estenda este Literal ao registrar novos tipos de modelo em config.py.
ModelType = Literal["tft", "nbeats"]

_REFERENCE_DATE_FIELD = (
    "Último instante do histórico meteorológico a considerar (inclusivo). "
    "A primeira hora prevista na resposta é imediatamente depois: reference_date + 1 hora. "
    "Internamente a API monta a janela de contexto (look-back) com os dados disponíveis até este momento.\n\n"
    "Formatos aceites no JSON: só a data (`YYYY-MM-DD`, interpretada como 00:00 desse dia) ou data e hora em ISO 8601 "
    '(ex.: `2024-06-01T14:30:00`; `Z` no fim indica UTC).'
)


def _parse_reference_instant(value: Any) -> datetime:
    """Normaliza entrada para datetime (data só → meia-noite local/naive)."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if isinstance(value, str):
        s = value.strip()
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            d = date.fromisoformat(s)
            return datetime.combine(d, time.min)
        iso = s[:-1] + "+00:00" if s.endswith("Z") else s
        return datetime.fromisoformat(iso)
    raise ValueError(
        "reference_date deve ser datetime, data ou string ISO (YYYY-MM-DD ou data e hora)."
    )


class ForecastRequest(BaseModel):
    """Corpo da requisição para o endpoint de previsão."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "reference_date": "2024-06-01",
                    "model_type": "tft",
                    "group_id": "station_1",
                },
                {
                    "reference_date": "2024-06-01T14:30:00",
                    "model_type": "tft",
                    "group_id": "station_1",
                },
            ]
        }
    )

    reference_date: Annotated[
        datetime,
        Field(
            description=_REFERENCE_DATE_FIELD,
        ),
    ]
    model_type: ModelType = Field(
        default="tft",
        description="Família de modelos a ser usada na inferência.",
    )
    group_id: str = Field(
        default="station_1",
        description="Identificador da estação meteorológica (chave de grupo na série temporal).",
    )

    @field_validator("reference_date", mode="before")
    @classmethod
    def parse_reference_date(cls, value: object) -> datetime:
        return _parse_reference_instant(value)


class ForecastResponse(BaseModel):
    """Corpo da resposta retornado pelo endpoint de previsão."""

    reference_date: datetime = Field(
        description="Instante de referência usado na previsão (eco do pedido, em formato completo data/hora).",
    )
    horizon: int = Field(description="Número de horas previstas.")
    model_type: str
    model_version: str = Field(description="Número de versão no MLflow Registry.")
    mape: float | None = Field(
        default=None,
        description="MAPE do modelo em produção (extraído das tags de promoção).",
    )
    predictions: list[ForecastPoint] = Field(
        description="Lista ordenada de pares (timestamp, valor) iniciando em reference_date + 1h."
    )
    latency_ms: float = Field(description="Latência de inferência ponta a ponta em milissegundos.")
