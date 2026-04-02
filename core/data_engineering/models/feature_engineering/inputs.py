from pydantic import BaseModel, Field
from typing import List, Optional


class FeatureEngineeringConfig(BaseModel):
    """
    Configuração para feature engineering meteorológico.
    
    Attributes:
        target_columns: Colunas principais para filtrar o ruído
        lag_hours: Lista com os lags (horas) para aplicar
        moving_avg_windows: Lista com os tamanhos de janelas para média móvel
        use_seasonal_features: Se deve usar features sazonais (sine/cosine)
    """
    target_columns: List[str] = Field(
        default=[
            'temp_ar_c',
            'umidade_rel_ar_percent',
            'pressao_atm_estacao_mb',
            'precipitacao_total_mm'
        ],
        description="Colunas principais para filtrar o ruído"
    )
    lag_hours: List[int] = Field(
        default=[1, 24],
        description="Lags (horas) para aplicar nas features de temperatura"
    )
    moving_avg_windows: List[int] = Field(
        default=[6, 12],
        description="Tamanhos de janelas para média móvel"
    )
    use_seasonal_features: bool = Field(
        default=True,
        description="Se deve usar features sazonais (sine/cosine)"
    )
    remove_nan: bool = Field(
        default=True,
        description="Se deve remover linhas com valores NaN"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "target_columns": [
                    "temp_ar_c",
                    "umidade_rel_ar_percent",
                    "pressao_atm_estacao_mb",
                    "precipitacao_total_mm"
                ],
                "lag_hours": [1, 24],
                "moving_avg_windows": [6, 12],
                "use_seasonal_features": True,
                "remove_nan": True
            }
        }


class WeatherFeatureEngineerInput(BaseModel):
    """
    Input model para o Weather Feature Engineer.
    
    Attributes:
        dataframe_info: Descrição das colunas esperadas no DataFrame
        config: Configuração para feature engineering
    """
    dataframe_columns: List[str] = Field(
        ...,
        description="Colunas esperadas no DataFrame de entrada"
    )
    has_data_hora: bool = Field(
        default=True,
        description="Se o DataFrame possui coluna 'data_hora' com datetime"
    )
    config: FeatureEngineeringConfig = Field(
        default_factory=FeatureEngineeringConfig,
        description="Configuração para feature engineering"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "dataframe_columns": [
                    "data_hora",
                    "temp_ar_c",
                    "umidade_rel_ar_percent",
                    "pressao_atm_estacao_mb",
                    "precipitacao_total_mm",
                    "vento_vel_ms"
                ],
                "has_data_hora": True,
                "config": {
                    "target_columns": [
                        "temp_ar_c",
                        "umidade_rel_ar_percent",
                        "pressao_atm_estacao_mb",
                        "precipitacao_total_mm"
                    ],
                    "lag_hours": [1, 24],
                    "moving_avg_windows": [6, 12],
                    "use_seasonal_features": True,
                    "remove_nan": True
                }
            }
        }
