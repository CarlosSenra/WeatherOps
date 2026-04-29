from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd


class WeatherFeatureEngineerOutput(BaseModel):
    """
    Output model para o Weather Feature Engineer.
    
    Attributes:
        num_rows: Número de linhas após transformação
        num_columns: Número de colunas após transformação
        original_features: Número de features originais
        new_features: Número de features criadas
        feature_names: Nomes de todas as features após transformação
        index_type: Tipo de índice do DataFrame (se houver)
        dtypes: Tipos de dados das colunas
    """
    num_rows: int = Field(
        ...,
        description="Número de linhas após transformação"
    )
    num_columns: int = Field(
        ...,
        description="Número de colunas após transformação (original + engineered)"
    )
    original_features: int = Field(
        ...,
        description="Número de features originais"
    )
    new_features: int = Field(
        ...,
        description="Número de features criadas (engineered)"
    )
    feature_names: List[str] = Field(
        ...,
        description="Nomes de todas as features após transformação"
    )
    index_type: str = Field(
        default="RangeIndex",
        description="Tipo de índice do DataFrame"
    )
    dtypes: Dict[str, str] = Field(
        ...,
        description="Dicionário com os tipos de dados de cada coluna"
    )

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        original_num_features: int
    ) -> "WeatherFeatureEngineerOutput":
        """
        Cria uma instância a partir de um DataFrame transformado.
        
        Args:
            df: DataFrame transformado
            original_num_features: Número de features antes da transformação
        """
        num_cols = len(df.columns)
        num_new_features = num_cols - original_num_features
        
        return WeatherFeatureEngineerOutput(
            num_rows=len(df),
            num_columns=num_cols,
            original_features=original_num_features,
            new_features=num_new_features,
            feature_names=df.columns.tolist(),
            index_type=str(type(df.index).__name__),
            dtypes={col: str(df[col].dtype) for col in df.columns}
        )

    class Config:
        json_schema_extra = {
            "example": {
                "num_rows": 8400,
                "num_columns": 16,
                "original_features": 4,
                "new_features": 12,
                "feature_names": [
                    "temp_ar_c",
                    "umidade_rel_ar_percent",
                    "pressao_atm_estacao_mb",
                    "precipitacao_total_mm",
                    "hora_sin",
                    "hora_cos",
                    "temp_lag_1h",
                    "temp_lag_24h",
                    "temp_ma_6h",
                    "pressao_ma_12h",
                    "pressao_tendencia_1h",
                    "temp_tendencia_1h"
                ],
                "index_type": "DatetimeIndex",
                "dtypes": {
                    "temp_ar_c": "float64",
                    "umidade_rel_ar_percent": "float64",
                    "pressao_atm_estacao_mb": "float64",
                    "precipitacao_total_mm": "float64",
                    "hora_sin": "float64",
                    "hora_cos": "float64",
                    "temp_lag_1h": "float64",
                    "temp_lag_24h": "float64",
                    "temp_ma_6h": "float64",
                    "pressao_ma_12h": "float64",
                    "pressao_tendencia_1h": "float64",
                    "temp_tendencia_1h": "float64"
                }
            }
        }
