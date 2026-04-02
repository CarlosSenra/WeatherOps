from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd


class DataCleaningProcessOutput(BaseModel):
    """
    Output model para o processamento de dados de limpeza.
    
    Attributes:
        num_rows: Número de linhas após processamento
        num_columns: Número de colunas após processamento
        columns: Lista de nomes das colunas
        dtypes: Dicionário com os tipos de dados de cada coluna
    """
    num_rows: int = Field(
        ..., 
        description="Número de linhas no dataframe processado"
    )
    num_columns: int = Field(
        ..., 
        description="Número de colunas no dataframe processado"
    )
    columns: List[str] = Field(
        ..., 
        description="Lista de nomes das colunas"
    )
    dtypes: Dict[str, str] = Field(
        ..., 
        description="Dicionário com os tipos de dados de cada coluna"
    )

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "DataCleaningProcessOutput":
        """Cria uma instância a partir de um DataFrame."""
        return DataCleaningProcessOutput(
            num_rows=len(df),
            num_columns=len(df.columns),
            columns=df.columns.tolist(),
            dtypes={col: str(df[col].dtype) for col in df.columns}
        )

    class Config:
        json_schema_extra = {
            "example": {
                "num_rows": 8760,
                "num_columns": 17,
                "columns": ["data_hora", "temp_ar_c", "umidade_rel_ar_percent"],
                "dtypes": {
                    "data_hora": "datetime64[ns]",
                    "temp_ar_c": "float64",
                    "umidade_rel_ar_percent": "float64"
                }
            }
        }


class DataCleaningPipelineOutput(BaseModel):
    """
    Output model para o pipeline de limpeza de dados.
    
    Attributes:
        num_files_processed: Número de arquivos processados
        final_dataframe_info: Informações sobre o dataframe final concatenado
        processing_status: Status do processamento
    """
    num_files_processed: int = Field(
        ...,
        description="Número de arquivos processados"
    )
    final_dataframe_info: DataCleaningProcessOutput = Field(
        ...,
        description="Informações sobre o dataframe final"
    )
    processing_status: str = Field(
        default="success",
        description="Status do processamento (success, partial_failure, failure)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "num_files_processed": 2,
                "final_dataframe_info": {
                    "num_rows": 17520,
                    "num_columns": 17,
                    "columns": ["data_hora", "temp_ar_c"],
                    "dtypes": {
                        "data_hora": "datetime64[ns]",
                        "temp_ar_c": "float64"
                    }
                },
                "processing_status": "success"
            }
        }
