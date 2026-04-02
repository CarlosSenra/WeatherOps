from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from typing import List, Dict


@dataclass
class DataEngInput:
    """
    Input model para operações de engenharia de dados.
    
    Attributes:
        csv_paths: Lista de caminhos para os arquivos CSV
    """
    csv_paths: List[str]


class CsvReadConfig(BaseModel):
    """
    Configuração para leitura de arquivos CSV.
    
    Attributes:
        sep: Separador do arquivo CSV
        encoding: Codificação do arquivo
        skiprows_start: Número de linhas a pular no início
    """
    sep: str = Field(
        default=';',
        description="Separador do arquivo CSV"
    )
    encoding: str = Field(
        default='latin-1',
        description="Codificação do arquivo CSV"
    )
    skiprows_start: int = Field(
        default=8,
        description="Número de linhas a pular no início"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "sep": ";",
                "encoding": "latin-1",
                "skiprows_start": 8
            }
        }


class DataConversionConfig(BaseModel):
    """
    Configuração para conversão de tipos de dados.
    
    Attributes:
        dtypes_to_convert: Lista de tipos de dados para converter para numérico
        decimal_separator: Caractere usado como separador decimal (padrão: ',')
    """
    dtypes_to_convert: List[str] = Field(
        default=['object', 'string'],
        description="Tipos de dados a converter para numérico"
    )
    decimal_separator: str = Field(
        default=',',
        description="Caractere usado como separador decimal"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "dtypes_to_convert": ["object", "string"],
                "decimal_separator": ","
            }
        }


class ColumnRenameConfig(BaseModel):
    """
    Configuração para renomeação de colunas.
    
    Attributes:
        rename_map: Dicionário de mapeamento de nomes antigos para novos
    """
    rename_map: Dict[str, str] = Field(
        ...,
        description="Dicionário com mapeamento de nomes antigos para novos"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "rename_map": {
                    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": "precipitacao_total_mm",
                    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": "temp_ar_c"
                }
            }
        }


class DataCleaningPipelineInput(BaseModel):
    """
    Input model para o pipeline de limpeza de dados.
    
    Attributes:
        csv_paths: Lista de caminhos para os arquivos CSV
        csv_read_config: Configuração para leitura de CSV
        data_conversion_config: Configuração para conversão de dados
        column_rename_config: Configuração para renomeação de colunas
    """
    csv_paths: List[str] = Field(
        ...,
        description="Lista de caminhos para arquivos CSV"
    )
    csv_read_config: CsvReadConfig = Field(
        default_factory=CsvReadConfig,
        description="Configuração para leitura de CSV"
    )
    data_conversion_config: DataConversionConfig = Field(
        default_factory=DataConversionConfig,
        description="Configuração para conversão de dados"
    )
    column_rename_config: ColumnRenameConfig = Field(
        ...,
        description="Configuração para renomeação de colunas"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "csv_paths": [
                    "/data/raw/2024/brasilia_001.csv",
                    "/data/raw/2024/brasilia_002.csv"
                ],
                "csv_read_config": {
                    "sep": ";",
                    "encoding": "latin-1",
                    "skiprows_start": 8
                },
                "data_conversion_config": {
                    "dtypes_to_convert": ["object", "string"],
                    "decimal_separator": ","
                },
                "column_rename_config": {
                    "rename_map": {
                        "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": "temp_ar_c"
                    }
                }
            }
        }
