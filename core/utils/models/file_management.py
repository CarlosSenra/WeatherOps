from pydantic import BaseModel, Field
from typing import List, Dict


class MapCsvFilesInput(BaseModel):
    """
    Input model para a função map_csv_files_by_name.
    
    Attributes:
        root_path: Caminho raiz para começar a busca pelos arquivos CSV
        search_names: Lista de nomes para filtrar os arquivos CSV
    """
    root_path: str = Field(
        ..., 
        description="Caminho raiz para a busca de arquivos CSV"
    )
    search_names: List[str] = Field(
        ..., 
        description="Lista de nomes para filtrar os arquivos CSV"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "root_path": "/data/raw",
                "search_names": ["salvador_", "brasilia_"]
            }
        }


class MapCsvFilesOutput(BaseModel):
    """
    Output model para a função map_csv_files_by_name.
    
    Attributes:
        results: Dicionário onde as chaves são os nomes de busca 
                 e os valores são listas de caminhos absolutos dos arquivos CSV encontrados
    """
    results: Dict[str, List[str]] = Field(
        ..., 
        description="Dicionário com os resultados da busca"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "results": {
                    "salvador_": [
                        "/data/raw/2024/salvador_001.csv",
                        "/data/raw/2024/salvador_002.csv"
                    ],
                    "brasilia_": [
                        "/data/raw/2024/brasilia_001.csv"
                    ]
                }
            }
        }
