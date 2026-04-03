import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import os

# Adiciona o diretório raiz ao path para permitir imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.data_engineering.data_cleaning.data_cleaning import DataCleaning
from core.data_engineering.models.data_cleaning.inputs import (
    CsvReadConfig,
    DataConversionConfig,
    ColumnRenameConfig,
)


@pytest.fixture
def temporary_csv_file():
    """Cria um arquivo CSV temporário com dados de teste."""
    with TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "test_data.csv"
        
        # Dados de teste com linhas de cabeçalho extras (primeiras 8 linhas)
        header_lines = "\n".join([
            "Estação Meteorológica de Teste",
            "Longitude:",
            "Latitude:",
            "Altitude:",
            "UF:",
            "Fonte:",
            "",
            "Variáveis:"
        ])
        
        # Cria um DataFrame de teste
        df = pd.DataFrame({
            'Data': ['2024/01/01', '2024/01/02', '2024/01/03'],
            'Hora UTC': ['00UTC', '01UTC', '02UTC'],
            'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': ['25,5', '26,3', '25,8'],
            'UMIDADE RELATIVA DO AR, HORARIA (%)': ['70,5', '68,3', '72,1'],
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': ['1013,25', '1013,15', '1013,35'],
            'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': ['0,0', '2,5', '0,0'],
            'Coluna_Descartada': ['a', 'b', 'c']  # Última coluna é descartada
        })
        
        # Escreve o CSV com as linhas de cabeçalho
        with open(csv_path, 'w', encoding='latin-1') as f:
            f.write(header_lines + '\n')
            df.to_csv(f, sep=';', index=False, encoding='latin-1')
        
        yield csv_path


@pytest.fixture
def multiple_csv_files():
    """Cria múltiplos arquivos CSV temporários."""
    with TemporaryDirectory() as temp_dir:
        csv_paths = []
        
        for i in range(2):
            csv_path = Path(temp_dir) / f"test_data_{i}.csv"
            
            header_lines = "\n".join([
                "Estação Meteorológica de Teste",
                "Longitude:",
                "Latitude:",
                "Altitude:",
                "UF:",
                "Fonte:",
                "",
                "Variáveis:"
            ])
            
            df = pd.DataFrame({
                'Data': [f'2024/01/0{i+1}', f'2024/01/0{i+2}'],
                'Hora UTC': ['00UTC', '01UTC'],
                'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': ['25,5', '26,3'],
                'UMIDADE RELATIVA DO AR, HORARIA (%)': ['70,5', '68,3'],
                'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': ['1013,25', '1013,15'],
                'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': ['0,0', '2,5'],
                'Coluna_Descartada': ['a', 'b']
            })
            
            with open(csv_path, 'w', encoding='latin-1') as f:
                f.write(header_lines + '\n')
                df.to_csv(f, sep=';', index=False, encoding='latin-1')
            
            csv_paths.append(csv_path)
        
        yield csv_paths


@pytest.fixture
def sample_dataframe():
    """Cria um DataFrame de teste com dados não processados."""
    return pd.DataFrame({
        'Data': ['2024/01/01', '2024/01/02'],
        'Hora UTC': ['00UTC', '01UTC'],
        'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': ['25,5', '26,3'],
        'UMIDADE RELATIVA DO AR, HORARIA (%)': ['70,5', '68,3'],
        'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': ['1013,25', '1013,15'],
    })


@pytest.fixture
def csv_read_config():
    """Retorna configuração padrão para leitura de CSV."""
    return CsvReadConfig(
        sep=';',
        encoding='latin-1',
        skiprows_start=8
    )


@pytest.fixture
def data_conversion_config():
    """Retorna configuração padrão para conversão de dados."""
    return DataConversionConfig(
        dtypes_to_convert=['object', 'string'],
        decimal_separator=','
    )


@pytest.fixture
def column_rename_config():
    """Retorna configuração padrão para renomeação de colunas."""
    return ColumnRenameConfig(
        rename_map={
            'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'temp_ar_c',
            'UMIDADE RELATIVA DO AR, HORARIA (%)': 'umidade_rel_ar_percent',
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': 'pressao_atm_estacao_mb',
            'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'precipitacao_total_mm',
            'DATA_HORA': 'data_hora'
        }
    )
