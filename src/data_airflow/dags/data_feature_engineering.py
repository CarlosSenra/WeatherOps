"""
DAG para aplicar Feature Engineering nos dados de staging.

Lê arquivos CSV de staging (ex: salvador_*.csv) e aplica transformações
de feature engineering, salvando o resultado em parquet em data/spec.
"""

from airflow.decorators import dag, task
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import logging
import pandas as pd

from core.data_engineering.data_feature_eng import WeatherFeatureEngineer
from core.utils.manege_files import map_csv_files_by_name

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'airflow',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
}

try:
    if Path('/opt/project').exists():
        PROJECT_ROOT = Path('/opt/project')
    else:
        PROJECT_ROOT = Path(__file__).resolve().parents[3]
except:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

STAGING_DATA_PATH = str(PROJECT_ROOT / 'data' / 'staging')
SPEC_DATA_PATH = str(PROJECT_ROOT / 'data' / 'spec')

CITIES_TO_PROCESS = ['salvador_']


@dag(
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["Feature Engineering"],
)
def data_feature_engineering():

    @task()
    def get_staging_files():
        """Busca arquivos CSV de staging com padrão de nome especificado."""
        dict_staging = map_csv_files_by_name(
            root_path=STAGING_DATA_PATH,
            search_names=CITIES_TO_PROCESS
        )
        return dict_staging

    @task()
    def apply_feature_engineering(dict_staging):
        """
        Aplica feature engineering nos dados de staging.
        
        Lê os CSV do staging, aplica as transformações de feature engineering
        e salva em parquet na pasta de spec.
        """
        # Inicializa o feature engineer
        feature_engineer = WeatherFeatureEngineer()
        
        # Processa arquivo de salvador
        if 'salvador_' in dict_staging:
            csv_files = dict_staging['salvador_']
            
            # Concatena todos os CSVs se houver múltiplos
            if isinstance(csv_files, list):
                dfs = [pd.read_csv(file) for file in csv_files]
                df_input = pd.concat(dfs, ignore_index=True)
            else:
                df_input = pd.read_csv(csv_files)
            
            # Aplica feature engineering
            df_engineered = feature_engineer.transform(df_input)
            
            # Define caminho de saída em parquet
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            output_path = f"{SPEC_DATA_PATH}/{CITIES_TO_PROCESS[0]}{timestamp}.parquet"
            
            # Garante que o diretório existe
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Salva em parquet
            df_engineered.to_parquet(output_path, index=True)
            logger.info(f"Feature engineering aplicado e salvo em: {output_path}")
            
            return {
                'output_path': output_path,
                'rows_processed': len(df_engineered),
                'columns': list(df_engineered.columns)
            }

    dict_staging = get_staging_files()
    apply_feature_engineering(dict_staging)


data_feature_engineering()
