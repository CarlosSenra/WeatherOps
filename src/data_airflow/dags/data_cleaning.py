"""
Exemplo avançado de DAG com processamento em paralelo de múltiplas cidades.

Use este arquivo como referência para estender a DAG simples.
Para usar, renomeie para data_cleaning.py ou adapte ao seu pipeline.
"""

from airflow.decorators import dag, task
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import logging

from core.data_engineering import DataCleaning
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

RAW_DATA_PATH = str(PROJECT_ROOT / 'data' / 'raw')
STAGING_DATA_PATH = str(PROJECT_ROOT / 'data' / 'staging')

CITIES_TO_PROCESS = ['salvador_']


@dag(
    schedule=None,           
    start_date=datetime(2024, 1, 1),
    catchup=False,           
    tags=["Data Cleaning"],
)
def data_cleaning():

    @task()
    def get_paths_csv():
        dict_raw_brutos = map_csv_files_by_name(root_path=RAW_DATA_PATH, search_names=CITIES_TO_PROCESS)
        return dict_raw_brutos

    @task()
    def apply_data_clean(dict_raw_brutos):
        data_cleaner = DataCleaning(csv_paths=dict_raw_brutos['salvador_'])
        df_final = data_cleaner.concat_csv()
        path_csv_save = f"{STAGING_DATA_PATH}/{CITIES_TO_PROCESS[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        df_final.to_csv(path_csv_save, index=False)
        
    dict_ = get_paths_csv()
    apply_data_clean(dict_)

data_cleaning()