"""
DAG de limpeza de dados meteorológicos do INMET.

Descobre automaticamente municípios a partir dos nomes de arquivos INMET
em data/raw/, aplica filtros do YAML de configuração e salva o resultado
limpo em data/staging/<municipio_slug>/<municipio_slug>.csv.

O arquivo de saída é sobrescrito a cada execução (sem timestamp),
garantindo que staging sempre reflita os dados mais recentes.
"""

from airflow.decorators import dag, task
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
import logging

import yaml

from core.data_engineering import DataCleaning
from core.utils.manege_files import apply_municipio_filters, map_inmet_csv_files_by_municipio

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
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_PATH = str(PROJECT_ROOT / 'data' / 'raw')
STAGING_DATA_PATH = str(PROJECT_ROOT / 'data' / 'staging')
MUNICIPIOS_CONFIG_PATH = PROJECT_ROOT / 'src' / 'data_airflow' / 'config' / 'municipios.yml'


def _load_municipios() -> Dict[str, Any]:
    """Carrega configuração de municípios do YAML."""
    raw = yaml.safe_load(MUNICIPIOS_CONFIG_PATH.read_text(encoding='utf-8')) or {}

    # Compatibilidade retroativa para formato legado.
    if "municipios" in raw:
        include = [m["nome"] for m in raw.get("municipios", []) if isinstance(m, dict) and m.get("nome")]
        return {
            "mode": "all",
            "include": include,
            "exclude": [],
            "slug_overrides": {},
        }

    mode = raw.get("mode", "all")
    if mode != "all":
        raise ValueError(f"mode inválido em {MUNICIPIOS_CONFIG_PATH}: '{mode}'. Use 'all'.")

    return {
        "mode": "all",
        "include": raw.get("include", []) or [],
        "exclude": raw.get("exclude", []) or [],
        "slug_overrides": raw.get("slug_overrides", {}) or {},
    }


@dag(
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["Data Cleaning"],
)
def data_cleaning():

    @task()
    def get_config() -> Dict[str, Any]:
        """Lê configuração de municípios do YAML."""
        config = _load_municipios()
        logger.info(
            "Config municípios: mode=%s include=%s exclude=%s overrides=%s",
            config["mode"], config["include"], config["exclude"], config["slug_overrides"],
        )
        return config

    @task()
    def get_paths_csv(config: Dict[str, Any]) -> Dict[str, list[str]]:
        """Descobre CSVs INMET em raw e aplica filtros de municípios."""
        detected = map_inmet_csv_files_by_municipio(root_path=RAW_DATA_PATH)
        filtered = apply_municipio_filters(
            grouped_files=detected,
            include=config["include"],
            exclude=config["exclude"],
            slug_overrides=config["slug_overrides"],
        )
        logger.info("Municípios detectados em raw: %s", list(detected.keys()))
        logger.info("Municípios após filtros: %s", list(filtered.keys()))
        logger.info("Arquivos por município: %s", {k: len(v) for k, v in filtered.items()})
        return filtered

    @task()
    def apply_data_clean(dict_raw: Dict[str, list[str]]) -> None:
        """Aplica limpeza para cada município e salva em staging/<slug>/<slug>.csv."""
        for municipio_slug, csv_paths in dict_raw.items():
            if not csv_paths:
                logger.warning("Nenhum CSV encontrado para o município '%s'", municipio_slug)
                continue

            logger.info("Processando '%s': %d arquivo(s) encontrado(s)", municipio_slug, len(csv_paths))

            data_cleaner = DataCleaning(csv_paths=csv_paths)
            df_final = data_cleaner.concat_csv()

            output_dir = Path(STAGING_DATA_PATH) / municipio_slug
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{municipio_slug}.csv"
            df_final.to_csv(output_path, index=False)

            logger.info("Staging salvo em: %s (%d linhas)", output_path, len(df_final))

    config = get_config()
    dict_raw = get_paths_csv(config)
    apply_data_clean(dict_raw)


data_cleaning()
