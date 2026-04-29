"""
DAG de feature engineering sobre os dados de staging.

Descobre municípios a partir de staging/<municipio_slug>/<municipio_slug>.csv,
aplica filtros do YAML e salva os resultados em
data/spec/<municipio_slug>/dados.parquet.

O arquivo de saída é sobrescrito a cada execução (sem timestamp),
garantindo que spec sempre reflita os dados mais recentes de staging.
"""

from airflow.decorators import dag, task
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
import logging

import pandas as pd
import yaml

from core.data_engineering.data_feature_eng import WeatherFeatureEngineer
from core.utils.manege_files import apply_municipio_filters

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

STAGING_DATA_PATH = PROJECT_ROOT / 'data' / 'staging'
SPEC_DATA_PATH = PROJECT_ROOT / 'data' / 'spec'
MUNICIPIOS_CONFIG_PATH = PROJECT_ROOT / 'src' / 'data_airflow' / 'config' / 'municipios.yml'


def _load_municipios_config() -> Dict[str, Any]:
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


def _discover_staging_files_by_municipio(staging_path: Path) -> Dict[str, List[str]]:
    """Descobre arquivos staging no padrão <slug>/<slug>.csv."""
    grouped: Dict[str, List[str]] = {}
    if not staging_path.exists():
        return grouped

    for csv_file in staging_path.rglob("*.csv"):
        if not csv_file.is_file():
            continue
        municipio_slug = csv_file.parent.name
        expected_name = f"{municipio_slug}.csv"
        if csv_file.name != expected_name:
            continue
        grouped.setdefault(municipio_slug, []).append(str(csv_file))
    return {k: sorted(v) for k, v in sorted(grouped.items(), key=lambda x: x[0])}


@dag(
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["Feature Engineering"],
)
def data_feature_engineering():

    @task()
    def get_municipios() -> List[str]:
        """Descobre municípios a partir de staging e aplica filtros do YAML."""
        config = _load_municipios_config()
        detected = _discover_staging_files_by_municipio(STAGING_DATA_PATH)
        filtered = apply_municipio_filters(
            grouped_files=detected,
            include=config["include"],
            exclude=config["exclude"],
            slug_overrides=config["slug_overrides"],
        )
        municipios = list(filtered.keys())
        logger.info("Municípios detectados em staging: %s", list(detected.keys()))
        logger.info("Municípios após filtros: %s", municipios)
        return municipios

    @task()
    def apply_feature_engineering(municipios: List[str]) -> List[dict]:
        """
        Aplica feature engineering para cada município.

        Lê o CSV de staging/<municipio>/<municipio>.csv, aplica as transformações
        e salva o resultado em spec/<municipio>/dados.parquet.
        """
        feature_engineer = WeatherFeatureEngineer()
        results = []

        for municipio_slug in municipios:
            staging_csv = STAGING_DATA_PATH / municipio_slug / f"{municipio_slug}.csv"
            if not staging_csv.exists():
                logger.warning(
                    "Arquivo de staging não encontrado para '%s': %s — pulando.",
                    municipio_slug,
                    staging_csv,
                )
                continue

            logger.info("Processando feature engineering para '%s'", municipio_slug)

            df_input = pd.read_csv(staging_csv)
            df_engineered = feature_engineer.transform(df_input)

            output_dir = SPEC_DATA_PATH / municipio_slug
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / 'dados.parquet'
            df_engineered.to_parquet(output_path, index=True)

            logger.info(
                "Spec salvo em: %s (%d linhas, %d colunas)",
                output_path, len(df_engineered), len(df_engineered.columns),
            )
            results.append({
                'municipio': municipio_slug,
                'output_path': str(output_path),
                'rows_processed': len(df_engineered),
                'columns': list(df_engineered.columns),
            })

        return results

    municipios = get_municipios()
    apply_feature_engineering(municipios)


data_feature_engineering()
