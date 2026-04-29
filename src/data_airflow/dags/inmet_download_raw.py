"""DAG para baixar dados históricos anuais do INMET para data/raw/<ano>/.

Regras:
- Lê intervalo de anos via YAML.
- Faz scraping da página de históricos para resolver o link ZIP de cada ano.
- Baixa ZIP em memória.
- Sobrescreve data/raw/<ano>/ e extrai apenas CSVs.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import logging
import time

from airflow.decorators import dag, task

from src.data_airflow.scraping.inmet import (
    INMET_HISTORICAL_URL,
    download_bytes,
    extract_csv_zip_overwrite,
    find_year_zip_url,
    load_year_range_from_yaml,
)

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
}

try:
    if Path("/opt/project").exists():
        PROJECT_ROOT = Path("/opt/project")
    else:
        PROJECT_ROOT = Path(__file__).resolve().parents[3]
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
SCRAPING_CONFIG_PATH = PROJECT_ROOT / "src" / "data_airflow" / "config" / "inmet_scraping.yml"


@dag(
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["INMET", "Web Scraping", "Data Ingestion"],
)
def inmet_download_raw():
    @task()
    def load_year_range() -> list[int]:
        cfg = load_year_range_from_yaml(SCRAPING_CONFIG_PATH)
        years = cfg.years()
        logger.info("Intervalo carregado: %s", years)
        return years

    @task()
    def fetch_index_html() -> str:
        logger.info("Baixando página índice do INMET: %s", INMET_HISTORICAL_URL)
        return download_bytes(INMET_HISTORICAL_URL).decode("utf-8", errors="ignore")

    @task(max_active_tis_per_dag=1)
    def scrape_and_extract_year(year: int, page_html: str) -> dict:
        logger.info("Iniciando scraping do ano %s", year)
        time.sleep(2.5)
        zip_url = find_year_zip_url(INMET_HISTORICAL_URL, year, page_html)
        logger.info("Ano %s: URL do ZIP resolvida: %s", year, zip_url)
        zip_bytes = download_bytes(zip_url)

        year_dir = RAW_DATA_PATH / str(year)
        csv_count = extract_csv_zip_overwrite(zip_bytes, year_dir)
        result = {"year": year, "zip_url": zip_url, "csv_count": csv_count, "output_dir": str(year_dir)}
        logger.info("Ano %s processado com sucesso: %s", year, result)
        return result

    @task()
    def summarize_download(results: list[dict]) -> None:
        missing = [r for r in results if int(r["csv_count"]) <= 0]
        if missing:
            raise ValueError(f"Anos sem CSV extraído: {missing}")

        total_files = sum(int(r["csv_count"]) for r in results)
        logger.info("Download concluído: %d anos | %d CSVs", len(results), total_files)

    years = load_year_range()
    page_html = fetch_index_html()
    per_year = scrape_and_extract_year.partial(page_html=page_html).expand(year=years)
    summarize_download(per_year)


inmet_download_raw()

