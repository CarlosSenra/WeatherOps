# Data Airflow

Modulo de orquestracao de pipelines com Apache Airflow.

## Objetivo

Orquestrar execucoes de engenharia de dados e rotinas relacionadas de forma observavel e reproduzivel.

## Estrutura

```text
src/data_airflow/
├── Dockerfile
├── dags/
├── config/
├── logs/
├── plugins/
└── __init__.py
```

## Como subir o ambiente

Comandos a partir da raiz do repositorio:

```bash
docker compose -f docker-compose-airflow.yaml up airflow-init
docker compose -f docker-compose-airflow.yaml up -d
```

Acesso:
- Airflow Webserver: `http://localhost:8080`
- Usuario/senha default (local): `airflow` / `airflow`

## DAG de scraping INMET (dados brutos por ano)

DAG `inmet_download_raw` para baixar os dados anuais do INMET e
popular `data/raw/<ano>/` com CSVs (sem salvar `.zip` no disco).

Configuracao em:

- `src/data_airflow/config/inmet_scraping.yml`

Formato:

```yaml
start_year: 2024
end_year: 2026
```

Comportamento:

- Intervalo inclusivo (`2024..2026` baixa 2024, 2025, 2026).
- Se `data/raw/<ano>/` ja existir, o conteudo e sobrescrito.
- Apenas arquivos `.csv` sao extraidos.

Ordem recomendada de execucao no Airflow:

1. `inmet_download_raw`
2. `data_cleaning`
3. `data_feature_engineering`

## Volumes montados (resumo)

- `./src/data_airflow/dags` -> `/opt/airflow/dags`
- `./src/data_airflow/logs` -> `/opt/airflow/logs`
- `./src/data_airflow/config` -> `/opt/airflow/config`
- `./src/data_airflow/plugins` -> `/opt/airflow/plugins`
- `./core` -> `/opt/project/core`
- `./src` -> `/opt/project/src`
- `./data` -> `/opt/project/data`

## Integracao com o projeto

O Airflow pode disparar etapas que leem dados em `data/raw`, processam via modulo `core` e publicam resultados para consumo do `ml_workstation`.

## Observacoes

- Esta stack e orientada a desenvolvimento local.
- Para producao, ajustar secrets, persistencia, seguranca e observabilidade.
