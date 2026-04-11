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
