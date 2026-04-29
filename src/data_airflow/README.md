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

## Runner automatizado (user pipeline)

Após clonar o repositório, é possível rodar o fluxo completo via scripts:

```bash
# Windows (PowerShell)
./scripts/run_pipeline.ps1 -Mode bootstrap -Device auto -StartYear 2024 -EndYear 2026

# Linux/macOS (Bash)
./scripts/run_pipeline.sh --mode bootstrap --device auto --start-year 2024 --end-year 2026
```

Esses scripts sobem Airflow, executam as DAGs de dados e validam modelos exportados da API.

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

## Configuracao de municipios

Municipios processados pelas DAGs de limpeza e feature engineering sao
descobertos automaticamente a partir dos arquivos INMET em `data/raw/`.
Os filtros e overrides sao definidos em:

- `src/data_airflow/config/municipios.yml`

Formato:

```yaml
mode: all
include: []
exclude: []
slug_overrides: {}
```

Exemplos de nomes detectados:

- `INMET_CO_DF_A001_BRASILIA_01-01-2026_A_31-03-2026.CSV` -> `brasilia`
- `INMET_S_RS_A881_DOM PEDRITO_01-01-2026_A_31-03-2026.CSV` -> `dom_pedrito`

Para restringir execucao, use:
- `include`: processa somente os slugs listados;
- `exclude`: remove slugs da lista final;
- `slug_overrides`: renomeia slugs detectados.

Comportamento das DAGs com multiplos municipios:

- `data_cleaning`: descobre e agrupa CSVs por municipio, salva em `data/staging/<municipio_slug>/<municipio_slug>.csv` (sobrescreve, sem timestamp).
- `data_feature_engineering`: le o CSV de staging por municipio, aplica feature engineering e salva em `data/spec/<municipio_slug>/dados.parquet` (sobrescreve, sem timestamp).

### Exemplos simples de configuracao

Todos os exemplos abaixo sao configurados em `src/data_airflow/config/municipios.yml`.

#### 1) Processar somente Salvador

```yaml
mode: all
include:
  - salvador
exclude: []
slug_overrides: {}
```

#### 2) Processar somente um municipio (exemplo: Brasilia)

```yaml
mode: all
include:
  - brasilia
exclude: []
slug_overrides: {}
```

#### 3) Processar dois ou mais municipios

```yaml
mode: all
include:
  - salvador
  - brasilia
  - dom_pedrito
exclude: []
slug_overrides: {}
```

#### 4) Processar todos os municipios detectados

```yaml
mode: all
include: []
exclude: []
slug_overrides: {}
```

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
