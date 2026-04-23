# WeatherOps

Plataforma de engenharia de dados e machine learning para previsão meteorológica com foco em séries temporais, reprodutibilidade e rastreabilidade.

---

## Visão Geral

O WeatherOps conecta cinco blocos principais em um pipeline ponta a ponta:

1. **Engenharia de dados** em `core/` — limpeza e geração de features a partir de CSVs do INMET
2. **Orquestração de pipelines** em `src/data_airflow/` — DAGs Airflow para automatizar o ETL
3. **Treinamento e tracking** em `src/ml_workstation/` — quatro arquiteturas de deep learning com MLflow
4. **Promoção de modelos** via MLflow Model Registry — governança e controle de versão de produção
5. **Serving de previsões** em `src/api/` — API FastAPI servindo previsões horárias de temperatura

---

## Arquitetura Ponta a Ponta

```mermaid
flowchart LR
    A[data/raw CSV] --> B[DataCleaning]
    B --> C[WeatherFeatureEngineer]
    C --> D[data/spec Parquet]
    D --> E[Airflow DAGs]
    E --> D
    D --> F[train.py + JSON config]
    F --> G[MLflow Tracking]
    G --> H[evaluation HTML]
    G --> I[run_promote.py]
    I --> J[MLflow Registry\nalias production]
    I --> K[src/api/ml_models]
    J --> L[API FastAPI :8888]
    K --> L
    L --> M[POST /v1/forecast\n72 | 168 | 336h]
```

---

## Componentes

| Componente | Caminho | Responsabilidade | Docs |
|------------|---------|-----------------|------|
| **Core** | `core/` | Limpeza de dados e geração de features | [core/README.md](core/README.md) |
| **Airflow** | `src/data_airflow/` | Orquestração do pipeline de dados | [src/data_airflow/README.md](src/data_airflow/README.md) |
| **ML Workstation** | `src/ml_workstation/` | Treinamento, tracking, avaliação e promoção | [src/ml_workstation/README.md](src/ml_workstation/README.md) |
| **API de Serving** | `src/api/` | Previsões horárias via REST | [src/api/README.md](src/api/README.md) |
| **Documentação** | `docs/` | Setup, pipeline, troubleshooting | [docs/](docs/) |

---

## Endpoints Locais

Tabela de todos os serviços acessíveis via `localhost` ao rodar o projeto localmente.

| Serviço | URL | Descrição | Credenciais |
|---|---|---|---|
| WeatherOps API | `http://localhost:8888` | Serving de previsões meteorológicas | — |
| API — Swagger UI | `http://localhost:8888/docs` | Documentação interativa dos endpoints | — |
| API — ReDoc | `http://localhost:8888/redoc` | Documentação alternativa | — |
| API — Métricas | `http://localhost:8888/metrics` | Métricas Prometheus (formato `text/plain`) | — |
| API — Liveness | `http://localhost:8888/health` | Probe de liveness do container | — |
| API — Readiness | `http://localhost:8888/health/ready` | Probe de readiness (modelos carregados) | — |
| Prometheus | `http://localhost:9090` | Coleta e consulta de métricas via PromQL | — |
| Grafana | `http://localhost:3000` | Dashboards de monitoramento | admin / admin |
| MLflow UI (api compose) | `http://localhost:5001` | Tracking de experimentos (via `src/api/docker-compose.yml`) | — |
| MLflow UI (workstation) | `http://localhost:5000` | Tracking de experimentos (via `src/ml_workstation/docker-compose.yml`) | — |
| Airflow Webserver | `http://localhost:8080` | Orquestração de pipelines ETL | airflow / airflow |

> Prometheus e Grafana sobem automaticamente com `docker compose --profile api up` via `src/api/docker-compose.yml`.
> O datasource do Prometheus é provisionado no Grafana sem configuração manual.

---

## Pré-Requisitos

| Ferramenta | Versão | Obrigatório |
|------------|--------|------------|
| Python | 3.12+ | Sim |
| Poetry | 1.8+ | Sim |
| Docker + Docker Compose V2 | 24+ | Sim |
| Git | 2.40+ | Sim |
| curl | qualquer | Recomendado (verificar API) |
| NVIDIA Driver + nvidia-container-toolkit | — | Apenas para treino com GPU |

---

## Estrutura do Repositório

```text
WeatherOps/
├── core/                          # Módulo de engenharia de dados
│   └── data_engineering/          # DataCleaning + WeatherFeatureEngineer
├── data/
│   ├── raw/                       # CSVs brutos do INMET
│   ├── staging/                   # CSVs limpos (intermediário)
│   └── spec/                      # Parquets com features (entrada do treino)
├── docs/                          # Documentação geral
│   ├── DEVELOPMENT_SETUP.md
│   ├── PIPELINE_FLOW.md
│   └── TROUBLESHOOTING.md
├── notebooks/                     # Experimentos exploratórios
├── src/
│   ├── data_airflow/              # DAGs Airflow
│   │   └── dags/
│   ├── ml_workstation/            # Treinamento e promoção
│   │   ├── config/
│   │   ├── data/
│   │   ├── evaluation/
│   │   ├── experiments/           # JSONs de configuração
│   │   ├── models/
│   │   ├── promotion/
│   │   ├── tracking/
│   │   ├── training/
│   │   └── train.py
│   └── api/                       # API FastAPI de serving
│       ├── engines/
│       ├── routers/
│       ├── schemas/
│       ├── services/
│       ├── main.py
│       ├── config.py
│       ├── Dockerfile
│       └── docker-compose.yml
├── test/                          # Testes unitários e de integração
├── evaluation_results/            # HTMLs gerados pela avaliação
├── pyproject.toml
├── docker-compose-airflow.yaml
└── data.dvc
```

---

## Stack Tecnológica

| Categoria | Tecnologias |
|-----------|------------|
| **Linguagem** | Python 3.12+ |
| **Gerenciamento** | Poetry, DVC |
| **Dados** | Pandas, Scikit-learn, PyArrow |
| **Deep Learning** | PyTorch, pytorch-forecasting, PyTorch Lightning |
| **Tracking** | MLflow |
| **Visualização** | Plotly |
| **API** | FastAPI, uvicorn, pydantic-settings |
| **Cache** | In-memory (padrão) ou Redis |
| **Observabilidade** | Prometheus, Grafana, prometheus-fastapi-instrumentator |
| **Orquestração** | Apache Airflow (CeleryExecutor) |
| **Containerização** | Docker Compose |

---

## Mapa de Documentação

| Documento | Conteúdo |
|-----------|---------|
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | **Arquitetura geral** — grafos do sistema completo com todos os componentes |
| [docs/DATA_ENGINEERING_WORKFLOW.md](docs/DATA_ENGINEERING_WORKFLOW.md) | **Fluxo ETL detalhado** — DataCleaning, features, DAGs Airflow com grafos passo a passo |
| [docs/ML_PIPELINE_WORKFLOW.md](docs/ML_PIPELINE_WORKFLOW.md) | **Pipeline de ML detalhado** — treinamento, promoção e ciclo de inferência da API com grafos |
| [docs/DEVELOPMENT_SETUP.md](docs/DEVELOPMENT_SETUP.md) | Setup do ambiente, Docker, API e fluxo de desenvolvimento |
| [docs/PIPELINE_FLOW.md](docs/PIPELINE_FLOW.md) | Fluxo ponta a ponta com contratos entre camadas |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Erros comuns e soluções (treino, Airflow e API) |
| [core/README.md](core/README.md) | Módulo de engenharia de dados |
| [core/ARCHITECTURE.md](core/ARCHITECTURE.md) | Arquitetura e acoplamentos do core |
| [core/data_engineering/README.md](core/data_engineering/README.md) | DataCleaning, WeatherFeatureEngineer, schema de saída |
| [src/data_airflow/README.md](src/data_airflow/README.md) | Setup e uso do Airflow |
| [src/ml_workstation/README.md](src/ml_workstation/README.md) | Treinamento, tracking, avaliação e promoção |
| [src/ml_workstation/ARCHITECTURE.md](src/ml_workstation/ARCHITECTURE.md) | Arquitetura de 5 camadas do ML Workstation |
| [src/ml_workstation/evaluation/README.md](src/ml_workstation/evaluation/README.md) | Avaliação visual de runs |
| [src/ml_workstation/promotion/PROMOTION.md](src/ml_workstation/promotion/PROMOTION.md) | Promoção de modelos para o Registry |
| [src/api/README.md](src/api/README.md) | API de serving: endpoints, variáveis e como usar |
| [src/api/ARCHITECTURE.md](src/api/ARCHITECTURE.md) | Arquitetura técnica da API (engines, cache, tracing) |

---

## Fluxo de Trabalho Completo

Sequência para ir do zero até a API servindo previsões:

1. **Preparar dados brutos** — colocar CSVs do INMET em `data/raw/`
2. **Executar limpeza** — via Airflow DAG `data_cleaning` ou diretamente com `DataCleaning`
3. **Executar feature engineering** — via Airflow DAG `data_feature_engineering` ou `WeatherFeatureEngineer`
4. **Treinar modelo** — `docker compose --profile train run --rm trainer --config //app/experiments/tft/tft_h72_v1.json`
5. **Avaliar run** — `poetry run python -m src.ml_workstation.evaluation.run_evaluation --run-id <RUN_ID>`
6. **Promover modelo** — `poetry run python -m src.ml_workstation.promotion.run_promote --experiment-name weather_forecasting_h72 --export-dir src/api/ml_models`
7. **Subir API** — `docker compose -f src/api/docker-compose.yml --profile api up --build`

---

## Setup Rápido

Instalação local:

```bash
poetry install
```

Testes:

```bash
poetry run pytest
poetry run pytest --cov=core --cov=src --cov-report=term-missing
```

---

## Airflow

```bash
# Inicialização (apenas na primeira vez)
docker compose -f docker-compose-airflow.yaml up airflow-init

# Subir stack
docker compose -f docker-compose-airflow.yaml up -d
```

Acesso: `http://localhost:8080` (usuário: `airflow`, senha: `airflow`)

---

## Treinamento

Treino via Docker (CPU):

```bash
cd src/ml_workstation
docker compose --profile train build trainer
docker compose --profile train run --rm trainer --config //app/experiments/lstm/lstm_h72_v1.json
```

Treino via Docker (GPU):

```bash
docker compose --profile train-gpu run --rm trainer-gpu --config //app/experiments/tft/tft_h72_v1.json
```

Treino local via Poetry:

```bash
poetry run python -m src.ml_workstation.train --config src/ml_workstation/experiments/lstm/lstm_h72_v1.json
```

> **Windows + Git Bash:** use `//app/...` (dupla barra) para caminhos de arquivos dentro do container.
>
> **Rebuild necessário** se o `pyproject.toml` for alterado.

---

## MLflow

Subir UI local:

```bash
cd src/ml_workstation
docker compose --profile ui up -d mlflow-ui
```

Acesso: `http://localhost:5000`

---

## Avaliação de Runs

```bash
poetry run python -m src.ml_workstation.evaluation.run_evaluation --run-id <RUN_ID>
```

Saída: HTML em `evaluation_results/` com gráfico Real vs Predito.

Suporta LSTM e Transformer. Para TFT/N-BEATS, as métricas estão no MLflow UI.

---

## Promoção de Modelos

O MLflow Model Registry é a fonte de verdade para modelos em produção. Cada modelo promovido recebe o alias `production` e é validado contra regressão de MAPE.

```bash
# Promover o melhor run por MAPE e exportar para a API
poetry run python -m src.ml_workstation.promotion.run_promote \
  --experiment-name weather_forecasting_h72 \
  --export-dir src/api/ml_models
```

Use `--force` para sobrescrever a proteção contra regressão de MAPE.

Detalhes: [src/ml_workstation/promotion/PROMOTION.md](src/ml_workstation/promotion/PROMOTION.md)

---

## API de Serving

```bash
# A partir da raiz do repositório
docker compose -f src/api/docker-compose.yml --profile api up --build
```

Verificar prontidão (aguardar ~30s para carregamento dos modelos):

```bash
curl http://localhost:8888/health/ready
```

Forecast de 72 horas:

```bash
curl -X POST http://localhost:8888/v1/forecast/72 \
  -H "Content-Type: application/json" \
  -d '{"reference_date": "2024-06-01", "model_type": "tft", "group_id": "station_1"}'
```

Documentação interativa: `http://localhost:8888/docs`

Horizontes disponíveis: `72`, `168`, `336` horas.

---

## Qualidade e Governança

- Cobertura de testes mínima: **60%** (verificada no CI/CD)
- Configs de experimento versionadas em `src/ml_workstation/experiments/`
- Metadados de governança registrados no MLflow (owner, git_sha, data_version, risk_level)
- Pipeline de CI no GitHub Actions: `push` e `PR` no branch `develop` executam os testes com cobertura
- Dados versionados com DVC (`data.dvc`)

---

## Licença

Este projeto está licenciado conforme `LICENSE`.
