# WeatherOps

Plataforma de engenharia de dados e machine learning para previsao meteorologica com foco em series temporais, rastreabilidade de experimentos e preparacao para operacao em escala.

## Visao Geral

O projeto combina:

- Pipeline de dados meteorologicos (ingestao, limpeza e engenharia de atributos).
- Treinamento de modelos de previsao (LSTM e Transformer).
- Tracking de experimentos e artefatos com MLflow.
- Avaliacao de runs com graficos interativos Real vs Predito.

Fluxo macro:

1. Dados brutos em CSV sao processados.
2. Dados tratados sao publicados em Parquet (data/spec).
3. O modulo de treinamento consome os Parquets e executa experimentos.
4. Metricas, parametros, artefatos e modelo ficam versionados no MLflow.
5. O modulo de avaliacao gera analise visual por run_id.

## Arquitetura

- core/: componentes de engenharia de dados (limpeza e feature engineering).
- src/ml_workstation/: treinamento, tracking e avaliacao de modelos.
- data/: dados brutos, staging e dados prontos para consumo (spec).
- notebooks/: exploracao e analise exploratoria.

Arquivos tecnicos de referencia:

- src/ml_workstation/ARCHITECTURE.md
- src/ml_workstation/README.md
- src/ml_workstation/evaluation/README.md

## Stack Tecnologica

- Python 3.12+
- Poetry (gerenciamento de dependencias)
- PyTorch (modelagem)
- MLflow (tracking e model logging)
- Pandas e Scikit-learn (processamento e normalizacao)
- Plotly (visualizacao)
- DVC (versionamento de dados)
- Docker Compose (execucao isolada de treino e UI de tracking)

## Estrutura do Repositorio

```text
WeatherOps/
├── core/                        # Engenharia de dados
├── data/                        # raw, staging e spec
├── docs/                        # Documentacao complementar
├── notebooks/                   # EDA e estudos
├── src/
│   ├── data_airflow/            # Infra de orquestracao
│   └── ml_workstation/          # Treino, tracking, avaliacao
├── pyproject.toml               # Dependencias e metadados Python
├── docker-compose-airflow.yaml  # Stack airflow
└── data.dvc                     # Versionamento do dataset
```

## Requisitos

### Ambiente local

- Python >= 3.12
- Poetry instalado

### Ambiente containerizado

- Docker
- Docker Compose

## Setup Local

A partir da raiz do repositorio:

```bash
poetry install
```

Opcional para desenvolvimento:

```bash
poetry run pytest
```

## Treinamento de Modelos

### Opcao 1: Execucao local (Python/Poetry)

Da raiz do projeto:

```bash
poetry run python -m src.ml_workstation.train --config src/ml_workstation/experiments/lstm/lstm_h72_v1.json
```

### Opcao 2: Execucao via Docker Compose (recomendado para reproducibilidade)

Entre em src/ml_workstation e execute:

```bash
docker compose --profile train build trainer
docker compose --profile train run --rm trainer --config //app/experiments/lstm/lstm_h72_v1.json
```

Observacao para Git Bash no Windows:

- Use //app/... para caminhos dentro do container.

## Tracking com MLflow

Suba a UI do MLflow em src/ml_workstation:

```bash
docker compose --profile ui up mlflow-ui -d
```

Acesso:

- http://localhost:5000

Dados de tracking ficam persistidos em:

- src/ml_workstation/mlruns

## Avaliacao de Experimentos

Da raiz do projeto:

```bash
poetry run python -m src.ml_workstation.evaluation.run_evaluation --run-id <RUN_ID>
```

Saida esperada:

- Arquivo HTML com serie Real vs Predito em evaluation_results/.

## Governanca e Rastreabilidade

Cada run registra no MLflow:

- Parametros de configuracao completos (data/model/training).
- Metadados de governanca (model_name, versionamento, responsavel, risco, git_sha, versao de dados).
- Metricas por epoca e snapshot final.
- Checkpoint e modelo final como artefatos.

Isso permite auditoria e reproducao de resultados ponta a ponta.

## Boas Praticas Operacionais

1. Sempre versione alteracoes de configuracao em src/ml_workstation/experiments/lstm e src/ml_workstation/experiments/transformer.
2. Mantenha consistencia entre feature_columns e colunas reais dos Parquets.
3. Rode um experimento de smoke test antes de execucoes longas.
4. Monitore no MLflow as metricas de validacao para evitar overfitting.
5. Registre mudancas de dados e modelos com versionamento explicito.

## Licenca

Este projeto esta licenciado sob os termos definidos em LICENSE.

## Proximos Passos

Nos proximos ciclos, vamos implementar agentes de IA com Deep Agents e tambem criar e disponibilizar APIs dos modelos e dos agentes.
