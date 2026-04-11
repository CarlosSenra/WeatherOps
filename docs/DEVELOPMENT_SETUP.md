# Setup de Desenvolvimento

Guia de setup para contribuir no WeatherOps de forma reproduzivel.

## Requisitos

- Python 3.12+
- Poetry
- Docker
- Git

## Setup Local

Na raiz do repositorio:

```bash
poetry install
```

Executar testes:

```bash
poetry run pytest
poetry run pytest --cov=core --cov=src --cov-report=term-missing
```

## Setup de Treino com Docker

Treino CPU (perfil `train`):

```bash
cd src/ml_workstation
docker compose --profile train build trainer
docker compose --profile train run --rm trainer --config //app/experiments/lstm/lstm_h72_v1.json
```

Treino GPU (perfil `train-gpu`):

```bash
docker compose --profile train-gpu run --rm --entrypoint python trainer-gpu -c "import torch; print(torch.cuda.is_available())"
docker compose --profile train-gpu run --rm trainer-gpu --config //app/experiments/lstm/lstm_h72_v1.json
```

Observacao para Git Bash no Windows: use caminho de container com `//app/...`.

## Setup de Airflow

Na raiz do repositorio:

```bash
docker compose -f docker-compose-airflow.yaml up airflow-init
docker compose -f docker-compose-airflow.yaml up -d
```

Acesso padrao do Airflow Webserver: `http://localhost:8080`.

## Convencoes de Teste

- Unitarios em `test/unit`.
- Integracao em `test/integration`.
- Marcadores disponiveis: `unit`, `integration`.

Exemplos:

```bash
poetry run pytest -m unit
poetry run pytest -m integration
```

## Fluxo Recomendado de Desenvolvimento

1. Criar branch de feature.
2. Rodar testes locais.
3. Validar treino smoke test.
4. Revisar metadados no MLflow.
5. Abrir PR com descricao tecnica e riscos.
