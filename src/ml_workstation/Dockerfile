# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

# ── Dependências de sistema ──────────────────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependências do projeto (Poetry) ─────────────────────────────────────────
ENV POETRY_VERSION=2.3.2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=1
ENV POETRY_SOLVER_LAZY_WHEEL=true
ENV PATH="${POETRY_HOME}/bin:${PATH}"

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock /app/
RUN poetry --version \
    && poetry check --lock
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    --mount=type=cache,target=/root/.cache/pip \
    poetry install --only main --no-root

# ── Código do projeto ─────────────────────────────────────────────────────────
COPY core/ /app/core/
COPY src/  /app/src/

# ── Diretórios persistidos via volumes ────────────────────────────────────────
# /app/data/spec        → dados de entrada (Parquet spec)
# /app/artifacts        → checkpoints do melhor modelo
# /app/mlruns           → experimentos MLflow (tracking local)
# /app/experiments      → arquivos JSON de configuração de runs
RUN mkdir -p /app/data/spec /app/artifacts /app/mlruns /app/experiments

ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "src.ml_workstation.train"]
# Uso padrão: docker run ... (usa defaults do TrainingConfig)
# Com config:  docker run ... --config /app/experiments/lstm_24h.json
