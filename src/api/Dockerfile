# syntax=docker/dockerfile:1.7
# ── WeatherOps ML Serving API ────────────────────────────────────────────────
# Separate from the training image: smaller entrypoint, no GPU deps needed
# for CPU inference, same Poetry base for dependency parity.
FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Poetry ────────────────────────────────────────────────────────────────────
ENV POETRY_VERSION=2.3.2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=1
ENV POETRY_SOLVER_LAZY_WHEEL=true
ENV PATH="${POETRY_HOME}/bin:${PATH}"

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# ── Python dependencies ───────────────────────────────────────────────────────
COPY pyproject.toml poetry.lock /app/
RUN poetry --version && poetry check --lock
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    --mount=type=cache,target=/root/.cache/pip \
    poetry install --only api,agent --no-root

# ── Application code ──────────────────────────────────────────────────────────
COPY src/api/ /app/src/api/
COPY src/api_agent/ /app/src/api_agent/
COPY src/ml_workstation/config/ /app/src/ml_workstation/config/
COPY src/ml_workstation/data/ /app/src/ml_workstation/data/
COPY src/ml_workstation/evaluation/mlflow_helpers.py /app/src/ml_workstation/evaluation/mlflow_helpers.py
COPY src/ml_workstation/models/ /app/src/ml_workstation/models/
COPY src/ml_workstation/promotion/ /app/src/ml_workstation/promotion/

# ── Runtime directories (mounted via docker-compose volumes) ──────────────────
# /app/mlruns        → MLflow tracking store (shared with trainer, read-only)
# /app/data/spec     → Parquet dataset (read-only)
RUN mkdir -p /app/mlruns /app/data/spec

# ── Environment ───────────────────────────────────────────────────────────────
ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV PYTHONPATH=/app
ENV PARQUET_PATH=/app/data/spec
ENV DEVICE=cpu

EXPOSE 8000

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Single worker: model loading is expensive; scale horizontally via replicas.
# For GPU inference set DEVICE=cuda and use the nvidia runtime.
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
