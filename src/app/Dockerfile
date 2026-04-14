# ──────────────────────────────────────────────────────────
# WeatherOps Forecast API
# Build context: raiz do repositório (WeatherOps/)
#
# docker build -f src/app/Dockerfile -t weatherops-api .
# docker run -p 8000:8000 \
#   -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
#   -v $(pwd)/src/ml_workstation/mlruns:/app/src/ml_workstation/mlruns:ro \
#   weatherops-api
# ──────────────────────────────────────────────────────────

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=2.1.2 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# ── Dependências de sistema ─────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Instala Poetry ──────────────────────────────────────────
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# ── Instala dependências via pyproject.toml ─────────────────
# Copia apenas os manifestos primeiro para aproveitar o cache de layers
COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-root

# ── Copia o código fonte ────────────────────────────────────
COPY src/ ./src/
COPY core/ ./core/
COPY __init__.py ./

# ── Usuário não-root ────────────────────────────────────────
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Expose & healthcheck ────────────────────────────────────
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# ── Entrypoint ──────────────────────────────────────────────
CMD ["uvicorn", "src.app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
