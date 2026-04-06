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

# ── Dependências do projeto ───────────────────────────────────────────────────
COPY pyproject.toml /app/
RUN pip install --no-cache-dir \
        "pydantic>=2.12.5,<3.0.0" \
        "plotly>=6.6.0,<7.0.0" \
        "torch>=2.3.0,<3.0.0" \
        "mlflow>=3.0.0,<4.0.0" \
        "scikit-learn>=1.5.0,<2.0.0"
# mlflow 3.0.0 declara pandas<3 no metadata mas funciona com pandas 3.
# --no-deps ignora a restrição do metadata e força a versão correta.
RUN pip install --no-cache-dir --no-deps "pandas>=3.0.1,<4.0.0"

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
