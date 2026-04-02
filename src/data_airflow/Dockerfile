FROM apache/airflow:2.11.2-python3.12

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir poetry==2.3.2

RUN poetry config virtualenvs.create false

WORKDIR /opt/project

COPY --chown=airflow:root pyproject.toml poetry.lock* /opt/project/

COPY --chown=airflow:root core/ /opt/project/core/
COPY --chown=airflow:root src/ /opt/project/src/

RUN poetry install --no-interaction --no-ansi --only main