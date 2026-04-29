"""PredictionLogger — persiste predições da API e resultados de acurácia em SQLite.

Cada chamada ao endpoint de previsão grava uma entrada em ``prediction_log``.
O ``AccuracyEvaluator`` consulta as entradas pendentes (cujo ground truth já
chegou no Parquet) e grava os resultados por bucket em ``accuracy_log``.

Design
------
- Usa ``sqlite3`` da stdlib (sem dependência extra).
- Mantém uma única conexão persistente com ``check_same_thread=False`` para
  uso seguro em ambiente asyncio.
- Operações bloqueantes são delegadas via ``asyncio.to_thread``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from src.api.schemas.forecast import ForecastResponse

logger = logging.getLogger(__name__)


@dataclass
class BucketResult:
    """Métricas de acurácia para um bucket de horizonte."""

    bucket: str  # "near" | "mid" | "far"
    mae: float
    rmse: float
    mape: float
    n_points: int


@dataclass
class PredictionRecord:
    """Registro de uma predição armazenada no SQLite."""

    id: str
    model_key: str
    model_version: str | None
    group_id: str | None
    reference_date: datetime
    horizon: int
    forecast_json: str


class PredictionLogger:
    """Gerencia o log de predições e resultados de acurácia em SQLite.

    Uso típico
    ----------
    1. Instanciar com o caminho do banco (ou ``":memory:"`` em testes).
    2. Chamar ``await init()`` durante o startup da API.
    3. Chamar ``await log(response, group_id)`` após cada inferência.
    4. O ``AccuracyEvaluator`` chama ``get_pending()`` e ``save_accuracy()``
       periodicamente em background.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Inicialização
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """Cria a conexão e as tabelas (idempotente)."""
        await asyncio.to_thread(self._setup)

    def _setup(self) -> None:
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS prediction_log (
                id             TEXT PRIMARY KEY,
                model_key      TEXT NOT NULL,
                model_version  TEXT,
                group_id       TEXT,
                reference_date TEXT NOT NULL,
                horizon        INTEGER NOT NULL,
                predicted_at   TEXT NOT NULL,
                forecast_json  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS accuracy_log (
                prediction_id  TEXT NOT NULL,
                bucket         TEXT NOT NULL,
                mae            REAL,
                rmse           REAL,
                mape           REAL,
                n_points       INTEGER,
                evaluated_at   TEXT NOT NULL,
                PRIMARY KEY (prediction_id, bucket)
            );
            """
        )
        self._conn.commit()
        logger.info("PredictionLogger inicializado em '%s'", self._db_path)

    # ------------------------------------------------------------------
    # Escrita
    # ------------------------------------------------------------------

    async def log(self, response: ForecastResponse, group_id: str) -> str:
        """Persiste uma predição e retorna o UUID gerado."""
        return await asyncio.to_thread(self._insert_prediction, response, group_id)

    def _insert_prediction(self, response: ForecastResponse, group_id: str) -> str:
        pred_id = str(uuid.uuid4())
        forecast_json = json.dumps(
            [
                {"timestamp": p.timestamp.isoformat(), "temp_ar_c": p.temp_ar_c}
                for p in response.predictions
            ]
        )
        self._conn.execute(  # type: ignore[union-attr]
            """
            INSERT INTO prediction_log
                (id, model_key, model_version, group_id, reference_date,
                 horizon, predicted_at, forecast_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pred_id,
                f"{response.model_type}_{response.horizon}",
                response.model_version,
                group_id,
                response.reference_date.isoformat(),
                response.horizon,
                datetime.now(tz=timezone.utc).isoformat(),
                forecast_json,
            ),
        )
        self._conn.commit()  # type: ignore[union-attr]
        logger.debug(
            "Predição registrada: id=%s model_key=%s_%s",
            pred_id,
            response.model_type,
            response.horizon,
        )
        return pred_id

    # ------------------------------------------------------------------
    # Leitura / escrita de acurácia
    # ------------------------------------------------------------------

    def get_pending(self, now: datetime) -> list[PredictionRecord]:
        """Retorna predições cujo ground truth já deveria estar disponível.

        Uma predição está "pronta para avaliar" quando:
        - ``reference_date + horizon horas < now``
        - Não existe nenhum registro em ``accuracy_log`` para ela.
        """
        if self._conn is None:
            return []
        rows = self._conn.execute(
            """
            SELECT p.*
            FROM prediction_log p
            LEFT JOIN accuracy_log a ON a.prediction_id = p.id
            WHERE a.prediction_id IS NULL
              AND datetime(p.reference_date, '+' || p.horizon || ' hours') < datetime(?)
            GROUP BY p.id
            """,
            (now.isoformat(),),
        ).fetchall()
        return [
            PredictionRecord(
                id=row["id"],
                model_key=row["model_key"],
                model_version=row["model_version"],
                group_id=row["group_id"],
                reference_date=datetime.fromisoformat(row["reference_date"]),
                horizon=row["horizon"],
                forecast_json=row["forecast_json"],
            )
            for row in rows
        ]

    def save_accuracy(self, prediction_id: str, results: list[BucketResult]) -> None:
        """Persiste os resultados de acurácia por bucket."""
        if self._conn is None or not results:
            return
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO accuracy_log
                (prediction_id, bucket, mae, rmse, mape, n_points, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (prediction_id, r.bucket, r.mae, r.rmse, r.mape, r.n_points, now_iso)
                for r in results
            ],
        )
        self._conn.commit()
        logger.debug(
            "Acurácia salva: prediction_id=%s buckets=%s",
            prediction_id,
            [r.bucket for r in results],
        )
