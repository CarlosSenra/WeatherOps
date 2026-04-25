"""AccuracyEvaluator — worker assíncrono de avaliação de qualidade dos modelos.

Fluxo
-----
1. A cada ``interval_seconds``, consulta o ``PredictionLogger`` por predições
   cujo horizonte já transcorreu e que ainda não foram avaliadas.
2. Para cada predição, divide o horizonte em buckets temporais:
   - ``near`` — horas 1 a 24   (curto prazo)
   - ``mid``  — horas 25 a 72  (médio prazo)
   - ``far``  — horas 73+      (longo prazo, só relevante para h168/h336)
3. Busca os valores reais via ``DataService.get_actual_values``.
4. Calcula MAE, RMSE e MAPE por bucket.
5. Persiste em ``accuracy_log`` e atualiza os Prometheus Gauges.

Funções puras
-------------
``compute_metrics`` e ``slice_forecast`` são funções livres e totalmente
testáveis sem dependências externas.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd

from src.api.metrics import model_mae, model_mape, model_rmse
from src.api.services.prediction_logger import BucketResult, PredictionLogger

if TYPE_CHECKING:
    from src.api.services.data_service import DataService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Buckets de horizonte: (nome, hora_inicio_inclusiva, hora_fim_inclusiva)
# ---------------------------------------------------------------------------
BUCKETS: list[tuple[str, int, int]] = [
    ("near", 1, 24),
    ("mid", 25, 72),
    ("far", 73, 9_999),
]


# ---------------------------------------------------------------------------
# Funções puras de cálculo
# ---------------------------------------------------------------------------


@dataclass
class MetricsResult:
    """Resultado de MAE, RMSE e MAPE para um conjunto de pares previsto/real."""

    mae: float
    rmse: float
    mape: float
    n_points: int


def compute_metrics(predicted: list[float], actual: pd.Series) -> MetricsResult:
    """Calcula MAE, RMSE e MAPE entre valores previstos e reais.

    Argumentos:
        predicted: Lista de valores previstos (ordem temporal).
        actual:    Series pandas com os valores reais alinhados temporalmente.

    Retorna:
        ``MetricsResult`` com as métricas calculadas. MAPE é ``nan`` quando
        todos os valores reais são zero.
    """
    n = min(len(predicted), len(actual))
    if n == 0:
        return MetricsResult(mae=float("nan"), rmse=float("nan"), mape=float("nan"), n_points=0)

    pred_arr = predicted[:n]
    actual_arr = actual.iloc[:n].to_numpy(dtype=float)

    abs_errors = [abs(float(p) - float(a)) for p, a in zip(pred_arr, actual_arr)]
    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum((float(p) - float(a)) ** 2 for p, a in zip(pred_arr, actual_arr)) / n)

    pct_errors = [
        abs(float(p) - float(a)) / abs(float(a))
        for p, a in zip(pred_arr, actual_arr)
        if float(a) != 0.0
    ]
    mape = (sum(pct_errors) / len(pct_errors)) if pct_errors else float("nan")

    return MetricsResult(mae=mae, rmse=rmse, mape=mape, n_points=n)


def slice_forecast(forecast_json: str, start_offset_h: int, end_offset_h: int) -> list[float]:
    """Extrai valores previstos para o intervalo de horas especificado.

    As horas são indexadas a partir de 1 (hora 1 = primeira hora após
    ``reference_date``). O intervalo é inclusivo em ambas as pontas.

    Argumentos:
        forecast_json:  JSON serializado de ``[{timestamp, temp_ar_c}, ...]``.
        start_offset_h: Hora de início (inclusiva, mínimo 1).
        end_offset_h:   Hora de fim (inclusiva).

    Retorna:
        Lista de ``float`` com os valores ``temp_ar_c`` do intervalo.
    """
    points = json.loads(forecast_json)
    start_idx = max(0, start_offset_h - 1)
    end_idx = end_offset_h  # slice é exclusivo no fim, então não subtrai
    return [float(p["temp_ar_c"]) for p in points[start_idx:end_idx]]


# ---------------------------------------------------------------------------
# Worker assíncrono
# ---------------------------------------------------------------------------


class AccuracyEvaluator:
    """Worker assíncrono que avalia a acurácia das predições em background.

    Instanciar e chamar ``asyncio.create_task(evaluator.start())`` no lifespan.
    """

    def __init__(
        self,
        logger_svc: PredictionLogger,
        data_service: DataService,
        interval_seconds: int = 3600,
    ) -> None:
        self._logger_svc = logger_svc
        self._data_service = data_service
        self._interval_seconds = interval_seconds

    async def run_once(self) -> None:
        """Avalia todas as predições pendentes disponíveis no momento."""
        pending = await asyncio.to_thread(
            self._logger_svc.get_pending, datetime.now(tz=timezone.utc).replace(tzinfo=None)
        )
        if not pending:
            logger.debug("AccuracyEvaluator: nenhuma predição pendente.")
            return

        logger.info(
            "AccuracyEvaluator: %d predição(ões) pendente(s) de avaliação.", len(pending)
        )

        for pred in pending:
            bucket_results: list[BucketResult] = []

            for bucket_name, start_h, end_h in BUCKETS:
                effective_end = min(end_h, pred.horizon)
                if start_h > pred.horizon:
                    continue

                actual = self._data_service.get_actual_values(
                    reference_date=pred.reference_date,
                    start_offset_h=start_h,
                    end_offset_h=effective_end,
                )
                if actual is None or len(actual) == 0:
                    logger.debug(
                        "Sem dados reais: model=%s bucket=%s reference_date=%s",
                        pred.model_key,
                        bucket_name,
                        pred.reference_date,
                    )
                    continue

                predicted_vals = slice_forecast(pred.forecast_json, start_h, effective_end)
                if not predicted_vals:
                    continue

                metrics = compute_metrics(predicted_vals, actual)

                bucket_results.append(
                    BucketResult(
                        bucket=bucket_name,
                        mae=metrics.mae,
                        rmse=metrics.rmse,
                        mape=metrics.mape,
                        n_points=metrics.n_points,
                    )
                )

                model_mae.labels(model_key=pred.model_key, bucket=bucket_name).set(
                    metrics.mae
                )
                model_rmse.labels(model_key=pred.model_key, bucket=bucket_name).set(
                    metrics.rmse
                )
                if not math.isnan(metrics.mape):
                    model_mape.labels(
                        model_key=pred.model_key, bucket=bucket_name
                    ).set(metrics.mape)

            if bucket_results:
                await asyncio.to_thread(
                    self._logger_svc.save_accuracy, pred.id, bucket_results
                )
                logger.info(
                    "Acurácia calculada: model=%s buckets=%s",
                    pred.model_key,
                    [r.bucket for r in bucket_results],
                )

    async def start(self) -> None:
        """Loop infinito que chama ``run_once`` a cada ``interval_seconds``."""
        logger.info(
            "AccuracyEvaluator iniciado (intervalo=%ds).", self._interval_seconds
        )
        while True:
            try:
                await self.run_once()
            except Exception as exc:
                logger.error(
                    "Erro em AccuracyEvaluator.run_once: %s", exc, exc_info=True
                )
            await asyncio.sleep(self._interval_seconds)
