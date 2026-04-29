"""Snapshot de features para o feature store local.

Gera perfis mensais e extremos sazonais a partir do Parquet de serving, gravando
JSONs em ``<export_dir>/feature_store/``. Esses arquivos alimentam a base vetorial
de RAG construída por ``src.api_agent.rag.knowledge_builder``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_STORE_DIRNAME = "feature_store"

_PROFILE_COLS = [
    "temp_ar_c",
    "umidade_rel_ar_percent",
    "pressao_atm_estacao_mb",
    "precipitacao_total_mm",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_datetime_series(df: pd.DataFrame) -> pd.Series:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.to_series()
    datetime_col_candidates = [
        "data_hora", "date_hour", "datetime", "timestamp", "time", "date", "data"
    ]
    for col in datetime_col_candidates:
        if col in df.columns:
            return pd.to_datetime(df[col])
    for col in df.columns:
        try:
            return pd.to_datetime(df[col])
        except Exception:
            continue
    raise ValueError(
        f"Não foi possível identificar coluna de datetime no Parquet. "
        f"Colunas disponíveis: {list(df.columns)}"
    )


def snapshot_features(
    parquet_path: Path,
    export_dir: Path,
    municipio: str | None = None,
) -> dict[str, Any]:
    """Gera snapshot de features do município e salva em ``export_dir/feature_store/``.

    Args:
        parquet_path: Caminho para o arquivo Parquet de serving.
        export_dir:   Diretório raiz do modelo exportado (ex.: ``src/api/ml_models/<model>``).
        municipio:    Nome do município; inferido do diretório pai do parquet se ``None``.

    Returns:
        Dicionário de metadados do snapshot (municipio, date_range, row_count, etc.).
    """
    if municipio is None:
        municipio = parquet_path.parent.name

    df = pd.read_parquet(parquet_path)
    dt_series = _extract_datetime_series(df)
    df = df.copy()
    df["_month"] = dt_series.values if isinstance(df.index, pd.DatetimeIndex) else dt_series.values

    # Usa o índice datetime diretamente para extrair mês quando é DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        df["_month"] = df.index.month
    else:
        df["_month"] = pd.to_datetime(df["_month"]).dt.month

    available_cols = [c for c in _PROFILE_COLS if c in df.columns]
    if not available_cols:
        logger.warning(
            "Feature store: nenhuma coluna de perfil encontrada em %s (esperado: %s).",
            parquet_path,
            _PROFILE_COLS,
        )

    monthly_profiles: dict[str, Any] = {}
    seasonal_extremes: dict[str, Any] = {}

    for month in range(1, 13):
        mask = df["_month"] == month
        if not mask.any():
            continue

        month_df = df[mask]
        month_key = str(month)

        if available_cols:
            agg = month_df[available_cols].agg(["mean", "std", "min", "max"])
            monthly_profiles[month_key] = {
                col: {
                    stat: round(float(agg.loc[stat, col]), 2)
                    for stat in ["mean", "std", "min", "max"]
                }
                for col in available_cols
            }
            seasonal_extremes[month_key] = {
                col: {
                    "p5": round(float(month_df[col].quantile(0.05)), 2),
                    "p95": round(float(month_df[col].quantile(0.95)), 2),
                }
                for col in available_cols
            }

    if isinstance(df.index, pd.DatetimeIndex):
        date_start = df.index.min().isoformat()
        date_end = df.index.max().isoformat()
    else:
        date_start = str(dt_series.min())
        date_end = str(dt_series.max())

    metadata: dict[str, Any] = {
        "municipio": municipio,
        "parquet_path": str(parquet_path),
        "sha256": _sha256_file(parquet_path),
        "date_range": {"start": date_start, "end": date_end},
        "row_count": len(df),
        "profile_columns": available_cols,
        "snapshot_at": datetime.now(timezone.utc).isoformat(),
    }

    feature_store_dir = export_dir / FEATURE_STORE_DIRNAME
    feature_store_dir.mkdir(parents=True, exist_ok=True)

    (feature_store_dir / "monthly_profiles.json").write_text(
        json.dumps(monthly_profiles, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (feature_store_dir / "seasonal_extremes.json").write_text(
        json.dumps(seasonal_extremes, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (feature_store_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        "Feature store snapshot criado para '%s' em %s (%d linhas, %d meses).",
        municipio,
        feature_store_dir,
        len(df),
        len(monthly_profiles),
    )
    return metadata
