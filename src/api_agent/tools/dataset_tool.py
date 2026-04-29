"""Tool de resumo de janela do DataService (LangChain @tool com closure)."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from src.api.schemas.forecast import _parse_reference_instant
from src.api.services.data_service import DataService


def _serialize_window(df, max_rows: int = 24) -> list[dict]:
    tail = df.tail(min(max_rows, len(df)))
    records: list[dict] = []
    for ts, row in tail.iterrows():
        rec = {"timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts)}
        for col in row.index:
            v = row[col]
            try:
                if hasattr(v, "item"):
                    v = v.item()
            except Exception:
                pass
            if isinstance(v, float):
                rec[str(col)] = round(v, 4)
            else:
                rec[str(col)] = v
        records.append(rec)
    return records


def make_dataset_window_tool(data_service: DataService):
    @tool
    def summarize_dataset_window(
        reference_date: str,
        sequence_length: int = 48,
        max_rows: int = 24,
    ) -> str:
        """Resume dados observados no Parquet até reference_date (últimas sequence_length horas).

        Args:
            reference_date: Limite superior inclusivo (ISO ou YYYY-MM-DD).
            sequence_length: Número de linhas horárias a olhar para trás.
            max_rows: Máximo de linhas incluídas no JSON de amostra.
        """
        try:
            ref = _parse_reference_instant(reference_date)
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": "data_invalida", "detail": str(e)}, ensure_ascii=False)

        dr = data_service.date_range
        if dr is None:
            return json.dumps({"error": "dataset", "detail": "DataService não carregado."}, ensure_ascii=False)

        try:
            win = data_service.get_context_window(ref, int(sequence_length))
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": "janela", "detail": str(e)}, ensure_ascii=False)

        numeric_cols = [c for c in win.columns if c not in ("group", "time_idx")]
        stats: dict[str, dict] = {}
        for c in numeric_cols:
            s = win[c]
            try:
                stats[c] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "mean": float(s.mean()),
                }
            except Exception:
                continue

        out = {
            "dataset_date_start": dr[0].isoformat(),
            "dataset_date_end": dr[1].isoformat(),
            "row_count_total": data_service.row_count,
            "window_rows": len(win),
            "stats": stats,
            "sample_rows": _serialize_window(win, max_rows=int(max_rows)),
        }
        return json.dumps(out, ensure_ascii=False)

    return summarize_dataset_window
