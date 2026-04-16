"""Exportação de artefatos MLflow para disco (ex.: ``src/api/ml_models`` para Docker)."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import mlflow

from src.ml_workstation.evaluation.mlflow_helpers import resolve_tracking_uri

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


def export_promoted_model_to_disk(
    *,
    run_id: str,
    effective_model_name: str,
    registry_version: str,
    experiment_name: str,
    tracking_uri: str | None,
    export_dir: str | Path,
    candidate_mape: float | None,
    promoted_at: str | None = None,
) -> Path:
    """Descarrega o artefato ``model`` do run para ``export_dir / effective_model_name``.

    Sobrescreve a pasta de destino se já existir. Grava ``manifest.json`` com metadados
    para a API poder servir sem depender do Registry (opcional).

    Returns:
        Caminho absoluto da pasta do modelo exportado.

    Raises:
        OSError, RuntimeError: falhas de I/O ou download MLflow (propagadas).
    """
    resolved = resolve_tracking_uri(tracking_uri)
    if resolved:
        mlflow.set_tracking_uri(resolved)

    root = Path(export_dir).resolve()
    dest = root / effective_model_name

    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    downloaded = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
    shutil.move(downloaded, dest)

    manifest: dict[str, Any] = {
        "model_name": effective_model_name,
        "registry_version": str(registry_version),
        "run_id": run_id,
        "experiment_name": experiment_name,
        "mape": float(candidate_mape) if candidate_mape is not None else None,
        "promoted_at": promoted_at or date.today().isoformat(),
    }
    manifest_path = dest / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info(
        "Modelo exportado para %s (versão Registry=%s, run_id=%s)",
        dest,
        registry_version,
        run_id,
    )
    return dest


def read_export_manifest(local_model_dir: Path) -> dict[str, Any] | None:
    """Lê ``manifest.json`` de uma pasta exportada, ou ``None`` se inexistente."""
    p = local_model_dir / MANIFEST_NAME
    if not p.is_file():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def is_mlflow_pytorch_model_dir(path: Path) -> bool:
    """Verifica se ``path`` parece um modelo PyTorch flavour MLflow (pasta com ``MLmodel``)."""
    return path.is_dir() and (path / "MLmodel").is_file()
