"""Exportação de artefatos MLflow para disco (ex.: ``src/api/ml_models`` para Docker)."""

from __future__ import annotations

import json
import logging
import hashlib
import shutil
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse, unquote

import mlflow
import mlflow.pytorch
import torch
from pytorch_forecasting import NBeats, TemporalFusionTransformer

from src.ml_workstation.evaluation.mlflow_helpers import (
    build_data_config,
    load_model_with_fallback,
    resolve_tracking_uri,
    workspace_root,
)

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
SERVING_DATA_DIRNAME = "serving_data"
SERVING_DATA_METADATA_NAME = "serving_data_metadata.json"


def _validate_exported_layout(model_dir: Path) -> None:
    missing: list[str] = []
    if not (model_dir / "MLmodel").is_file():
        missing.append("MLmodel")
    if not (model_dir / "data").is_dir():
        missing.append("data/")
    if missing:
        raise RuntimeError(
            f"Export incompleto em '{model_dir}': faltando {', '.join(missing)}."
        )


def _download_model_artifacts(
    *,
    run_id: str,
    effective_model_name: str,
    registry_version: str,
) -> Path:
    def _resolve_candidate_model_dir(path: Path) -> Path | None:
        if (path / "MLmodel").is_file():
            return path
        child_candidates = [p for p in path.iterdir() if p.is_dir()]
        for child in child_candidates:
            if (child / "MLmodel").is_file():
                return child
        return None

    def _materialize_from_run_checkpoint(target_dir: Path) -> Path | None:
        client = mlflow.MlflowClient()
        run = client.get_run(run_id)
        params = run.data.params
        model_type = str(params.get("model.model_type", "")).lower()

        run_artifacts_dir = (
            workspace_root()
            / "src"
            / "ml_workstation"
            / "mlruns"
            / run.info.experiment_id
            / run_id
            / "artifacts"
        )
        ckpt_candidates = sorted(run_artifacts_dir.glob("best_model*.ckpt"))

        if model_type in {"tft", "nbeats"} and ckpt_candidates:
            checkpoint = ckpt_candidates[-1]
            loaders = (
                (TemporalFusionTransformer, "TemporalFusionTransformer"),
                (NBeats, "NBeats"),
            )
            for cls, name in loaders:
                try:
                    model = cls.load_from_checkpoint(
                        str(checkpoint), map_location=torch.device("cpu")
                    )
                    mlflow.pytorch.save_model(model, path=str(target_dir))
                    logger.info(
                        "Artefatos reconstruídos a partir de checkpoint Lightning (%s: %s).",
                        name,
                        checkpoint,
                    )
                    return target_dir
                except Exception:
                    continue

        data_config = build_data_config(params)
        model, source = load_model_with_fallback(
            run=run,
            params=params,
            data_config=data_config,
            device=torch.device("cpu"),
        )
        mlflow.pytorch.save_model(model, path=str(target_dir))
        logger.info(
            "Artefatos reconstruídos a partir do checkpoint do run (%s) em %s",
            source,
            target_dir,
        )
        return target_dir

    def _normalize_artifact_uri(uri: str | None) -> str | None:
        if not uri:
            return None
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            return uri
        host_path = PurePosixPath(unquote(parsed.path))
        if str(host_path).startswith("/app/mlruns"):
            mapped = (
                workspace_root()
                / "src"
                / "ml_workstation"
                / "mlruns"
                / Path(str(host_path).removeprefix("/app/mlruns/"))
            )
            return mapped.resolve().as_uri()
        return uri

    attempts: list[str] = []
    model_uri = f"models:/{effective_model_name}/{registry_version}"
    try:
        downloaded = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
        downloaded_path = Path(downloaded).resolve()
        candidate = _resolve_candidate_model_dir(downloaded_path)
        if candidate is not None:
            logger.info("Artefatos baixados via model URI: %s", model_uri)
            return candidate
        attempts.append(f"{model_uri} -> {downloaded_path}")
    except Exception as exc:
        logger.warning(
            "Falha ao baixar artefatos via model URI (%s). Fallback para run artifact. Erro: %s",
            model_uri,
            exc,
        )

    try:
        client = mlflow.MlflowClient()
        mv = client.get_model_version(name=effective_model_name, version=str(registry_version))
        source_uri = _normalize_artifact_uri(getattr(mv, "source", None))
        if source_uri:
            downloaded = mlflow.artifacts.download_artifacts(artifact_uri=source_uri)
            downloaded_path = Path(downloaded).resolve()
            candidate = _resolve_candidate_model_dir(downloaded_path)
            if candidate is not None:
                logger.info("Artefatos baixados via source URI da model version: %s", source_uri)
                return candidate
            attempts.append(f"{source_uri} -> {downloaded_path}")
    except Exception as exc:
        logger.warning(
            "Falha ao baixar artefatos via source URI da model version. Erro: %s",
            exc,
        )

    downloaded = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
    downloaded_path = Path(downloaded).resolve()
    candidate = _resolve_candidate_model_dir(downloaded_path)
    if candidate is not None:
        logger.info("Artefatos baixados via run URI: runs:/%s/model", run_id)
        return candidate
    attempts.append(f"runs:/{run_id}/model -> {downloaded_path}")

    try:
        rebuilt_dir = Path(tempfile.mkdtemp(prefix=f"rebuilt-{effective_model_name}-"))
        candidate = _materialize_from_run_checkpoint(rebuilt_dir)
        if candidate is not None and (candidate / "MLmodel").is_file():
            return candidate
    except Exception as exc:
        logger.warning(
            "Falha ao reconstruir artefatos do run via checkpoint fallback. Erro: %s",
            exc,
        )

    attempts_text = "; ".join(attempts) if attempts else "sem detalhes"
    raise RuntimeError(
        "Nenhuma origem retornou artefatos válidos com MLmodel. "
        f"Tentativas: {attempts_text}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preferred_slug_from_run(run_id: str) -> str | None:
    try:
        client = mlflow.MlflowClient()
        run = client.get_run(run_id)
    except Exception:
        return None
    for key, value in run.data.params.items():
        if "parquet_path" in key and value:
            return Path(value).name
    return None


def _discover_serving_parquet(run_id: str) -> Path | None:
    data_spec_root = workspace_root() / "data" / "spec"
    if not data_spec_root.exists():
        return None

    preferred_slug = _preferred_slug_from_run(run_id)
    if preferred_slug:
        preferred_dir = data_spec_root / preferred_slug
        preferred_files = sorted(preferred_dir.rglob("*.parquet"))
        if preferred_files:
            return max(preferred_files, key=lambda p: p.stat().st_size)

    default_slug_dir = data_spec_root / "salvador"
    if default_slug_dir.exists():
        default_files = sorted(default_slug_dir.rglob("*.parquet"))
        if default_files:
            return max(default_files, key=lambda p: p.stat().st_size)

    files = sorted(data_spec_root.rglob("*.parquet"))
    return max(files, key=lambda p: p.stat().st_size) if files else None


def _export_serving_data(
    *,
    staging_dir: Path,
    run_id: str,
    effective_model_name: str,
    experiment_name: str,
    registry_version: str,
) -> dict[str, Any] | None:
    source_file = _discover_serving_parquet(run_id)
    if source_file is None:
        logger.warning(
            "Nenhum parquet de serving encontrado em data/spec para %s; prosseguindo sem serving_data.",
            effective_model_name,
        )
        return None

    serving_dir = staging_dir / SERVING_DATA_DIRNAME
    serving_dir.mkdir(parents=True, exist_ok=True)
    target_file = serving_dir / source_file.name
    shutil.copy2(source_file, target_file)

    metadata: dict[str, Any] = {
        "experiment_name": experiment_name,
        "model_name": effective_model_name,
        "registry_version": str(registry_version),
        "run_id": run_id,
        "source_file": str(source_file),
        "exported_file": str(target_file.relative_to(staging_dir)),
        "sha256": _sha256_file(target_file),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    (serving_dir / SERVING_DATA_METADATA_NAME).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


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
    root.mkdir(parents=True, exist_ok=True)
    dest = root / effective_model_name

    downloaded = _download_model_artifacts(
        run_id=run_id,
        effective_model_name=effective_model_name,
        registry_version=registry_version,
    )
    if not downloaded.is_dir():
        raise RuntimeError(f"Artefato de modelo inválido: '{downloaded}' não é diretório.")

    with tempfile.TemporaryDirectory(prefix=f"{effective_model_name}-", dir=root) as tmp_dir:
        staging = Path(tmp_dir) / effective_model_name
        shutil.copytree(downloaded, staging)
        _validate_exported_layout(staging)

        manifest: dict[str, Any] = {
            "model_name": effective_model_name,
            "registry_version": str(registry_version),
            "run_id": run_id,
            "experiment_name": experiment_name,
            "mape": float(candidate_mape) if candidate_mape is not None else None,
            "promoted_at": promoted_at or date.today().isoformat(),
        }
        serving_metadata = _export_serving_data(
            staging_dir=staging,
            run_id=run_id,
            effective_model_name=effective_model_name,
            experiment_name=experiment_name,
            registry_version=str(registry_version),
        )
        if serving_metadata is not None:
            manifest["serving_data"] = {
                "directory": SERVING_DATA_DIRNAME,
                "metadata_file": f"{SERVING_DATA_DIRNAME}/{SERVING_DATA_METADATA_NAME}",
                "source_file": serving_metadata["source_file"],
                "sha256": serving_metadata["sha256"],
            }
        manifest_path = staging / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(staging), str(dest))

    _validate_exported_layout(dest)
    _maybe_snapshot_features(dest)

    logger.info(
        "Modelo exportado para %s (versão Registry=%s, run_id=%s)",
        dest,
        registry_version,
        run_id,
    )
    return dest


def _maybe_snapshot_features(model_dir: Path) -> None:
    """Gera o feature store snapshot a partir do serving_data exportado, se disponível."""
    serving_dir = model_dir / SERVING_DATA_DIRNAME
    metadata_file = serving_dir / SERVING_DATA_METADATA_NAME
    if not metadata_file.is_file():
        return

    try:
        with metadata_file.open(encoding="utf-8") as f:
            serving_meta = json.load(f)
    except Exception as exc:
        logger.warning("Feature store: falha ao ler serving_data_metadata.json: %s", exc)
        return

    parquet_files = list(serving_dir.glob("*.parquet"))
    if not parquet_files:
        return

    parquet_path = max(parquet_files, key=lambda p: p.stat().st_size)
    source_file = serving_meta.get("source_file", "")
    municipio = Path(source_file).parent.name if source_file else parquet_path.parent.parent.name

    try:
        from src.ml_workstation.promotion.feature_store import snapshot_features
        snapshot_features(parquet_path=parquet_path, export_dir=model_dir, municipio=municipio or "desconhecido")
    except Exception as exc:
        logger.warning("Feature store snapshot falhou (não bloqueia export): %s", exc)


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
