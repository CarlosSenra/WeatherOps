from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import plotly.graph_objects as go
import torch

from src.ml_workstation.data.loader import ParquetDataLoader
from src.ml_workstation.evaluation.mlflow_helpers import (
    build_data_config,
    load_model_with_fallback,
    parse_mlflow_param,
    resolve_tracking_uri,
)

logger = logging.getLogger(__name__)


def inverse_targets(
    values: np.ndarray,
    *,
    feature_columns: list[str],
    target_columns: list[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> np.ndarray:
    """Desfaz normalizacao somente para os alvos, usando estatisticas do scaler global."""
    target_indices = [feature_columns.index(col) for col in target_columns]
    means = scaler_mean[target_indices].reshape(1, 1, -1)
    scales = scaler_scale[target_indices].reshape(1, 1, -1)
    return values * scales + means


def run_inference(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x)
            preds.append(pred.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())

    if not preds:
        raise RuntimeError("Test loader vazio: nao ha amostras para avaliar")

    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def build_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    run_id: str,
    target_name: str,
    horizon_step: int,
) -> go.Figure:
    x = np.arange(len(y_true))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_true,
            mode="lines",
            name="Real",
            line={"width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_pred,
            mode="lines",
            name="Predito",
            line={"width": 2},
        )
    )

    fig.update_layout(
        title=(
            f"Real vs Predito | run_id={run_id[:8]}... | "
            f"target={target_name} | horizonte_t+{horizon_step + 1}"
        ),
        xaxis_title="Amostra no conjunto de teste",
        yaxis_title=target_name,
        template="plotly_white",
        legend={"orientation": "h", "y": 1.05, "x": 0.0},
    )
    return fig


def evaluate_run(
    *,
    run_id: str,
    target_index: int,
    horizon_step: int,
    max_points: int,
    device_name: str,
    output_dir: str,
    tracking_uri: str | None,
) -> Path:
    resolved_tracking_uri = resolve_tracking_uri(tracking_uri)
    if resolved_tracking_uri:
        mlflow.set_tracking_uri(resolved_tracking_uri)
        logger.info("MLflow tracking_uri definido para: %s", resolved_tracking_uri)

    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params

    data_config = build_data_config(params)
    batch_size = int(parse_mlflow_param(params.get("batch_size", "64")))

    device = torch.device(device_name)
    model, model_uri = load_model_with_fallback(
        run=run,
        params=params,
        data_config=data_config,
        device=device,
    )

    data_loader = ParquetDataLoader(data_config)
    _, _, test_loader, data_output = data_loader.build(batch_size=batch_size)

    y_pred, y_true = run_inference(model=model, test_loader=test_loader, device=device)

    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape inconsistente entre predito {y_pred.shape} e real {y_true.shape}")

    if y_pred.shape[1] != data_config.horizon:
        raise ValueError(
            f"Horizonte do modelo ({y_pred.shape[1]}) difere do configurado ({data_config.horizon})"
        )

    if y_pred.shape[2] != data_output.n_targets:
        raise ValueError(
            f"Numero de targets do modelo ({y_pred.shape[2]}) difere dos dados ({data_output.n_targets})"
        )

    if not 0 <= target_index < data_output.n_targets:
        raise IndexError(
            f"target_index fora do range: recebido {target_index}, permitido [0, {data_output.n_targets - 1}]"
        )

    if not 0 <= horizon_step < data_config.horizon:
        raise IndexError(
            f"horizon_step fora do range: recebido {horizon_step}, permitido [0, {data_config.horizon - 1}]"
        )

    y_pred = inverse_targets(
        y_pred,
        feature_columns=data_config.feature_columns,
        target_columns=data_config.target_columns,
        scaler_mean=data_loader.scaler.mean_,
        scaler_scale=data_loader.scaler.scale_,
    )
    y_true = inverse_targets(
        y_true,
        feature_columns=data_config.feature_columns,
        target_columns=data_config.target_columns,
        scaler_mean=data_loader.scaler.mean_,
        scaler_scale=data_loader.scaler.scale_,
    )

    y_pred_series = y_pred[:, horizon_step, target_index]
    y_true_series = y_true[:, horizon_step, target_index]

    if max_points > 0 and len(y_true_series) > max_points:
        stride = int(np.ceil(len(y_true_series) / max_points))
        y_true_series = y_true_series[::stride]
        y_pred_series = y_pred_series[::stride]
        logger.info("Aplicado downsampling no grafico com stride=%d", stride)

    target_name = data_config.target_columns[target_index]
    fig = build_figure(
        y_true=y_true_series,
        y_pred=y_pred_series,
        run_id=run_id,
        target_name=target_name,
        horizon_step=horizon_step,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{run_id}_target-{target_name}_h-{horizon_step + 1}.html"
    fig.write_html(out_file, include_plotlyjs="cdn")

    summary = {
        "run_id": run_id,
        "experiment_id": run.info.experiment_id,
        "n_test_samples": int(len(y_true_series)),
        "target_name": target_name,
        "horizon_step": int(horizon_step),
        "model_uri": model_uri,
        "output_file": str(out_file),
        "tracking_uri": resolved_tracking_uri,
    }
    logger.info("Avaliacao concluida: %s", json.dumps(summary, ensure_ascii=True))
    return out_file
