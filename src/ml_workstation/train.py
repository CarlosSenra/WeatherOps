"""
Entrypoint do ML Workstation para treinamento de modelos de séries temporais.

Suporta quatro arquiteturas via `--config`:
    lstm        — LSTM empilhado (loop manual PyTorch)
    transformer — Transformer encoder (loop manual PyTorch)
    tft         — Temporal Fusion Transformer (pytorch-forecasting + Lightning)
    nbeats      — N-BEATS (pytorch-forecasting + Lightning)

Uso:
    # Com configuração padrão (LSTM, dados em data/spec):
    python -m src.ml_workstation.train

    # Com arquivo de configuração JSON:
    python -m src.ml_workstation.train --config experiments/lstm/lstm_h72_v1.json
    python -m src.ml_workstation.train --config experiments/tft/tft_h72_v1.json
    python -m src.ml_workstation.train --config experiments/nbeats/nbeats_h72_v1.json
"""
import argparse
import json
import logging
import tempfile
from pathlib import Path

import joblib

from src.ml_workstation.config.training_config import TrainingConfig
from src.ml_workstation.data.loader import ParquetDataLoader
from src.ml_workstation.models import build_model, build_pf_model
from src.ml_workstation.tracking.mlflow_tracker import MLflowTracker
from src.ml_workstation.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_PF_MODEL_TYPES = frozenset({"tft", "nbeats"})


def main(config: TrainingConfig) -> None:
    logger.info(
        "Iniciando treinamento — experimento: %s | modelo: %s | device: %s",
        config.experiment_name,
        config.model.model_type,
        config.device,
    )

    if config.model.model_type in _PF_MODEL_TYPES:
        _train_pytorch_forecasting(config)
    else:
        _train_pytorch(config)


# ---------------------------------------------------------------------------
# LSTM / Transformer — loop de treinamento manual PyTorch
# ---------------------------------------------------------------------------

def _train_pytorch(config: TrainingConfig) -> None:
    """Caminho de treino para LSTM e Transformer."""
    data_loader = ParquetDataLoader(config.data)
    train_loader, val_loader, _test_loader, data_output = data_loader.build(
        batch_size=config.batch_size
    )
    target_indices = [config.data.feature_columns.index(c) for c in config.data.target_columns]
    target_mean = data_loader.scaler.mean_[target_indices]
    target_scale = data_loader.scaler.scale_[target_indices]

    logger.info(
        "Dados carregados — treino: %d amostras, val: %d, teste: %d",
        data_output.num_train_samples,
        data_output.num_val_samples,
        data_output.num_test_samples,
    )

    model = build_model(
        n_features=data_output.n_features,
        n_targets=data_output.n_targets,
        horizon=config.data.horizon,
        config=config.model,
    )
    logger.info("Modelo construído: %s", type(model).__name__)

    tracker = MLflowTracker(config)
    tracker.start_run()

    try:
        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            config,
            tracker,
            target_mean=target_mean,
            target_scale=target_scale,
        )
        result = trainer.fit()

        logger.info(
            "Treinamento concluído — best_val_loss=%.4f na época %d/%d",
            result.best_val_loss,
            result.best_epoch,
            result.total_epochs_run,
        )

        tracker.log_governance_metrics(result.final_metrics)

        if Path(result.checkpoint_path).exists():
            tracker.log_artifact(result.checkpoint_path)

        tracker.log_model(model)

        with tempfile.TemporaryDirectory() as tmp:
            scaler_path = Path(tmp) / "scaler.pkl"
            joblib.dump(data_loader.scaler, scaler_path)
            tracker.log_artifact(str(scaler_path))
            logger.info("Scaler salvo como artefato MLflow.")

    finally:
        tracker.end_run()


# ---------------------------------------------------------------------------
# TFT / NBEATS — pytorch-forecasting + PyTorch Lightning
# ---------------------------------------------------------------------------

def _train_pytorch_forecasting(config: TrainingConfig) -> None:
    """Caminho de treino para TFT e NBEATS via pytorch-forecasting."""
    from src.ml_workstation.data.pf_loader import PytorchForecastingDataLoader
    from src.ml_workstation.training.pf_trainer import PytorchForecastingTrainer

    pf_loader = PytorchForecastingDataLoader(config.data, model_type=config.model.model_type)
    train_loader, val_loader, _test_loader, pf_output = pf_loader.build(
        batch_size=config.batch_size
    )

    logger.info(
        "Dados PF carregados — treino: %d amostras, val: %d, teste: %d",
        pf_output.num_train_samples,
        pf_output.num_val_samples,
        pf_output.num_test_samples,
    )

    model = build_pf_model(
        dataset=pf_output.training_dataset,
        config=config.model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    logger.info("Modelo PF construído: %s", type(model).__name__)

    tracker = MLflowTracker(config)
    tracker.start_run()

    try:
        pf_trainer = PytorchForecastingTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            tracker=tracker,
        )
        result = pf_trainer.fit()

        logger.info(
            "Treinamento PF concluído — best_val_loss=%.4f na época %d/%d",
            result.best_val_loss,
            result.best_epoch,
            result.total_epochs_run,
        )

        tracker.log_governance_metrics(result.final_metrics)

        if result.checkpoint_path and Path(result.checkpoint_path).exists():
            tracker.log_artifact(result.checkpoint_path)

        tracker.log_model(model)

    finally:
        tracker.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WeatherOps ML Workstation — treinamento")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Caminho para arquivo JSON de configuração (opcional)",
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = TrainingConfig.model_validate(json.load(f))
        logger.info("Configuração carregada de: %s", args.config)
    else:
        config = TrainingConfig()
        logger.info("Usando configuração padrão")

    main(config)
