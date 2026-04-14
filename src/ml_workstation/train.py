"""
Entrypoint do ML Workstation para treinamento de modelos de séries temporais.

Uso:
    # Com configuração padrão (LSTM, dados em data/spec):
    python -m src.ml_workstation.train

    # Com arquivo de configuração JSON:
    python -m src.ml_workstation.train --config experiments/lstm/lstm_h72_v1.json

    # Exemplo de configuração JSON (experiments/lstm/lstm_h72_v1.json):
    {
        "experiment_name": "weather_forecasting_h72",
        "run_name": "lstm_h72_v1",
        "epochs": 80,
        "batch_size": 64,
        "device": "cpu",
        "data": {
            "parquet_path": "data/spec",
            "target_columns": ["temp_ar_c"],
            "sequence_length": 168,
            "horizon": 72
        },
        "model": {
            "model_type": "lstm",
            "hidden_size": 128,
            "num_layers": 2
        }
    }
"""
import argparse
import json
import logging
import tempfile
from pathlib import Path

import joblib

from src.ml_workstation.config.training_config import TrainingConfig
from src.ml_workstation.data.loader import ParquetDataLoader
from src.ml_workstation.models import build_model
from src.ml_workstation.tracking.mlflow_tracker import MLflowTracker
from src.ml_workstation.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(config: TrainingConfig) -> None:
    logger.info(
        "Iniciando treinamento — experimento: %s | modelo: %s | device: %s",
        config.experiment_name,
        config.model.model_type,
        config.device,
    )

    # 1. Dados
    data_loader = ParquetDataLoader(config.data)
    train_loader, val_loader, test_loader, data_output = data_loader.build(
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

    # 2. Modelo
    model = build_model(
        n_features=data_output.n_features,
        n_targets=data_output.n_targets,
        horizon=config.data.horizon,
        config=config.model,
    )
    logger.info("Modelo construído: %s", type(model).__name__)

    # 3. Tracker
    tracker = MLflowTracker(config)
    tracker.start_run()

    try:
        # 4. Treinamento
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

        # 5. Loga artefatos e modelo final
        if Path(result.checkpoint_path).exists():
            tracker.log_artifact(result.checkpoint_path)

        tracker.log_model(model)

        # 6. Salva e loga o scaler para inferência futura
        with tempfile.TemporaryDirectory() as tmp:
            scaler_path = Path(tmp) / "scaler.pkl"
            joblib.dump(data_loader.scaler, scaler_path)
            tracker.log_artifact(str(scaler_path))
            logger.info("Scaler salvo como artefato MLflow.")

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
