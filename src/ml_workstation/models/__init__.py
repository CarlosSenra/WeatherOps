from pytorch_forecasting import TimeSeriesDataSet

from src.ml_workstation.config.training_config import ModelConfig
from src.ml_workstation.models.interface import ITimeSeriesModel
from src.ml_workstation.models.lstm import WeatherLSTM
from src.ml_workstation.models.nbeats import WeatherNBEATS
from src.ml_workstation.models.tft import WeatherTFT
from src.ml_workstation.models.transformer import WeatherTransformer


def build_model(n_features: int, n_targets: int, horizon: int, config: ModelConfig) -> ITimeSeriesModel:
    """
    Factory para modelos PyTorch puros (LSTM e Transformer).

    Args:
        n_features: Número de features de entrada.
        n_targets: Número de variáveis alvo.
        horizon: Número de passos à frente para prever.
        config: ModelConfig com arquitetura e hiperparâmetros.

    Returns:
        Instância do modelo que implementa `ITimeSeriesModel`.
    """
    if config.model_type == "lstm":
        return WeatherLSTM(n_features, n_targets, horizon, config)
    if config.model_type == "transformer":
        return WeatherTransformer(n_features, n_targets, horizon, config)
    raise ValueError(
        f"build_model suporta apenas 'lstm' e 'transformer'; "
        f"para 'tft'/'nbeats' use build_pf_model(). Recebido: {config.model_type!r}"
    )


def build_pf_model(
    dataset: TimeSeriesDataSet,
    config: ModelConfig,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
):
    """
    Factory para modelos pytorch-forecasting (TFT e NBEATS).

    Esses modelos são `LightningModule` e requerem um `TimeSeriesDataSet`
    para configurar automaticamente a estrutura de entrada/saída.

    Args:
        dataset:       Dataset de treino construído pelo `PytorchForecastingDataLoader`.
        config:        ModelConfig com hiperparâmetros específicos do modelo.
        learning_rate: Taxa de aprendizado para o otimizador interno do modelo.
        weight_decay:  L2 regularization para o otimizador interno.

    Returns:
        Instância de `TemporalFusionTransformer` ou `NBeats` (ambos `LightningModule`).
    """
    if config.model_type == "tft":
        return WeatherTFT.from_dataset(dataset, config, learning_rate, weight_decay)
    if config.model_type == "nbeats":
        return WeatherNBEATS.from_dataset(dataset, config, learning_rate, weight_decay)
    raise ValueError(
        f"build_pf_model suporta apenas 'tft' e 'nbeats'. Recebido: {config.model_type!r}"
    )


__all__ = [
    "build_model",
    "build_pf_model",
    "ITimeSeriesModel",
    "WeatherLSTM",
    "WeatherTransformer",
    "WeatherTFT",
    "WeatherNBEATS",
]
