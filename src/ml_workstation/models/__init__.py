from src.ml_workstation.config.training_config import ModelConfig
from src.ml_workstation.models.interface import ITimeSeriesModel
from src.ml_workstation.models.lstm import WeatherLSTM
from src.ml_workstation.models.transformer import WeatherTransformer


def build_model(n_features: int, n_targets: int, horizon: int, config: ModelConfig) -> ITimeSeriesModel:
    """
    Factory que instancia o modelo correto com base em config.model_type.

    Args:
        n_features: Número de features de entrada.
        n_targets: Número de variáveis alvo.
        horizon: Número de passos à frente para prever.
        config: ModelConfig com arquitetura e hiperparâmetros.

    Returns:
        Instância do modelo (WeatherLSTM ou WeatherTransformer).
    """
    if config.model_type == "lstm":
        return WeatherLSTM(n_features, n_targets, horizon, config)
    elif config.model_type == "transformer":
        return WeatherTransformer(n_features, n_targets, horizon, config)
    raise ValueError(f"model_type desconhecido: {config.model_type!r}")


__all__ = ["build_model", "ITimeSeriesModel", "WeatherLSTM", "WeatherTransformer"]
