import logging

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE

from src.ml_workstation.config.training_config import ModelConfig

logger = logging.getLogger(__name__)


class WeatherTFT:
    """
    Hiperparâmetros relevantes de `ModelConfig`:
        hidden_size           — tamanho do estado oculto do LSTM interno e das
                                projeções de variável selection network
        attention_head_size   — tamanho de cada cabeça de atenção multi-head
        hidden_continuous_size — dimensão das projeções para variáveis contínuas
        dropout               — taxa de dropout aplicada nas sub-redes internas
        learning_rate         — passado via `TrainingConfig` ao `pl.Trainer`
    """

    @staticmethod
    def from_dataset(
        dataset: TimeSeriesDataSet,
        config: ModelConfig,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> TemporalFusionTransformer:
        """
        Args:
            dataset:        Dataset de treino (define a estrutura de entrada/saída).
            config:         Hiperparâmetros do modelo lidos do `ModelConfig`.
            learning_rate:  Taxa de aprendizado para o otimizador interno.
            weight_decay:   L2 regularization do otimizador interno.

        Returns:
            Instância de `TemporalFusionTransformer` pronta para `pl.Trainer.fit()`.
        """
        model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            hidden_size=config.hidden_size,
            attention_head_size=config.attention_head_size,
            dropout=config.dropout,
            hidden_continuous_size=config.hidden_continuous_size,
            loss=MAE(),
            log_interval=10,
            optimizer="adam",
            reduce_on_plateau_patience=4,
        )
        logger.info(
            "WeatherTFT criado — hidden_size=%d, attention_head_size=%d, "
            "hidden_continuous_size=%d, dropout=%.2f, lr=%.4f",
            config.hidden_size,
            config.attention_head_size,
            config.hidden_continuous_size,
            config.dropout,
            learning_rate,
        )
        logger.info("Parâmetros treináveis: %d", sum(p.numel() for p in model.parameters()))
        return model
