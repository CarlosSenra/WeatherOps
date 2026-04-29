import logging

from pytorch_forecasting import NBeats, TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE

from src.ml_workstation.config.training_config import ModelConfig

logger = logging.getLogger(__name__)


class WeatherNBEATS:
    """
    Hiperparâmetros relevantes de `ModelConfig`:
        hidden_size          — largura das camadas FC dentro de cada bloco
        num_stacks           — número de stacks empilhados
        num_blocks           — número de blocos por stack
        num_block_layers     — profundidade (camadas FC) dentro de cada bloco
        backcast_loss_ratio  — peso relativo do backcast loss no total
        learning_rate        — passado via `TrainingConfig` ao `pl.Trainer`
    """

    @staticmethod
    def from_dataset(
        dataset: TimeSeriesDataSet,
        config: ModelConfig,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> NBeats:
    
        n = config.num_stacks
        model = NBeats.from_dataset(
            dataset,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            backcast_loss_ratio=config.backcast_loss_ratio,
            stack_types=["generic"] * n,
            num_blocks=[config.num_blocks] * n,
            num_block_layers=[config.num_block_layers] * n,
            widths=[config.hidden_size] * n,
            expansion_coefficient_lengths=[32] * n,
            loss=SMAPE(),
            log_interval=10,
            optimizer="adam",
        )
        logger.info(
            "WeatherNBEATS criado — widths=%d, num_stacks=%d, num_blocks=%d, "
            "num_block_layers=%d, backcast_loss_ratio=%.2f, lr=%.4f",
            config.hidden_size,
            config.num_stacks,
            config.num_blocks,
            config.num_block_layers,
            config.backcast_loss_ratio,
            learning_rate,
        )
        logger.info("Parâmetros treináveis: %d", sum(p.numel() for p in model.parameters()))
        return model
