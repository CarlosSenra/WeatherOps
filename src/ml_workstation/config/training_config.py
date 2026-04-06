from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Configuração para carregamento e preparação dos dados de séries temporais."""

    parquet_path: str = Field(
        default="data/spec",
        description="Caminho para o arquivo ou diretório Parquet com dados spec"
    )
    feature_columns: List[str] = Field(
        default=[
            "temp_ar_c",
            "umidade_rel_ar_percent",
            "pressao_atm_estacao_mb",
            "precipitacao_total_mm",
            "hora_sin",
            "hora_cos",
            "temp_lag_1h",
            "temp_lag_24h",
            "temp_ma_6h",
            "temp_ma_12h",
            "pressao_ma_6h",
            "pressao_ma_12h",
            "pressao_tendencia_1h",
            "temp_tendencia_1h",
        ],
        description="Colunas usadas como features de entrada do modelo"
    )
    target_columns: List[str] = Field(
        default=["temp_ar_c"],
        description="Colunas alvo para previsão"
    )
    sequence_length: int = Field(
        default=24,
        description="Tamanho da janela de look-back em horas"
    )
    horizon: int = Field(
        default=1,
        description="Número de passos à frente para prever"
    )
    train_ratio: float = Field(
        default=0.8,
        description="Proporção dos dados para treino (split temporal)"
    )
    val_ratio: float = Field(
        default=0.1,
        description="Proporção dos dados para validação (o restante é teste)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "parquet_path": "data/spec",
                "feature_columns": ["temp_ar_c", "umidade_rel_ar_percent", "hora_sin", "hora_cos"],
                "target_columns": ["temp_ar_c"],
                "sequence_length": 24,
                "horizon": 1,
                "train_ratio": 0.8,
                "val_ratio": 0.1,
            }
        }


class ModelConfig(BaseModel):
    """Configuração da arquitetura do modelo de séries temporais."""

    model_type: Literal["lstm", "transformer"] = Field(
        default="lstm",
        description="Tipo de modelo: 'lstm' ou 'transformer'"
    )
    hidden_size: int = Field(
        default=128,
        description="Dimensão oculta do LSTM ou d_model do Transformer"
    )
    num_layers: int = Field(
        default=2,
        description="Número de camadas empilhadas (LSTM layers ou Transformer encoder layers)"
    )
    dropout: float = Field(
        default=0.1,
        description="Taxa de dropout"
    )
    num_heads: int = Field(
        default=4,
        description="Número de cabeças de atenção (somente Transformer; deve dividir hidden_size)"
    )
    ffn_dim: int = Field(
        default=256,
        description="Dimensão da camada feedforward no Transformer"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "lstm",
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.1,
                "num_heads": 4,
                "ffn_dim": 256,
            }
        }


class TrainingConfig(BaseModel):
    """Configuração completa de treinamento, compondo DataConfig e ModelConfig."""

    experiment_name: str = Field(
        default="weather_forecasting",
        description="Nome do experimento no MLflow"
    )
    run_name: Optional[str] = Field(
        default=None,
        description="Nome do run MLflow (gerado automaticamente se None)"
    )
    epochs: int = Field(
        default=50,
        description="Número máximo de épocas de treinamento"
    )
    batch_size: int = Field(
        default=64,
        description="Tamanho do batch"
    )
    learning_rate: float = Field(
        default=1e-3,
        description="Taxa de aprendizado do otimizador Adam"
    )
    weight_decay: float = Field(
        default=1e-4,
        description="Weight decay (L2 regularization)"
    )
    early_stopping_patience: int = Field(
        default=7,
        description="Épocas sem melhora na val_loss antes de parar"
    )
    checkpoint_dir: str = Field(
        default="artifacts/checkpoints",
        description="Diretório para salvar os checkpoints do melhor modelo"
    )
    device: str = Field(
        default="cpu",
        description="Dispositivo de treinamento: 'cpu' ou 'cuda'"
    )
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Configuração dos dados"
    )
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Configuração do modelo"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "experiment_name": "weather_forecasting",
                "run_name": "lstm_baseline",
                "epochs": 50,
                "batch_size": 64,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "early_stopping_patience": 7,
                "checkpoint_dir": "artifacts/checkpoints",
                "device": "cpu",
                "data": {"parquet_path": "data/spec", "target_columns": ["temp_ar_c"]},
                "model": {"model_type": "lstm", "hidden_size": 128},
            }
        }
