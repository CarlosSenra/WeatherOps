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

    model_type: Literal["lstm", "transformer", "tft", "nbeats"] = Field(
        default="lstm",
        description="Tipo de modelo: 'lstm', 'transformer', 'tft' ou 'nbeats'"
    )
    hidden_size: int = Field(
        default=128,
        description="Dimensão oculta do LSTM / d_model do Transformer / tamanho dos blocos TFT e NBEATS"
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
    # --- TFT-specific ---
    attention_head_size: int = Field(
        default=4,
        description="[TFT] Tamanho de cada cabeça de atenção multi-head"
    )
    hidden_continuous_size: int = Field(
        default=16,
        description="[TFT] Dimensão das projeções para variáveis contínuas"
    )
    # --- NBEATS-specific ---
    num_stacks: int = Field(
        default=30,
        description="[NBEATS] Número de stacks (blocos de expansão)"
    )
    num_blocks: int = Field(
        default=1,
        description="[NBEATS] Número de blocos por stack"
    )
    num_block_layers: int = Field(
        default=4,
        description="[NBEATS] Número de camadas lineares dentro de cada bloco"
    )
    backcast_loss_ratio: float = Field(
        default=0.1,
        description="[NBEATS] Peso do backcast loss relativo ao forecast loss"
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


class GovernanceConfig(BaseModel):
    """Metadados de governança para rastreabilidade de modelos no MLflow."""

    model_name: str = Field(
        default="weather_forecaster",
        description="Nome lógico do modelo para governança"
    )
    model_version: str = Field(
        default="0.1.0",
        description="Versão semântica do modelo"
    )
    model_type: Literal["classification", "regression", "generation"] = Field(
        default="regression",
        description="Tipo de tarefa do modelo (governança)"
    )
    owner: Optional[str] = Field(
        default=None,
        description="Responsável pelo modelo (normalmente e-mail)"
    )
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        default="medium",
        description="Nível de risco do modelo"
    )
    fairness_checked: bool = Field(
        default=False,
        description="Indica se auditoria de viés foi realizada"
    )
    training_data_version: Optional[str] = Field(
        default=None,
        description="Versão/hash dos dados de treino (preenchido automaticamente se ausente)"
    )
    git_sha: Optional[str] = Field(
        default=None,
        description="Commit SHA do código (preenchido automaticamente se ausente)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "weather_lstm_h72",
                "model_version": "1.0.0",
                "model_type": "regression",
                "owner": "owner@example.com",
                "risk_level": "medium",
                "fairness_checked": False,
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
    governance: GovernanceConfig = Field(
        default_factory=GovernanceConfig,
        description="Metadados de governança para tracking no MLflow"
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
                "governance": {
                    "model_name": "weather_lstm_baseline",
                    "model_version": "1.0.0",
                    "model_type": "regression",
                    "risk_level": "medium",
                    "fairness_checked": False,
                },
            }
        }
