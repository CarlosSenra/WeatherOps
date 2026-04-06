import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.ml_workstation.config.training_config import DataConfig
from src.ml_workstation.data.dataset import WeatherSequenceDataset

logger = logging.getLogger(__name__)


class DataLoaderOutput(BaseModel):
    """Saída do ParquetDataLoader com os DataLoaders e metadados."""

    num_train_samples: int
    num_val_samples: int
    num_test_samples: int
    n_features: int
    n_targets: int
    feature_columns: list[str]
    target_columns: list[str]

    class Config:
        arbitrary_types_allowed = True


class ParquetDataLoader:
    """
    Carrega dados Parquet da pipeline spec, normaliza e gera DataLoaders PyTorch.

    O split é feito em ordem temporal para evitar data leakage.
    O StandardScaler é ajustado apenas no conjunto de treino.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = StandardScaler()

    def _read(self) -> pd.DataFrame:
        path = Path(self.config.parquet_path)
        if path.is_dir():
            files = sorted(path.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"Nenhum arquivo .parquet encontrado em: {path}")
            df = pd.concat([pd.read_parquet(f) for f in files])
            logger.info("Lidos %d arquivos parquet de %s", len(files), path)
        else:
            df = pd.read_parquet(path)
            logger.info("Lido arquivo parquet: %s", path)

        df = df.sort_index()

        missing_features = [c for c in self.config.feature_columns if c not in df.columns]
        missing_targets = [c for c in self.config.target_columns if c not in df.columns]
        if missing_features or missing_targets:
            raise ValueError(
                f"Colunas ausentes no parquet — features: {missing_features}, targets: {missing_targets}"
            )

        df = df[self.config.feature_columns].dropna()
        logger.info("DataFrame carregado: %d linhas, %d colunas", len(df), len(df.columns))
        return df

    def _split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        logger.info(
            "Split temporal — treino: %d, validação: %d, teste: %d",
            len(train), len(val), len(test),
        )
        return train, val, test

    def _scale(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_arr = self.scaler.fit_transform(train.values)
        val_arr = self.scaler.transform(val.values)
        test_arr = self.scaler.transform(test.values)
        return train_arr, val_arr, test_arr

    def build(self, batch_size: int = 64) -> tuple[DataLoader, DataLoader, DataLoader, DataLoaderOutput]:
        """
        Executa o pipeline completo e retorna (train_loader, val_loader, test_loader, output).

        Args:
            batch_size: Tamanho do batch para os DataLoaders.

        Returns:
            Tupla com os três DataLoaders e o DataLoaderOutput com metadados.
        """
        df = self._read()
        train_df, val_df, test_df = self._split(df)
        train_arr, val_arr, test_arr = self._scale(train_df, val_df, test_df)

        target_indices = [self.config.feature_columns.index(c) for c in self.config.target_columns]

        train_ds = WeatherSequenceDataset(train_arr, self.config, target_indices)
        val_ds = WeatherSequenceDataset(val_arr, self.config, target_indices)
        test_ds = WeatherSequenceDataset(test_arr, self.config, target_indices)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        output = DataLoaderOutput(
            num_train_samples=len(train_ds),
            num_val_samples=len(val_ds),
            num_test_samples=len(test_ds),
            n_features=len(self.config.feature_columns),
            n_targets=len(self.config.target_columns),
            feature_columns=self.config.feature_columns,
            target_columns=self.config.target_columns,
        )

        return train_loader, val_loader, test_loader, output
