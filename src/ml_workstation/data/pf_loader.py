"""
Carregador de dados para modelos pytorch-forecasting (TFT, NBEATS).

Converte os arquivos Parquet do pipeline spec para o formato
`TimeSeriesDataSet` exigido pelo pytorch-forecasting, respeitando
o mesmo split temporal sem data-leakage usado pelo `ParquetDataLoader`.
"""
import logging
from pathlib import Path

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from torch.utils.data import DataLoader

from src.ml_workstation.config.training_config import DataConfig

logger = logging.getLogger(__name__)

# Features cujos valores futuros são conhecidos em tempo de inferência
# (ciclo horário derivado do timestamp — disponível para qualquer horizonte).
_KNOWN_FUTURE_REALS = frozenset(["hora_sin", "hora_cos"])


class PFDataLoaderOutput:
    """Contêiner de metadados retornado por PytorchForecastingDataLoader.build()."""

    def __init__(
        self,
        training_dataset: TimeSeriesDataSet,
        num_train_samples: int,
        num_val_samples: int,
        num_test_samples: int,
        n_features: int,
        n_targets: int,
        feature_columns: list[str],
        target_columns: list[str],
        known_reals: list[str],
        unknown_reals: list[str],
    ) -> None:
        self.training_dataset = training_dataset
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.n_features = n_features
        self.n_targets = n_targets
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.known_reals = known_reals
        self.unknown_reals = unknown_reals


class PytorchForecastingDataLoader:
    """
    Prepara dados Parquet para modelos pytorch-forecasting.

    O `TimeSeriesDataSet` exige:
    - Coluna `time_idx` (inteiro sequencial por grupo)
    - Coluna `group` (identificador de série; usamos uma constante pois temos
      uma única estação meteorológica)
    - Separação explícita entre `time_varying_known_reals` (conhecidos no futuro,
      ex. hora_sin/hora_cos) e `time_varying_unknown_reals` (demais features)

    O split temporal é feito antes da construção do dataset, garantindo que o
    `TimeSeriesDataSet` de validação/teste use os parâmetros de normalização
    aprendidos no treino (via `from_dataset`).
    """

    GROUP_COL = "group"
    GROUP_VALUE = "station_1"

    def __init__(self, config: DataConfig, model_type: str = "tft") -> None:
        self.config = config
        self.model_type = model_type

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self, batch_size: int = 64
    ) -> tuple[DataLoader, DataLoader, DataLoader, PFDataLoaderOutput]:
        """
        Executa o pipeline completo e retorna (train_loader, val_loader, test_loader, output).

        Args:
            batch_size: Tamanho do batch para todos os DataLoaders.

        Returns:
            Tupla com os três DataLoaders pytorch-forecasting e um PFDataLoaderOutput.
        """
        df = self._read()
        df = self._add_time_idx(df)
        train_df, val_df, test_df = self._split(df)

        known_reals, unknown_reals = self._classify_features()
        logger.info(
            "Features — conhecidas (futuro): %s | desconhecidas: %s",
            known_reals,
            unknown_reals,
        )

        training_dataset = self._build_training_dataset(train_df, known_reals, unknown_reals)
        val_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, val_df, predict=False, stop_randomization=True
        )
        test_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, test_df, predict=False, stop_randomization=True
        )

        train_loader = training_dataset.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0
        )
        val_loader = val_dataset.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0
        )
        test_loader = test_dataset.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0
        )

        output = PFDataLoaderOutput(
            training_dataset=training_dataset,
            num_train_samples=len(training_dataset),
            num_val_samples=len(val_dataset),
            num_test_samples=len(test_dataset),
            n_features=len(self.config.feature_columns),
            n_targets=len(self.config.target_columns),
            feature_columns=self.config.feature_columns,
            target_columns=self.config.target_columns,
            known_reals=known_reals,
            unknown_reals=unknown_reals,
        )

        logger.info(
            "Datasets PF — treino: %d, validação: %d, teste: %d amostras",
            output.num_train_samples,
            output.num_val_samples,
            output.num_test_samples,
        )
        return train_loader, val_loader, test_loader, output

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read(self) -> pd.DataFrame:
        path = Path(self.config.parquet_path)
        if path.is_dir():
            files = sorted(path.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"Nenhum .parquet encontrado em: {path}")
            df = pd.concat([pd.read_parquet(f) for f in files])
            logger.info("Lidos %d arquivos parquet de %s", len(files), path)
        else:
            df = pd.read_parquet(path)
            logger.info("Lido arquivo parquet: %s", path)

        df = df.sort_index()

        all_cols = set(self.config.feature_columns) | set(self.config.target_columns)
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas ausentes no parquet: {missing}")

        needed = list(dict.fromkeys(self.config.feature_columns + self.config.target_columns))
        df = df[needed].dropna()
        logger.info("DataFrame carregado: %d linhas, %d colunas", len(df), len(df.columns))
        return df

    def _split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:].copy()

        logger.info(
            "Split temporal — treino: %d, validação: %d, teste: %d",
            len(train),
            len(val),
            len(test),
        )
        return train, val, test

    def _classify_features(self) -> tuple[list[str], list[str]]:
        """Separa features em conhecidas-no-futuro vs desconhecidas."""
        known = [f for f in self.config.feature_columns if f in _KNOWN_FUTURE_REALS]
        unknown = [f for f in self.config.feature_columns if f not in _KNOWN_FUTURE_REALS]
        return known, unknown

    def _add_time_idx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona `time_idx` global (0..N-1) e coluna de grupo constante na série completa."""
        df = df.copy()
        df[self.GROUP_COL] = self.GROUP_VALUE
        df["time_idx"] = range(len(df))
        return df

    def _build_training_dataset(
        self,
        train_df: pd.DataFrame,
        known_reals: list[str],
        unknown_reals: list[str],
    ) -> TimeSeriesDataSet:
        # O alvo principal é a primeira coluna target; para múltiplos targets
        # seria necessário um MultiTarget — para o dataset atual há apenas um.
        target = self.config.target_columns[0]

        # N-BEATS is univariate: only the target series as input, fixed encoder length,
        # no relative time index.  TFT supports covariates and variable encoder length.
        is_nbeats = self.model_type == "nbeats"
        min_enc = self.config.sequence_length if is_nbeats else self.config.sequence_length // 2
        known_reals_used = [] if is_nbeats else known_reals
        unknown_reals_used = [target] if is_nbeats else unknown_reals

        dataset = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=target,
            group_ids=[self.GROUP_COL],
            min_encoder_length=min_enc,
            max_encoder_length=self.config.sequence_length,
            min_prediction_length=self.config.horizon,
            max_prediction_length=self.config.horizon,
            time_varying_known_reals=known_reals_used,
            time_varying_unknown_reals=unknown_reals_used,
            target_normalizer=GroupNormalizer(
                groups=[self.GROUP_COL], transformation="softplus"
            ),
            add_relative_time_idx=not is_nbeats,
            add_target_scales=not is_nbeats,
            add_encoder_length=not is_nbeats,
        )
        return dataset
