"""DataService — carrega o dataset Parquet uma vez na inicialização e serve
janelas de contexto para inferência com base em um timestamp de referência.

Design
------
O DataFrame completo (após seleção de colunas e remoção de NaN) é mantido em
memória com duas colunas pré-computadas:

- ``time_idx``: inteiro sequencial global (0, 1, 2, …) exigido pelo
  ``TimeSeriesDataSet`` do pytorch-forecasting.
- ``group``:    constante ``"station_1"`` (chave de grupo da única estação
  meteorológica).

Fatiar por ``reference_date`` preserva o alinhamento correto de ``time_idx``,
garantindo que ``TimeSeriesDataSet.from_dataset()`` receba dados cujo índice
inteiro é consistente com o split de treinamento.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Valor da coluna de grupo — deve corresponder a PytorchForecastingDataLoader.GROUP_VALUE.
_GROUP_VALUE = "station_1"
_GROUP_COL = "group"


class DataService:
    """Repositório Parquet em memória que fornece janelas de contexto para inferência.

    Segurança de threads
    --------------------
    ``_df`` é escrito uma única vez durante o startup (``load``) e é somente
    leitura a partir de então. Nenhum bloqueio é necessário.
    """

    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Inicialização
    # ------------------------------------------------------------------

    async def load(self, parquet_path: str, columns: list[str], model_root: str | None = None) -> None:
        """Carrega e prepara o dataset a partir do disco.

        Delega o I/O bloqueante a uma thread pool para não bloquear a
        corrotina de lifespan.

        Argumentos:
            parquet_path: Caminho para um arquivo Parquet ou diretório
                          contendo arquivos Parquet.
            columns:      União de todas as colunas de features e targets
                          dos modelos registrados. Colunas ausentes no
                          arquivo são ignoradas com aviso em vez de levantar
                          exceção.
        """
        self._df = await asyncio.to_thread(self._read_and_prepare, parquet_path, columns, model_root)
        logger.info(
            "DataService pronto: %d linhas, intervalo de datas %s → %s",
            len(self._df),
            self._df.index.min(),
            self._df.index.max(),
        )

    # ------------------------------------------------------------------
    # API de consulta
    # ------------------------------------------------------------------

    def get_context_window(
        self,
        reference_date: datetime,
        sequence_length: int,
    ) -> pd.DataFrame:
        """Retorna as últimas ``sequence_length`` linhas até ``reference_date``.

        A fatia retornada preserva os valores globais de ``time_idx`` para que
        ``TimeSeriesDataSet.from_dataset(..., predict=True)`` receba um índice
        inteiro sequencialmente consistente.

        Argumentos:
            reference_date: Limite superior do timestamp (inclusive).
            sequence_length: Número de linhas a retornar.

        Retorna:
            Fatia do DataFrame de shape (sequence_length, n_colunas).

        Levanta:
            RuntimeError: Se ``load()`` ainda não foi chamado.
            ValueError:   Se houver menos de ``sequence_length`` linhas
                          disponíveis antes de ``reference_date``.
        """
        self._assert_ready()
        mask = self._df.index <= pd.Timestamp(reference_date)
        available = self._df[mask]

        if len(available) < sequence_length:
            raise ValueError(
                f"Apenas {len(available)} linhas disponíveis antes de {reference_date}; "
                f"são necessárias pelo menos {sequence_length} (sequence_length)."
            )

        return available.iloc[-sequence_length:].copy()

    def get_actual_values(
        self,
        reference_date: datetime,
        start_offset_h: int,
        end_offset_h: int,
        target_column: str = "temp_ar_c",
    ) -> pd.Series | None:
        """Retorna os valores reais observados em uma janela após ``reference_date``.

        Usado pelo ``AccuracyEvaluator`` para confrontar predições com o ground truth
        que chegou ao Parquet após o horizonte ter decorrido.

        Argumentos:
            reference_date:  Instante de referência da predição (exclusive).
            start_offset_h:  Início da janela em horas após ``reference_date`` (inclusive).
            end_offset_h:    Fim da janela em horas após ``reference_date`` (inclusive).
            target_column:   Coluna alvo a retornar (padrão: ``temp_ar_c``).

        Retorna:
            ``pd.Series`` com os valores observados, ou ``None`` se não houver dados
            no intervalo ou a coluna não existir.
        """
        self._assert_ready()
        if target_column not in self._df.columns:  # type: ignore[union-attr]
            logger.warning(
                "DataService.get_actual_values: coluna '%s' não encontrada no DataFrame.",
                target_column,
            )
            return None
        start = pd.Timestamp(reference_date) + pd.Timedelta(hours=start_offset_h)
        end = pd.Timestamp(reference_date) + pd.Timedelta(hours=end_offset_h)
        mask = (self._df.index >= start) & (self._df.index <= end)  # type: ignore[union-attr]
        series = self._df.loc[mask, target_column]  # type: ignore[index]
        return series if len(series) > 0 else None

    def get_training_slice(self, train_ratio: float) -> pd.DataFrame:
        """Retorna o split temporal de treinamento do DataFrame completo.

        Usado pelo ``prepare_metadata`` dos engines na inicialização para
        reconstruir o ``TimeSeriesDataSet`` com seu ``GroupNormalizer`` ajustado.
        O slice espelha exatamente o que ``PytorchForecastingDataLoader._split``
        faz durante o treinamento, garantindo estatísticas do normalizador idênticas.

        Argumentos:
            train_ratio: Fração das linhas usada para treinamento (ex.: 0.8).

        Retorna:
            DataFrame com as linhas 0 … int(n * train_ratio) - 1.
        """
        self._assert_ready()
        n = len(self._df)
        return self._df.iloc[: int(n * train_ratio)].copy()

    @property
    def is_ready(self) -> bool:
        """Retorna ``True`` se o dataset foi carregado com sucesso."""
        return self._df is not None

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Intervalo de datas do dataset ou ``None`` se ainda não carregado."""
        if self._df is None:
            return None
        return self._df.index.min(), self._df.index.max()

    @property
    def row_count(self) -> int:
        """Número de linhas no dataset carregado (0 se ainda não carregado)."""
        return len(self._df) if self._df is not None else 0

    # ------------------------------------------------------------------
    # Métodos auxiliares privados
    # ------------------------------------------------------------------

    def _assert_ready(self) -> None:
        if self._df is None:
            raise RuntimeError(
                "O DataService ainda não foi carregado. "
                "Certifique-se de que DataService.load() seja aguardado durante o startup da aplicação."
            )

    def _candidate_from_model_root(self, model_root: str | None) -> list[Path]:
        if not model_root:
            return []
        root = Path(model_root)
        if not root.exists():
            return []
        candidates = sorted(p for p in root.rglob("*.parquet") if "serving_data" in p.parts)
        # Evita concatenar múltiplos horizontes e introduzir índice temporal duplicado.
        return candidates[:1]

    def _read_parquet_files(self, files: Iterable[Path]) -> pd.DataFrame:
        return pd.concat([pd.read_parquet(f) for f in files])

    def _read_and_prepare(
        self,
        parquet_path: str,
        columns: list[str],
        model_root: str | None = None,
    ) -> pd.DataFrame:
        """Bloqueante: lê o Parquet, seleciona colunas e adiciona time_idx e grupo."""
        path = Path(parquet_path)
        model_root_files = self._candidate_from_model_root(model_root)
        if model_root_files:
            df = self._read_parquet_files(model_root_files)
            logger.info(
                "DataService: usando parquet de serving_data no model_root (%s)",
                model_root_files[0],
            )
        elif path.is_dir():
            files = sorted(path.glob("*.parquet"))
            if not files:
                files = sorted(path.rglob("*.parquet"))
                if len(files) > 1:
                    logger.warning(
                        "DataService: %d parquets recursivos detectados em %s; usando somente %s para evitar índice duplicado.",
                        len(files),
                        path,
                        files[0],
                    )
                    files = files[:1]
            if not files:
                extra = f" e fallback em model_root={model_root}" if model_root else ""
                raise FileNotFoundError(f"Nenhum arquivo .parquet encontrado em: {path}{extra}")
            df = self._read_parquet_files(files)
            logger.info("DataService: lidos %d arquivos parquet (base=%s)", len(files), path)
        else:
            df = pd.read_parquet(path)
            logger.info("DataService: lido arquivo parquet %s", path)

        df = df.sort_index()

        # Seleciona apenas as colunas que estão tanto na lista solicitada quanto no arquivo.
        available_cols = [c for c in columns if c in df.columns]
        missing = [c for c in columns if c not in df.columns]
        if missing:
            logger.warning("DataService: colunas não encontradas no parquet (ignoradas): %s", missing)

        df = df[available_cols].replace([np.inf, -np.inf], pd.NA).dropna()

        # Pré-computa time_idx e grupo para que fatias de inferência estejam sempre prontas.
        df = df.copy()
        df[_GROUP_COL] = _GROUP_VALUE
        df["time_idx"] = range(len(df))

        logger.info(
            "DataService: DataFrame preparado — %d linhas, %d colunas",
            len(df),
            len(df.columns),
        )
        return df
