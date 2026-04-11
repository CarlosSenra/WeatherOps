from __future__ import annotations

import pandas as pd
import pytest

from core.data_analynitcs.graphs_maker.graphs import AnaliseGraficaAnualEDA, AnaliseGraficaEDA


@pytest.fixture
def weather_df() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 12:00:00",
            "2024-06-10 06:00:00",
            "2024-06-10 18:00:00",
            "2025-01-02 00:00:00",
            "2025-01-02 12:00:00",
            "2025-07-20 06:00:00",
            "2025-07-20 18:00:00",
        ]
    )

    return pd.DataFrame(
        {
            "data_hora": timestamps,
            "temp_max_c": [30, 31, 28, 27, 33, 34, 29, 28],
            "temp_ar_c": [25, 26, 22, 21, 27, 28, 23, 22],
            "temp_min_c": [20, 21, 18, 17, 22, 23, 19, 18],
            "precipitacao_total_mm": [0, 2, 10, 4, 1, 0, 12, 6],
            "umidade_rel_ar_percent": [60, 58, 70, 75, 55, 53, 72, 74],
            "pressao_atm_estacao_mb": [1010, 1011, 1008, 1007, 1012, 1013, 1009, 1008],
        }
    )


def test_eda_plots_and_missing_data_summary(weather_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("plotly.basedatatypes.BaseFigure.show", lambda self: None)

    eda = AnaliseGraficaEDA(weather_df.assign(temp_ar_c=[25, None, 22, 21, 27, 28, 23, 22]))

    eda.plot_serie_temporal("temp_ar_c")
    eda.plot_temperaturas_conjuntas()
    eda.plot_distribuicao("temp_ar_c")
    eda.plot_matriz_correlacao()
    missing = eda.resumo_dados_ausentes()

    assert "temp_ar_c" in missing.index


def test_analise_anual_executes_all_visual_methods(weather_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("plotly.basedatatypes.BaseFigure.show", lambda self: None)

    annual = AnaliseGraficaAnualEDA(weather_df)

    annual.comparar_distribuicao_anual("temp_ar_c")
    annual.evolucao_mensal_por_ano("temp_ar_c", agregacao="mean")
    annual.evolucao_mensal_por_ano("precipitacao_total_mm", agregacao="sum")
    annual.acumulado_anual_precipitacao()
    annual.perfil_horario_por_ano("temp_ar_c", agregacao="mean")
    annual.distribuicao_horaria_por_ano("temp_ar_c")
    annual.evolucao_diaria_por_ano("temp_ar_c", agregacao="mean")
    annual.heatmap_hora_mes_por_ano("temp_ar_c", agregacao="mean")
    annual.contar_dias_extremos(temperatura_limite=30, tipo="max")
    annual.contar_dias_extremos(temperatura_limite=20, tipo="min")


def test_analise_anual_validates_required_columns(weather_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        AnaliseGraficaAnualEDA(weather_df.drop(columns=["data_hora"]))


def test_evolucao_mensal_por_ano_rejects_invalid_aggregation(weather_df: pd.DataFrame) -> None:
    annual = AnaliseGraficaAnualEDA(weather_df)

    with pytest.raises(ValueError):
        annual.evolucao_mensal_por_ano("temp_ar_c", agregacao="median")
