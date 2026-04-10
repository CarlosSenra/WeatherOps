import pytest
import pandas as pd
import numpy as np

from core.data_engineering.data_feature_eng.feature_eng import WeatherFeatureEngineer
from core.data_engineering.models.feature_engineering.inputs import (
    FeatureEngineeringConfig,
    WeatherFeatureEngineerInput,
)


@pytest.fixture
def sample_cleaned_dataframe():
    """DataFrame após DataCleaning com colunas padrão."""
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    df = pd.DataFrame({
        'data_hora': dates,
        'temp_ar_c': 25.0 + np.sin(np.arange(100) * 2 * np.pi / 24) * 5,
        'umidade_rel_ar_percent': 70.0 + np.random.randn(100),
        'pressao_atm_estacao_mb': 1013.0 + np.random.randn(100) * 0.5,
        'precipitacao_total_mm': np.random.exponential(0.5, 100),
        'vento_vel_ms': np.random.uniform(0, 10, 100),
    })
    return df


@pytest.fixture
def feature_engineering_config():
    """Configuração padrão para feature engineering."""
    return FeatureEngineeringConfig(
        target_columns=['temp_ar_c', 'umidade_rel_ar_percent', 'pressao_atm_estacao_mb', 'precipitacao_total_mm'],
        lag_hours=[1, 24],
        moving_avg_windows=[6, 12],
        use_seasonal_features=True,
        remove_nan=True
    )


@pytest.fixture
def feature_engineering_config_custom():
    """Configuração customizada para feature engineering."""
    return FeatureEngineeringConfig(
        target_columns=['temp_ar_c', 'pressao_atm_estacao_mb'],
        lag_hours=[2, 12, 48],
        moving_avg_windows=[3, 6, 12, 24],
        use_seasonal_features=False,
        remove_nan=False
    )


@pytest.fixture
def feature_engineer():
    """Feature engineer com configuração padrão."""
    return WeatherFeatureEngineer()


@pytest.fixture
def feature_engineer_custom(feature_engineering_config_custom):
    """Feature engineer com configuração customizada."""
    return WeatherFeatureEngineer(config=feature_engineering_config_custom)
