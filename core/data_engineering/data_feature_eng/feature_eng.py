import pandas as pd
import numpy as np
from typing import Optional
from ..models.feature_engineering.inputs import (
    WeatherFeatureEngineerInput,
    FeatureEngineeringConfig,
)
from ..models.feature_engineering.outputs import WeatherFeatureEngineerOutput


class WeatherFeatureEngineer:
    """
    Feature Engineer para dados meteorológicos.
    
    Responsável por criar features engineered para modelos preditivos,
    incluindo features sazonais, lags, médias móveis e tendências.
    """
    
    def __init__(self, config: Optional[FeatureEngineeringConfig] = None):
        """
        Inicializa o Feature Engineer.
        
        Args:
            config: Configuração para feature engineering. Usa padrão se None.
        """
        self.config = config or FeatureEngineeringConfig()
        self.original_num_features = None
        self.input_info = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica as transformações de feature engineering no dataframe.
        
        Args:
            df: DataFrame com dados meteorológicos
            
        Returns:
            DataFrame com features engineered
            
        Raises:
            ValueError: Se o DataFrame não contiver as colunas esperadas
        """
        df_feat = df.copy()
        self.original_num_features = len(df_feat.columns)

        # 1. Configuração de Tempo e Índice
        if 'data_hora' in df_feat.columns:
            df_feat['data_hora'] = pd.to_datetime(df_feat['data_hora'])
            df_feat = df_feat.set_index('data_hora')
        
        df_feat = df_feat.sort_index()
        
        # Valida que as colunas alvo existem
        missing_cols = set(self.config.target_columns) - set(df_feat.columns)
        if missing_cols:
            raise ValueError(f"Colunas faltando: {missing_cols}")
        
        df_feat = df_feat[self.config.target_columns].copy()

        # 2. Sazonalidade Cíclica (Hora do Dia)
        if self.config.use_seasonal_features:
            horas = df_feat.index.hour
            df_feat['hora_sin'] = np.sin(2 * np.pi * horas / 24)
            df_feat['hora_cos'] = np.cos(2 * np.pi * horas / 24)

        # 3. Lags (Memória do Passado)
        for lag in self.config.lag_hours:
            df_feat[f'temp_lag_{lag}h'] = df_feat['temp_ar_c'].shift(lag)

        # 4. Médias Móveis (Suavização de Tendência)
        for window in self.config.moving_avg_windows:
            df_feat[f'temp_ma_{window}h'] = df_feat['temp_ar_c'].rolling(
                window=window, min_periods=1
            ).mean()
            df_feat[f'pressao_ma_{window}h'] = df_feat['pressao_atm_estacao_mb'].rolling(
                window=window, min_periods=1
            ).mean()

        # 5. Taxa de Variação (Diferenciação para prever tempestades/frentes)
        df_feat['pressao_tendencia_1h'] = df_feat['pressao_atm_estacao_mb'].diff(1)
        df_feat['temp_tendencia_1h'] = df_feat['temp_ar_c'].diff(1)

        # Remove linhas com valores nulos se configurado
        if self.config.remove_nan:
            df_feat = df_feat.dropna()

        return df_feat

    def get_output_info(self, df: pd.DataFrame) -> WeatherFeatureEngineerOutput:
        """
        Retorna informações sobre o dataframe transformado.
        
        Args:
            df: DataFrame transformado
            
        Returns:
            WeatherFeatureEngineerOutput com metadados da transformação
        """
        if self.original_num_features is None:
            raise ValueError("Deve executar transform() antes de chamar get_output_info()")
        
        return WeatherFeatureEngineerOutput.from_dataframe(
            df, 
            self.original_num_features
        )

    def validate_input(self, df: pd.DataFrame) -> WeatherFeatureEngineerInput:
        """
        Valida e cria um input model para o dataframe.
        
        Args:
            df: DataFrame a validar
            
        Returns:
            WeatherFeatureEngineerInput com metadados da entrada
        """
        has_data_hora = 'data_hora' in df.columns
        
        input_info = WeatherFeatureEngineerInput(
            dataframe_columns=df.columns.tolist(),
            has_data_hora=has_data_hora,
            config=self.config
        )
        
        self.input_info = input_info
        return input_info