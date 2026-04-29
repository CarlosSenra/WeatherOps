import pandas as pd
from typing import List, Union 
from ..interface.i_data_eng import IDataCleaning
from ..models.data_cleaning.inputs import (
    DataEngInput,
    CsvReadConfig,
    DataConversionConfig,
    ColumnRenameConfig,
)
from ..models.data_cleaning.outputs import DataCleaningProcessOutput


class DataCleaning(IDataCleaning):
    # Configuração padrão para leitura de CSV
    DEFAULT_CSV_CONFIG = CsvReadConfig()
    
    # Configuração padrão para conversão de tipos
    DEFAULT_CONVERSION_CONFIG = DataConversionConfig()
    
    # Mapa padrão de renomeação de colunas
    DEFAULT_COLUMN_RENAME_MAP = {
        'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'precipitacao_total_mm',
        'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': 'pressao_atm_estacao_mb',
        'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)': 'pressao_atm_max_mb',
        'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)': 'pressao_atm_min_mb',
        'RADIACAO GLOBAL (Kj/m²)': 'radiacao_global_kj_m2',
        'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'temp_ar_c',
        'TEMPERATURA DO PONTO DE ORVALHO (°C)': 'temp_ponto_orvalho_c',
        'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'temp_max_c',
        'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'temp_min_c',
        'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)': 'temp_orvalho_max_c',
        'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)': 'temp_orvalho_min_c',
        'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)': 'umidade_rel_max_percent',
        'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)': 'umidade_rel_min_percent',
        'UMIDADE RELATIVA DO AR, HORARIA (%)': 'umidade_rel_ar_percent',
        'VENTO, DIREÇÃO HORARIA (gr) (° (gr))': 'vento_direcao_graus',
        'VENTO, RAJADA MAXIMA (m/s)': 'vento_rajada_ms',
        'VENTO, VELOCIDADE HORARIA (m/s)': 'vento_vel_ms',
        'DATA_HORA': 'data_hora'
    }

    def __init__(
        self,
        csv_paths: Union[str, List[str]],
        csv_config: CsvReadConfig = None,
        conversion_config: DataConversionConfig = None,
        rename_config: ColumnRenameConfig = None,
    ):
        """
        Inicializa o processador de limpeza de dados.
        
        Args:
            csv_paths: Caminho(s) para arquivo(s) CSV
            csv_config: Configuração para leitura de CSV (usa padrão se None)
            conversion_config: Configuração para conversão de tipos (usa padrão se None)
            rename_config: Configuração para renomeação de colunas (usa padrão se None)
        """
        paths = [csv_paths] if isinstance(csv_paths, str) else csv_paths
        self.input = DataEngInput(csv_paths=paths)
        
        # Usar configurações fornecidas ou padrões
        self.csv_config = csv_config or self.DEFAULT_CSV_CONFIG
        self.conversion_config = conversion_config or self.DEFAULT_CONVERSION_CONFIG
        self.rename_config = rename_config or ColumnRenameConfig(rename_map=self.DEFAULT_COLUMN_RENAME_MAP)
        
        self.raw_dataframes = self._load_all_data()
        self.process_data()

    def _default_read_csv(self, path: str) -> pd.DataFrame:
        """Lê arquivo CSV com configurações padrão."""
        skiprows = lambda x: x in range(self.csv_config.skiprows_start)
        _df = pd.read_csv(
            path,
            sep=self.csv_config.sep,
            encoding=self.csv_config.encoding,
            skiprows=skiprows
        )
        return _df.iloc[:, :-1]

    def _load_all_data(self) -> List[pd.DataFrame]:
        """Carrega todos os arquivos CSV."""
        return [self._default_read_csv(path) for path in self.input.csv_paths]

    def _cleaning_str_data_hours_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combina colunas de data e hora em uma coluna DATA_HORA."""
        df['DATA_HORA'] = df['Data'] + df['Hora UTC'].str.strip('UTC').str.strip(' ')
        df['DATA_HORA'] = pd.to_datetime(df['DATA_HORA'], format='%Y/%m/%d%H%M')
        return df.drop(['Data', 'Hora UTC'], axis=1)

    def _convert_str_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converte colunas de string para numérico."""
        col_strings_type = df.select_dtypes(include=self.conversion_config.dtypes_to_convert).columns
        for col in col_strings_type:
            df[col] = pd.to_numeric(
                df[col].str.replace(
                    self.conversion_config.decimal_separator,
                    '.',
                    regex=False
                ),
                errors='coerce'
            )
        return df

    def _rename_all_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renomeia colunas de acordo com o mapa configurado."""
        return df.rename(columns=self.rename_config.rename_map)

    def process_data(self) -> List[pd.DataFrame]:
        """Aplica a transformação em toda a lista de dataframes."""
        self.raw_dataframes = [ self._cleaning_str_data_hours_columns(df) for df in self.raw_dataframes]
        self.raw_dataframes = [self._convert_str_to_numeric(df) for df in self.raw_dataframes]
        self.raw_dataframes = [self._rename_all_columns(df) for df in self.raw_dataframes]
        return self.raw_dataframes

    def concat_csv(self) -> pd.DataFrame:
        """Concatena todos os dataframes em um único dataframe."""
        return pd.concat(self.raw_dataframes)

    def get_process_output(self) -> DataCleaningProcessOutput:
        """Retorna informações sobre o dataframe final processado."""
        final_df = self.concat_csv()
        return DataCleaningProcessOutput.from_dataframe(final_df)
