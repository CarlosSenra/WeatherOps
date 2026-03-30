import pandas as pd
from typing import List, Union 
from .interface.i_data_eng import IDataCleaning
from .inputs.csv_path_input import DataEngInput

class DataCleaning(IDataCleaning):
    def __init__(self, csv_paths:Union[str, List[str]]):

        paths = [csv_paths] if isinstance(csv_paths, str) else csv_paths
        self.input = DataEngInput(csv_paths=paths)

        self.raw_dataframes = self._load_all_data()
        self.process_data()

    def _default_read_csv(self,path:str) -> pd.DataFrame:
        _df = pd.read_csv(path, sep=';', encoding='latin-1', skiprows=lambda x: x in range(8))
        return _df.iloc[:,:-1]

    def _load_all_data(self) -> List[pd.DataFrame]:
        return [self._default_read_csv(path) for path in self.input.csv_paths]

    def _cleaning_str_data_hours_columns(self, df:pd.DataFrame) -> pd.DataFrame:
        df['DATA_HORA'] = df['Data'] + df['Hora UTC'].str.strip('UTC').str.strip(' ')
        df['DATA_HORA'] = pd.to_datetime(df['DATA_HORA'], format='%Y/%m/%d%H%M')
        return df.drop(['Data', 'Hora UTC'], axis=1)
    

    def _convert_str_to_numeric(self, df:pd.DataFrame, dtype:List[str]=['object', 'string']) -> pd.DataFrame:
        col_strings_type = df.select_dtypes(include=dtype).columns
        for col in col_strings_type:
            df[col] = pd.to_numeric(df[col].str.replace(',', '.', regex=False), errors='coerce')
        return df
    
    def _rename_all_columns(self, df:pd.DataFrame) -> pd.DataFrame:
        self.rename_map = {
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

        return df.rename(columns=self.rename_map)

    def process_data(self) -> List[pd.DataFrame]:
        """Aplica a transformação em toda a lista de dataframes."""
        self.raw_dataframes = [self._cleaning_str_data_hours_columns(df) for df in self.raw_dataframes]
        self.raw_dataframes = [self._convert_str_to_numeric(df) for df in self.raw_dataframes]
        self.raw_dataframes = [self._rename_all_columns(df) for df in self.raw_dataframes]

    def concat_csv(self) -> pd.DataFrame:
        return pd.concat(self.raw_dataframes)
            
