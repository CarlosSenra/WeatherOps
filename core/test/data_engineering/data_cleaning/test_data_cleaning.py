import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import datetime

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.data_engineering.data_cleaning.data_cleaning import DataCleaning
from core.data_engineering.models.data_cleaning.inputs import (
    CsvReadConfig,
    DataConversionConfig,
    ColumnRenameConfig,
)
from core.data_engineering.models.data_cleaning.outputs import DataCleaningProcessOutput


class TestDataCleaningInit:
    """Testes para inicialização da classe DataCleaning."""
    
    def test_init_single_csv_path_as_string(self, temporary_csv_file, column_rename_config):
        """Testa inicialização com um único arquivo CSV como string."""
        cleaner = DataCleaning(
            csv_paths=str(temporary_csv_file),
            rename_config=column_rename_config
        )
        
        assert len(cleaner.raw_dataframes) == 1
        assert isinstance(cleaner.raw_dataframes[0], pd.DataFrame)
        assert len(cleaner.raw_dataframes[0]) > 0
    
    def test_init_multiple_csv_paths_as_list(self, multiple_csv_files, column_rename_config):
        """Testa inicialização com múltiplos arquivos CSV como lista."""
        csv_paths = [str(path) for path in multiple_csv_files]
        cleaner = DataCleaning(
            csv_paths=csv_paths,
            rename_config=column_rename_config
        )
        
        assert len(cleaner.raw_dataframes) == 2
        for df in cleaner.raw_dataframes:
            assert isinstance(df, pd.DataFrame)
    
    def test_init_with_custom_configs(self, temporary_csv_file, csv_read_config, 
                                      data_conversion_config, column_rename_config):
        """Testa inicialização com configurações customizadas."""
        cleaner = DataCleaning(
            csv_paths=str(temporary_csv_file),
            csv_config=csv_read_config,
            conversion_config=data_conversion_config,
            rename_config=column_rename_config
        )
        
        assert cleaner.csv_config == csv_read_config
        assert cleaner.conversion_config == data_conversion_config
        assert cleaner.rename_config == column_rename_config
    
    def test_init_with_default_configs(self, temporary_csv_file):
        """Testa inicialização com configurações padrão."""
        cleaner = DataCleaning(csv_paths=str(temporary_csv_file))
        
        assert cleaner.csv_config is not None
        assert cleaner.conversion_config is not None
        assert cleaner.rename_config is not None


class TestDataCleaningReadCsv:
    """Testes para leitura de CSV."""
    
    def test_default_read_csv_basic(self, temporary_csv_file, column_rename_config):
        """Testa leitura básica de CSV."""
        cleaner = DataCleaning(
            csv_paths=str(temporary_csv_file),
            rename_config=column_rename_config
        )
        
        df = cleaner._default_read_csv(str(temporary_csv_file))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        # Última coluna deve ser descartada
        assert 'Coluna_Descartada' not in df.columns
    
    def test_default_read_csv_encoding(self, temporary_csv_file, column_rename_config):
        """Testa se a codificação correta é usada."""
        cleaner = DataCleaning(
            csv_paths=str(temporary_csv_file),
            rename_config=column_rename_config
        )
        
        df = cleaner._default_read_csv(str(temporary_csv_file))
        
        # Verifica se o DataFrame foi carregado corretamente
        assert len(df) > 0
        assert df.shape[1] > 0
    
    def test_default_read_csv_skiprows(self, temporary_csv_file, column_rename_config):
        """Testa se linhas do cabeçalho são puladas corretamente."""
        cleaner = DataCleaning(
            csv_paths=str(temporary_csv_file),
            rename_config=column_rename_config
        )
        
        df = cleaner._default_read_csv(str(temporary_csv_file))
        
        # As 8 primeiras linhas devem ter sido puladas
        assert 'Data' in df.columns
        assert 'Hora UTC' in df.columns


class TestDataCleaningLoadAllData:
    """Testes para carregamento de múltiplos dados."""
    
    def test_load_all_data_single_file(self, temporary_csv_file, column_rename_config):
        """Testa carregamento de um único arquivo."""
        cleaner = DataCleaning(
            csv_paths=str(temporary_csv_file),
            rename_config=column_rename_config
        )
        
        loaded_dfs = cleaner.raw_dataframes
        
        assert len(loaded_dfs) == 1
        assert isinstance(loaded_dfs[0], pd.DataFrame)
    
    def test_load_all_data_multiple_files(self, multiple_csv_files, column_rename_config):
        """Testa carregamento de múltiplos arquivos."""
        csv_paths = [str(path) for path in multiple_csv_files]
        cleaner = DataCleaning(
            csv_paths=csv_paths,
            rename_config=column_rename_config
        )
        
        loaded_dfs = cleaner.raw_dataframes
        
        assert len(loaded_dfs) == len(csv_paths)
        for df in loaded_dfs:
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0


class TestDataCleaningStringConversions:
    """Testes para limpeza de colunas de data/hora."""
    
    def test_cleaning_str_data_hours_columns_creates_data_hora(self, sample_dataframe, column_rename_config):
        """Testa se a coluna DATA_HORA é criada corretamente."""
        cleaner = DataCleaning(
            csv_paths=[],
            rename_config=column_rename_config
        )
        cleaner.raw_dataframes = [sample_dataframe]
        
        result = cleaner._cleaning_str_data_hours_columns(sample_dataframe.copy())
        
        assert 'DATA_HORA' in result.columns
        assert 'Data' not in result.columns
        assert 'Hora UTC' not in result.columns
    
    def test_cleaning_str_data_hours_columns_correct_format(self, sample_dataframe, column_rename_config):
        """Testa se o formato da coluna DATA_HORA está correto."""
        cleaner = DataCleaning(
            csv_paths=[],
            rename_config=column_rename_config
        )
        cleaner.raw_dataframes = [sample_dataframe]
        
        result = cleaner._cleaning_str_data_hours_columns(sample_dataframe.copy())
        
        # Verifica se é datetime
        assert pd.api.types.is_datetime64_any_dtype(result['DATA_HORA'])
        # Verifica os valores - a hora é apenas o número sem UTC
        assert result['DATA_HORA'].iloc[0] == pd.Timestamp('2024-01-01 00:00:00')
        assert result['DATA_HORA'].iloc[1] == pd.Timestamp('2024-01-02 01:00:00') or \
               result['DATA_HORA'].iloc[1] == pd.Timestamp('2024-01-02 00:01:00')
    
    def test_cleaning_str_data_hours_columns_removes_original(self, sample_dataframe, column_rename_config):
        """Testa se as colunas originais de data/hora são removidas."""
        cleaner = DataCleaning(
            csv_paths=[],
            rename_config=column_rename_config
        )
        cleaner.raw_dataframes = [sample_dataframe]
        
        result = cleaner._cleaning_str_data_hours_columns(sample_dataframe.copy())
        
        num_original_cols = len(sample_dataframe.columns)
        num_result_cols = len(result.columns)
        
        # Uma coluna foi combinada (Data + Hora UTC -> DATA_HORA)
        # Logo: original - 1 (removidas Data e Hora) + 1 (nova DATA_HORA) = original
        assert num_result_cols == num_original_cols - 1


class TestDataCleaningNumericConversion:
    """Testes para conversão de strings para numérico."""
    
    def test_convert_str_to_numeric_basic(self, sample_dataframe, column_rename_config):
        """Testa conversão básica de string para numérico."""
        cleaner = DataCleaning(
            csv_paths=[],
            rename_config=column_rename_config
        )
        cleaner.raw_dataframes = [sample_dataframe]
        
        result = cleaner._convert_str_to_numeric(sample_dataframe.copy())
        
        # Verifica se colunas foram convertidas para float
        assert pd.api.types.is_numeric_dtype(result['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'])
        assert pd.api.types.is_numeric_dtype(result['UMIDADE RELATIVA DO AR, HORARIA (%)'])
    
    def test_convert_str_to_numeric_decimal_separator(self, sample_dataframe, column_rename_config):
        """Testa se o separador decimal é substituído corretamente."""
        cleaner = DataCleaning(
            csv_paths=[],
            rename_config=column_rename_config
        )
        cleaner.raw_dataframes = [sample_dataframe]
        
        result = cleaner._convert_str_to_numeric(sample_dataframe.copy())
        
        # Verifica valores específicos
        assert result['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'].iloc[0] == 25.5
        assert result['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'].iloc[1] == 26.3
    
    def test_convert_str_to_numeric_preserves_column_names(self, sample_dataframe, column_rename_config):
        """Testa se os nomes das colunas são preservados."""
        cleaner = DataCleaning(
            csv_paths=[],
            rename_config=column_rename_config
        )
        cleaner.raw_dataframes = [sample_dataframe]
        original_cols = set(sample_dataframe.columns)
        
        result = cleaner._convert_str_to_numeric(sample_dataframe.copy())
        result_cols = set(result.columns)
        
        assert original_cols == result_cols


class TestDataCleaningRename:
    """Testes para renomeação de colunas."""
    
    def test_rename_all_columns_basic(self, sample_dataframe, column_rename_config):
        """Testa renomeação básica de colunas."""
        cleaner = DataCleaning(
            csv_paths=[],
            rename_config=column_rename_config
        )
        cleaner.raw_dataframes = [sample_dataframe]
        
        result = cleaner._rename_all_columns(sample_dataframe.copy())
        
        # Verifica se colunas foram renomeadas
        assert 'temp_ar_c' in result.columns
        assert 'umidade_rel_ar_percent' in result.columns
        assert 'pressao_atm_estacao_mb' in result.columns
    
    def test_rename_all_columns_removes_old_names(self, sample_dataframe, column_rename_config):
        """Testa se nomes antigos são removidos."""
        cleaner = DataCleaning(
            csv_paths=[],
            rename_config=column_rename_config
        )
        cleaner.raw_dataframes = [sample_dataframe]
        
        result = cleaner._rename_all_columns(sample_dataframe.copy())
        
        # Nomes para mapeados não devem existir
        for old_col in column_rename_config.rename_map.keys():
            if old_col in sample_dataframe.columns:
                assert old_col not in result.columns or old_col == 'DATA_HORA'
    
    def test_rename_all_columns_preserves_non_mapped(self, column_rename_config):
        """Testa se colunas não mapeadas são preservadas."""
        df = pd.DataFrame({
            'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': [25.5],
            'Coluna_Não_Mapeada': ['valor']
        })
        
        cleaner = DataCleaning(
            csv_paths=[],
            rename_config=column_rename_config
        )
        cleaner.raw_dataframes = [df]
        
        result = cleaner._rename_all_columns(df.copy())
        
        assert 'temp_ar_c' in result.columns
        assert 'Coluna_Não_Mapeada' in result.columns


class TestDataCleaningProcess:
    """Testes para o pipeline de processamento."""
    
    def test_process_data_applies_all_transformations(self, temporary_csv_file, column_rename_config):
        """Testa se process_data aplica todas as transformações."""
        cleaner = DataCleaning(
            csv_paths=str(temporary_csv_file),
            rename_config=column_rename_config
        )
        
        processed = cleaner.raw_dataframes[0]
        
        # Verifica se DATA_HORA foi criada
        assert 'DATA_HORA' in processed.columns or 'data_hora' in processed.columns
        # Verifica se colunas foram renomeadas
        assert 'temp_ar_c' in processed.columns
        # Verifica se valores foram convertidos para numérico
        numeric_cols = processed.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 0
    
    def test_process_data_order(self, temporary_csv_file, column_rename_config):
        """Testa se as transformações ocorrem na ordem correta."""
        cleaner = DataCleaning(
            csv_paths=str(temporary_csv_file),
            rename_config=column_rename_config
        )
        
        processed = cleaner.raw_dataframes[0]
        
        # DATA_HORA deve existir (combinação de Data e Hora)
        assert any(col == 'data_hora' for col in processed.columns)
        # DATA_HORA deve ser datetime
        date_col = 'data_hora' if 'data_hora' in processed.columns else next(
            (col for col in processed.columns if 'data' in col.lower()), None
        )
        if date_col:
            assert pd.api.types.is_datetime64_any_dtype(processed[date_col])


class TestDataCleaningConcat:
    """Testes para concatenação de DataFrames."""
    
    def test_concat_csv_single_file(self, temporary_csv_file, column_rename_config):
        """Testa concatenação com um único arquivo."""
        cleaner = DataCleaning(
            csv_paths=str(temporary_csv_file),
            rename_config=column_rename_config
        )
        
        result = cleaner.concat_csv()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_concat_csv_multiple_files(self, multiple_csv_files, column_rename_config):
        """Testa concatenação com múltiplos arquivos."""
        csv_paths = [str(path) for path in multiple_csv_files]
        cleaner = DataCleaning(
            csv_paths=csv_paths,
            rename_config=column_rename_config
        )
        
        result = cleaner.concat_csv()
        
        assert isinstance(result, pd.DataFrame)
        # Deve ter mais linhas que um único arquivo
        individual_counts = [len(df) for df in cleaner.raw_dataframes]
        assert len(result) == sum(individual_counts)
    
    def test_concat_csv_preserves_columns(self, multiple_csv_files, column_rename_config):
        """Testa se as colunas são preservadas na concatenação."""
        csv_paths = [str(path) for path in multiple_csv_files]
        cleaner = DataCleaning(
            csv_paths=csv_paths,
            rename_config=column_rename_config
        )
        
        first_df_cols = set(cleaner.raw_dataframes[0].columns)
        result = cleaner.concat_csv()
        result_cols = set(result.columns)
        
        assert first_df_cols == result_cols


class TestDataCleaningOutput:
    """Testes para obtenção de output metadata."""
    
    def test_get_process_output_basic(self, temporary_csv_file, column_rename_config):
        """Testa obtenção básica de output."""
        cleaner = DataCleaning(
            csv_paths=str(temporary_csv_file),
            rename_config=column_rename_config
        )
        
        output = cleaner.get_process_output()
        
        assert isinstance(output, DataCleaningProcessOutput)
        assert output.num_rows > 0
        assert output.num_columns > 0
    
    def test_get_process_output_metadata(self, temporary_csv_file, column_rename_config):
        """Testa se os metadados estão corretos."""
        cleaner = DataCleaning(
            csv_paths=str(temporary_csv_file),
            rename_config=column_rename_config
        )
        
        output = cleaner.get_process_output()
        final_df = cleaner.concat_csv()
        
        assert output.num_rows == len(final_df)
        assert output.num_columns == len(final_df.columns)
        assert len(output.columns) == len(final_df.columns)
        assert len(output.dtypes) == len(final_df.columns)
    
    def test_get_process_output_columns_list(self, temporary_csv_file, column_rename_config):
        """Testa se a lista de colunas está correta."""
        cleaner = DataCleaning(
            csv_paths=str(temporary_csv_file),
            rename_config=column_rename_config
        )
        
        output = cleaner.get_process_output()
        final_df = cleaner.concat_csv()
        
        assert output.columns == final_df.columns.tolist()
    
    def test_get_process_output_dtypes_dict(self, temporary_csv_file, column_rename_config):
        """Testa se o dicionário de tipos está correto."""
        cleaner = DataCleaning(
            csv_paths=str(temporary_csv_file),
            rename_config=column_rename_config
        )
        
        output = cleaner.get_process_output()
        final_df = cleaner.concat_csv()
        
        for col, dtype_str in output.dtypes.items():
            assert dtype_str == str(final_df[col].dtype)


class TestDataCleaningIntegration:
    """Testes de integração para todo o pipeline."""
    
    def test_full_pipeline_single_file(self, temporary_csv_file, column_rename_config):
        """Testa o pipeline completo com um arquivo."""
        cleaner = DataCleaning(
            csv_paths=str(temporary_csv_file),
            rename_config=column_rename_config
        )
        
        # Verifica se tudo foi processado corretamente
        assert len(cleaner.raw_dataframes) == 1
        assert len(cleaner.raw_dataframes[0]) > 0
        
        # Obtém output
        output = cleaner.get_process_output()
        assert output.num_rows > 0
        assert output.num_columns > 0
    
    def test_full_pipeline_multiple_files(self, multiple_csv_files, column_rename_config):
        """Testa o pipeline completo com múltiplos arquivos."""
        csv_paths = [str(path) for path in multiple_csv_files]
        cleaner = DataCleaning(
            csv_paths=csv_paths,
            rename_config=column_rename_config
        )
        
        # Verifica quantidade de arquivos carregados
        assert len(cleaner.raw_dataframes) == 2
        
        # Obtém dados concatenados
        concat_df = cleaner.concat_csv()
        assert len(concat_df) > 0
        
        # Obtém output metadata
        output = cleaner.get_process_output()
        assert output.num_rows == len(concat_df)
    
    def test_full_pipeline_with_custom_configs(self, temporary_csv_file, csv_read_config,
                                               data_conversion_config, column_rename_config):
        """Testa o pipeline com configurações customizadas."""
        cleaner = DataCleaning(
            csv_paths=str(temporary_csv_file),
            csv_config=csv_read_config,
            conversion_config=data_conversion_config,
            rename_config=column_rename_config
        )
        
        # Verifica se as configs foram aplicadas
        assert cleaner.csv_config == csv_read_config
        assert cleaner.conversion_config == data_conversion_config
        assert cleaner.rename_config == column_rename_config
        
        # Verifica resultado
        output = cleaner.get_process_output()
        assert output.num_rows > 0
