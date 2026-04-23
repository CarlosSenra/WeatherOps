# Data Engineering

Módulo responsável por transformar CSVs meteorológicos brutos em arquivos Parquet prontos para treinamento de modelos.

---

## Estrutura

```
core/data_engineering/
├── data_cleaning/
│   ├── data_cleaning.py     # DataCleaning — limpeza e padronização
│   └── __init__.py
├── data_feature_eng/
│   ├── feature_eng.py       # WeatherFeatureEngineer — geração de features
│   └── __init__.py
├── interface/
│   ├── i_data_eng.py        # IDataEngineering — contrato base
│   └── __init__.py
├── models/
│   ├── data_cleaning/       # Modelos Pydantic de configuração da limpeza
│   └── feature_engineering/ # Modelos Pydantic de configuração das features
└── __init__.py
```

---

## Componentes

### `DataCleaning` — `data_cleaning/data_cleaning.py`

Lê um ou mais CSVs de estações meteorológicas do INMET, concatena, padroniza o schema e entrega um DataFrame limpo.

**Operações realizadas:**
1. Leitura de CSVs com configuração flexível (separador, encoding, linhas a pular)
2. Renomeação de colunas para snake_case padronizado
3. Conversão de tipos (strings numéricas → float, data/hora → datetime)
4. Tratamento de valores ausentes (remoção ou interpolação, conforme config)
5. Padronização do índice temporal como `DATA_HORA`

**Configurações (modelos em `models/data_cleaning/`):**

| Classe | Campo | Padrão | Descrição |
|--------|-------|--------|-----------|
| `CsvReadConfig` | `sep` | `";"` | Separador do CSV |
| `CsvReadConfig` | `encoding` | `"latin-1"` | Encoding dos arquivos INMET |
| `CsvReadConfig` | `skiprows_start` | `8` | Linhas de cabeçalho a pular (metadados INMET) |
| `ColumnRenameConfig` | `rename_map` | dict | Mapa de renomeação (nomes INMET → snake_case) |
| `DataConversionConfig` | `numeric_columns` | list | Colunas a converter para float |

**Uso básico:**
```python
from core.data_engineering import DataCleaning

cleaner = DataCleaning(csv_paths=["data/raw/salvador_2023.csv", "data/raw/salvador_2024.csv"])
df_clean = cleaner.processed_data
# DataFrame com índice datetime e colunas padronizadas
```

---

### `WeatherFeatureEngineer` — `data_feature_eng/feature_eng.py`

Recebe um DataFrame limpo e gera as features derivadas necessárias para treinamento.

**Features criadas:**

| Feature | Fórmula | Propósito |
|---------|---------|----------|
| `hora_sin` | `sin(2π × hora / 24)` | Codificação cíclica da hora (componente seno) |
| `hora_cos` | `cos(2π × hora / 24)` | Codificação cíclica da hora (componente cosseno) |
| `temp_lag_1h` | `temp_ar_c.shift(1)` | Temperatura 1 hora atrás |
| `temp_lag_24h` | `temp_ar_c.shift(24)` | Temperatura 24 horas atrás (mesmo horário do dia anterior) |
| `temp_ma_6h` | Rolling mean 6 janelas | Média móvel de temperatura — tendência de curto prazo |
| `temp_ma_12h` | Rolling mean 12 janelas | Média móvel de temperatura — tendência de médio prazo |
| `pressao_ma_6h` | Rolling mean 6 janelas | Média móvel de pressão — passagem de frentes |
| `pressao_ma_12h` | Rolling mean 12 janelas | Média móvel de pressão — passagem de frentes |
| `pressao_tendencia_1h` | `pressao_atm_estacao_mb.diff(1)` | Taxa de variação de pressão (indica chegada de sistemas) |
| `temp_tendencia_1h` | `temp_ar_c.diff(1)` | Taxa de variação de temperatura |

**Configuração (modelos em `models/feature_engineering/`):**

| Campo | Padrão | Descrição |
|-------|--------|-----------|
| `lag_hours` | `[1, 24]` | Janelas de lag para temperatura |
| `moving_avg_windows` | `[6, 12]` | Janelas para médias móveis |
| `use_seasonal_features` | `True` | Gera `hora_sin` e `hora_cos` |

**Uso básico:**
```python
from core.data_engineering.data_feature_eng.feature_eng import WeatherFeatureEngineer

engineer = WeatherFeatureEngineer()
df_features = engineer.transform(df_clean)
# DataFrame com todas as features derivadas adicionadas
```

---

### `IDataEngineering` — `interface/i_data_eng.py`

Contrato (classe abstrata) que todas as etapas de engenharia de dados implementam. Garante que `DataCleaning` e `WeatherFeatureEngineer` possam ser compostos de forma previsível.

```python
class IDataEngineering(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

---

## Schema de Saída — `data/spec`

Os Parquets em `data/spec` devem conter as seguintes colunas para compatibilidade com o treinamento de modelos TFT:

| Coluna | Tipo | Obrigatório para TFT | Descrição |
|--------|------|---------------------|-----------|
| `temp_ar_c` | float | Sim (target) | Temperatura do ar em °C |
| `umidade_rel_ar_percent` | float | Sim | Umidade relativa do ar |
| `pressao_atm_estacao_mb` | float | Sim | Pressão atmosférica na estação |
| `precipitacao_total_mm` | float | Sim | Precipitação total |
| `hora_sin` | float | **Sim** (known_future) | Seno da hora — obrigatório no TFT |
| `hora_cos` | float | **Sim** (known_future) | Cosseno da hora — obrigatório no TFT |
| `temp_lag_1h` | float | Sim | Lag de 1 hora |
| `temp_lag_24h` | float | Sim | Lag de 24 horas |
| `temp_ma_6h` | float | Sim | Média móvel 6h |
| `temp_ma_12h` | float | Sim | Média móvel 12h |
| `pressao_ma_6h` | float | Sim | Média móvel pressão 6h |
| `pressao_ma_12h` | float | Sim | Média móvel pressão 12h |
| `pressao_tendencia_1h` | float | Sim | Tendência de pressão |
| `temp_tendencia_1h` | float | Sim | Tendência de temperatura |

O índice deve ser um `DatetimeIndex` com frequência horária.

> **Atenção:** alterar nomes de colunas em `data/spec` sem atualizar `feature_columns` nos JSONs de experimento de treinamento quebra o pipeline de treino silenciosamente.

---

## Como Usar

### Pipeline completo (limpeza + features + Parquet)

```python
from core.data_engineering import DataCleaning
from core.data_engineering.data_feature_eng.feature_eng import WeatherFeatureEngineer
import glob

# 1. Limpeza
csv_paths = glob.glob("data/raw/salvador_*.csv")
cleaner = DataCleaning(csv_paths=csv_paths)
df_clean = cleaner.processed_data

# 2. Feature engineering
engineer = WeatherFeatureEngineer()
df_features = engineer.transform(df_clean)

# 3. Salvar como Parquet
df_features.to_parquet("data/spec/salvador_features.parquet")

print(f"Parquet gerado: {len(df_features)} linhas, {len(df_features.columns)} colunas")
print(f"Período: {df_features.index.min()} → {df_features.index.max()}")
```

### Verificar schema do Parquet gerado

```python
import pandas as pd

df = pd.read_parquet("data/spec/salvador_features.parquet")
print(df.dtypes)
print(df.isnull().sum())  # verificar nulos residuais nas bordas (lag/rolling)
```

---

## Integração com Airflow

Os DAGs em `src/data_airflow/dags/` executam esse pipeline automaticamente:

| DAG | Entrada | Saída | Aciona |
|-----|---------|-------|--------|
| `data_cleaning` | `data/raw/*.csv` | `data/staging/*.csv` | Manual |
| `data_feature_engineering` | `data/staging/*.csv` | `data/spec/*.parquet` | Manual |

Para rodar via Airflow, consulte `src/data_airflow/README.md`.

---

## Testes

Os testes unitários cobrem as principais transformações:

```
test/unit/core/data_engineering/
├── data_cleaning/
│   ├── test_data_cleaning.py  # leitura CSV, renomeação, conversão de tipos
│   └── conftest.py
└── data_feature_eng/
    ├── test_feature_eng.py    # lags, médias móveis, features cíclicas
    └── conftest.py
```

Executar:
```bash
poetry run pytest test/unit/core/data_engineering/ -v
```
