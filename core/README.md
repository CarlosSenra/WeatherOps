# Core

Módulo de engenharia de dados do WeatherOps.

## Objetivo

Transformar dados meteorológicos brutos em dados confiáveis e prontos para treinamento.

## Estrutura

```text
core/
├── data_engineering/
│   ├── data_cleaning/
│   │   └── data_cleaning.py
│   ├── data_feature_eng/
│   │   └── feature_eng.py
│   └── interface/
│       └── i_data_eng.py
└── data_analynitcs/       # utilitários de gráficos exploratórios (não produção)
```

## Fluxo

```
data/raw/ CSV → DataCleaning → WeatherFeatureEngineer → data/spec/ Parquet → ml_workstation
```

## Componentes

- `data_cleaning` — padronização de schema, tratamento de nulos e consistência de valores
- `data_feature_eng` — criação de variáveis de tempo, lags e agregações
- `interface/i_data_eng.py` — contrato para pipelines de engenharia de dados

Documentação detalhada com exemplos, configurações e schema de saída: [data_engineering/README.md](data_engineering/README.md)

## Testes

```
test/unit/core/data_engineering/data_cleaning
test/unit/core/data_engineering/data_feature_eng
```
