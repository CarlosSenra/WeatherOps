# Data Airflow Config

Diretorio para configuracoes do Airflow compartilhadas com os containers.

## Conteudo esperado

- `airflow.cfg` customizado (quando necessario).
- Arquivos auxiliares de configuracao referenciados por DAGs.

## inmet_scraping.yml

Intervalo de anos para download dos dados historicos do INMET.

```yaml
start_year: 2024
end_year: 2026
```

## municipios.yml

Configuracao de descoberta e filtro de municipios processados pelas DAGs
`data_cleaning` e `data_feature_engineering`.

```yaml
mode: all
include: []
exclude: []
slug_overrides: {}
```

Campos:

| Campo | Descricao |
|-------|-----------|
| `mode` | Modo de descoberta. Atualmente suportado: `all` (descobre todos os municipios a partir dos arquivos INMET em `data/raw/`). |
| `include` | Lista opcional de slugs para processar (ex.: `["brasilia", "dom_pedrito"]`). Se vazio, inclui todos detectados. |
| `exclude` | Lista opcional de slugs para remover do processamento final. |
| `slug_overrides` | Mapa opcional para renomear slugs detectados (ex.: `{"dom_pedrito": "dompedrito_rs"}`). |

A deteccao de municipio e feita pelo nome do arquivo no padrao INMET:

- `INMET_CO_DF_A001_BRASILIA_01-01-2026_A_31-03-2026.CSV`
- `INMET_S_RS_A881_DOM PEDRITO_01-01-2026_A_31-03-2026.CSV`

Os resultados sao gravados em:

- `data/staging/<municipio_slug>/<municipio_slug>.csv`
- `data/spec/<municipio_slug>/dados.parquet`

## Como e usado

No `docker-compose-airflow.yaml`, este diretorio e montado como:

- Host: `./src/data_airflow/config`
- Container: `/opt/airflow/config`

A variavel `AIRFLOW_CONFIG` aponta para:

- `/opt/airflow/config/airflow.cfg`

## Boas praticas

1. Nao commitar secrets em texto puro.
2. Preferir variaveis de ambiente para credenciais.
3. Versionar somente parametros nao sensiveis e defaults.
4. Documentar qualquer override relevante no README do modulo.
