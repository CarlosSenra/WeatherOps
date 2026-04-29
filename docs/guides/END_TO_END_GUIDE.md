# Guia End-to-End — WeatherOps

Guia passo a passo para rodar o pipeline completo: da preparação dos dados brutos
até servir previsões meteorológicas via API.

> **Só quer usar a API com modelos já promovidos?** Veja [QUICKSTART.md](QUICKSTART.md).

---

## Visão Geral

```
data/raw/ (CSVs INMET)
    │  Passo 2 — Airflow DAG inmet_download_raw
    ▼
data/raw/<ano>/
    │  Passo 3 — DataCleaning + WeatherFeatureEngineer
    ▼
data/spec/<municipio>/ (Parquet com features)
    │  Passo 4 — Configurar experimento JSON
    │  Passo 5 — Treinar modelo
    ▼
MLflow Tracking (mlruns/)
    │  Passo 6 — Promover modelo
    ▼
MLflow Registry (alias: production) + src/api/ml_models/
    │  Passo 7 — Servir via API
    ▼
POST /v1/forecast/{horizon}
```

---

## Pré-requisitos

| Ferramenta | Versão | Necessário para |
|---|---|---|
| Python | 3.12+ | Tudo |
| Poetry | 1.8+ | Instalar dependências |
| Docker + Docker Compose V2 | 24+ | Airflow, treinamento em container, API |
| Git | 2.40+ | Clonar o repositório |
| NVIDIA Driver + nvidia-container-toolkit | — | Treinamento com GPU (opcional) |

---

## Passo 1 — Configurar o ambiente

Clone o repositório e instale as dependências:

```bash
git clone <url-do-repositorio>
cd WeatherOps
poetry install
```

Copie o arquivo de variáveis de ambiente:

```bash
cp .env.example .env
```

Edite `.env` e preencha ao menos:

```dotenv
AIRFLOW_UID=50000
EMAIL=seu@email.com
```

---

## Passo 2 — Baixar dados brutos do INMET

A DAG `inmet_download_raw` faz o download dos CSVs anuais do INMET e popula
`data/raw/<ano>/` automaticamente.

**Subir o Airflow:**

```bash
# Inicializar banco de dados do Airflow (apenas na primeira vez)
docker compose -f docker-compose-airflow.yaml up airflow-init

# Subir o Airflow
docker compose -f docker-compose-airflow.yaml up -d
```

Acesse `http://localhost:8080` (usuário: `airflow` / senha: `airflow`).

**Configurar o período de download** em `src/data_airflow/config/inmet_scraping.yml`:

```yaml
start_year: 2024
end_year: 2026
```

Ative e execute a DAG `inmet_download_raw` pela UI ou via CLI:

```bash
docker compose -f docker-compose-airflow.yaml exec airflow-webserver \
  airflow dags trigger inmet_download_raw
```

Resultado esperado: arquivos CSV em `data/raw/2024/`, `data/raw/2025/`, etc.

> **Alternativa (script automático):** Os scripts abaixo orquestram Airflow +
> engenharia de dados de uma vez:
> ```bash
> # Windows (PowerShell)
> ./scripts/run_pipeline.ps1 -Mode bootstrap -Device auto -StartYear 2024 -EndYear 2026
>
> # Linux/macOS
> ./scripts/run_pipeline.sh --mode bootstrap --device auto --start-year 2024 --end-year 2026
> ```

---

## Passo 3 — Engenharia de dados

Converte os CSVs brutos em Parquet com features prontas para treinamento.
O resultado é salvo em `data/spec/<municipio>/`.

**Via DAG Airflow** (recomendado): ative a DAG `data_feature_engineering` pela UI
após o download dos CSVs — ela cobre `DataCleaning` e `WeatherFeatureEngineer` em sequência.

**Via Python** (para desenvolvimento ou debugging):

```python
from core.data_engineering import DataCleaning
from core.data_engineering.data_feature_eng.feature_eng import WeatherFeatureEngineer

# 1. Limpeza
cleaner = DataCleaning(csv_paths=["data/raw/2024/salvador.csv"])
df_clean = cleaner.processed_data

# 2. Feature engineering
engineer = WeatherFeatureEngineer()
df_features = engineer.transform(df_clean)

# 3. Salvar
df_features.to_parquet("data/spec/salvador/data.parquet")
```

**Schema de saída esperado** (colunas obrigatórias para TFT):

| Coluna | Tipo | Descrição |
|---|---|---|
| `temp_ar_c` | float | Temperatura do ar °C (target) |
| `umidade_rel_ar_percent` | float | Umidade relativa |
| `pressao_atm_estacao_mb` | float | Pressão atmosférica |
| `hora_sin` / `hora_cos` | float | Codificação cíclica da hora |
| `temp_lag_1h` / `temp_lag_24h` | float | Lags de temperatura |
| `temp_ma_6h` / `temp_ma_12h` | float | Médias móveis de temperatura |
| `pressao_ma_6h` / `pressao_ma_12h` | float | Médias móveis de pressão |
| `pressao_tendencia_1h` / `temp_tendencia_1h` | float | Taxas de variação |

Referência completa: [core/data_engineering/README.md](../core/data_engineering/README.md).

---

## Passo 4 — Configurar experimento de treinamento

Crie ou edite um arquivo JSON em `src/ml_workstation/experiments/tft/`.

Convenção de nome: `tft_h<horizonte>_v<versao>.json`
(ex.: `tft_h72_v1.json`)

**Estrutura mínima:**

```json
{
  "data": {
    "parquet_path": "/app/data/spec/salvador",
    "feature_columns": [
      "temp_ar_c", "umidade_rel_ar_percent", "pressao_atm_estacao_mb",
      "precipitacao_total_mm", "hora_sin", "hora_cos",
      "temp_lag_1h", "temp_lag_24h", "temp_ma_6h", "temp_ma_12h",
      "pressao_ma_6h", "pressao_ma_12h", "pressao_tendencia_1h", "temp_tendencia_1h"
    ],
    "target_columns": ["temp_ar_c"],
    "sequence_length": 168,
    "horizon": 72,
    "train_ratio": 0.8,
    "val_ratio": 0.1
  },
  "model": {
    "model_type": "tft",
    "hidden_size": 64,
    "attention_head_size": 4,
    "dropout": 0.1,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 50,
    "early_stopping_patience": 5
  },
  "device": "cuda"
}
```

**Regras importantes:**

- `"device": "cuda"` — mantenha mesmo treinando em CPU via container; o trainer resolve o dispositivo disponível automaticamente.
- `parquet_path` é o caminho **dentro do container** (`/app/...`). Para treino fora do Docker use o caminho absoluto local.
- Para outros horizontes, altere `horizon` e `sequence_length` (recomendado: `sequence_length = horizon × 2`).

Referência completa: [src/ml_workstation/README.md](../src/ml_workstation/README.md).

---

## Passo 5 — Treinar o modelo

**Via Docker (recomendado):**

```bash
cd src/ml_workstation

# CPU
docker compose --profile train run --rm trainer \
  --config //app/experiments/tft/tft_h72_v1.json

# GPU
docker compose --profile train-gpu run --rm trainer-gpu \
  --config //app/experiments/tft/tft_h72_v1.json
```

**Via Poetry (sem Docker):**

```bash
poetry run python -m src.ml_workstation.train \
  --config src/ml_workstation/experiments/tft/tft_h72_v1.json
```

**Acompanhar o treinamento no MLflow:**

```bash
cd src/ml_workstation
docker compose --profile ui up -d mlflow-ui
```

Acesse `http://localhost:5000` para ver métricas por época, parâmetros e artefatos.

O treinamento salva automaticamente:
- Métricas por época (train_loss, val_loss, MAE, RMSE, MAPE) no MLflow
- Checkpoint do melhor modelo (`best_model.ckpt` para TFT)
- `scaler.pkl` (para LSTM/Transformer)
- Metadados de governança (git_sha, owner, data_version)

---

## Passo 6 — Promover o modelo para produção

A promoção registra o melhor run do experimento no **MLflow Model Registry** com
o alias `production` e, opcionalmente, exporta o artefato para disco.

**Promover o melhor run automaticamente (menor MAPE):**

```bash
# Promover e exportar para a pasta usada pela API
poetry run python -m src.ml_workstation.promotion.run_promote \
  --experiment-name weather_tft_h72 \
  --export-dir src/api/ml_models
```

**Promover um run específico:**

```bash
poetry run python -m src.ml_workstation.promotion.run_promote \
  --experiment-name weather_tft_h72 \
  --run-id <run_id_do_mlflow> \
  --export-dir src/api/ml_models
```

**Forçar promoção mesmo com MAPE pior** (ex.: em experimentos):

```bash
poetry run python -m src.ml_workstation.promotion.run_promote \
  --experiment-name weather_tft_h72 \
  --force \
  --export-dir src/api/ml_models
```

**Verificar a promoção** no MLflow UI (`http://localhost:5000`):
- Vá em **Models** → `weather_forecasting_h72`
- Confirme que a versão mais recente tem o alias `production`

**Repita para os outros horizontes** que quiser servir:

```bash
poetry run python -m src.ml_workstation.promotion.run_promote \
  --experiment-name weather_tft_h168 \
  --export-dir src/api/ml_models

poetry run python -m src.ml_workstation.promotion.run_promote \
  --experiment-name weather_tft_h336 \
  --export-dir src/api/ml_models
```

Referência completa: [src/ml_workstation/promotion/PROMOTION.md](../src/ml_workstation/promotion/PROMOTION.md).

---

## Passo 7 — Ativar o Agente com RAG (opcional)

Se quiser usar o **agente conversacional** com contexto histórico de clima, adicione
`--update-knowledge-base` ao promote. Isso gera os perfis mensais do município e
constrói a base vetorial ChromaDB usada pelo agente.

**Pré-requisitos:**
- `GOOGLE_API_KEY` configurada no `.env`
- Dependências do grupo `agent`: `poetry install --only agent`

```bash
# Promote + gerar feature store + rebuild ChromaDB (tudo em um comando)
GOOGLE_API_KEY=$(grep GOOGLE_API_KEY .env | cut -d= -f2) \
poetry run python -m src.ml_workstation.promotion.run_promote \
  --experiment-name weather_tft_h72 \
  --export-dir src/api/ml_models \
  --update-knowledge-base
```

Saída esperada:
```
Melhor run promovido com sucesso.
  Experimento : weather_tft_h72
  ...
Base vetorial atualizada: 13 chunks em .../src/api_agent/knowledge/chroma_db
```

Após subir a API (próximo passo), o log de startup mostrará:
```
INFO  FeatureStoreRetriever carregado — RAG ativo no agente.
```

> Sem esse passo o agente ainda funciona, mas sem a tool `get_historical_weather_context`.
> Veja o guia detalhado: [docs/guides/AGENT_QUICKSTART.md](AGENT_QUICKSTART.md)

---

## Passo 8 — Servir e testar

Com os modelos promovidos e (opcionalmente) o RAG gerado, siga o [QUICKSTART.md](QUICKSTART.md) para
subir a API e fazer a primeira previsão.

```bash
docker compose -f src/api/docker-compose.yml --profile api up --build -d
curl http://localhost:8888/health/ready
```

---

## Referências

| Tópico | Documento |
|---|---|
| Usar a API (guia rápido) | [docs/guides/QUICKSTART.md](QUICKSTART.md) |
| Usar o agente e RAG | [docs/guides/AGENT_QUICKSTART.md](AGENT_QUICKSTART.md) |
| Engenharia de dados | [core/data_engineering/README.md](../core/data_engineering/README.md) |
| Orquestração Airflow | [src/data_airflow/README.md](../src/data_airflow/README.md) |
| Treinamento de modelos | [src/ml_workstation/README.md](../src/ml_workstation/README.md) |
| Promoção de modelos | [src/ml_workstation/promotion/PROMOTION.md](../src/ml_workstation/promotion/PROMOTION.md) |
| API de serving | [src/api/README.md](../src/api/README.md) |
| Agente conversacional | [src/api_agent/README.md](../src/api_agent/README.md) |
