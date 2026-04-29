# API de Serving — WeatherOps

API REST assíncrona (FastAPI) para previsão horária de temperatura via modelos TFT e N-BEATS carregados do MLflow Model Registry. Expõe horizontes de 72, 168 e 336 horas.

Para iniciar a API: [docs/guides/QUICKSTART.md](../../docs/guides/QUICKSTART.md)
Para o agente conversacional: [src/api_agent/README.md](../api_agent/README.md)

---

## Endpoints

| Endpoint | Método | Descrição |
|---|---|---|
| `/health` | GET | Liveness — retorna `200` enquanto o processo estiver ativo |
| `/health/ready` | GET | Readiness — `200` quando todos os modelos estão carregados |
| `/v1/forecast/72` | POST | Previsão de temperatura para 72 horas |
| `/v1/forecast/168` | POST | Previsão de temperatura para 168 horas (7 dias) |
| `/v1/forecast/336` | POST | Previsão de temperatura para 336 horas (14 dias) |
| `/v1/agent/chat` | POST | Agente conversacional em PT-BR (requer `GOOGLE_API_KEY`) |
| `/metrics` | GET | Métricas Prometheus |
| `/docs` | GET | Swagger UI interativo |

### `GET /health/ready`

| Código | Situação |
|---|---|
| `200` | Pronto — todos os modelos carregados |
| `207` | Degradado — ao menos um modelo falhou ao carregar |
| `503` | Inicializando — aguardar e tentar novamente |

### `POST /v1/forecast/{horizon}`

**Body:**
```json
{
  "reference_date": "2024-06-01",
  "model_type": "tft",
  "group_id": "station_1"
}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `reference_date` | `"YYYY-MM-DD"` ou ISO 8601 | Último instante do histórico (inclusivo) |
| `model_type` | `"tft"` \| `"nbeats"` | Família do modelo |
| `group_id` | string | Identificador da estação nos dados Parquet |

**Códigos de resposta:**

| Código | Causa |
|---|---|
| `200` | Previsão gerada com sucesso |
| `404` | Modelo não carregado (não promovido no MLflow) |
| `422` | Horizonte inválido, `reference_date` inválido ou dados insuficientes |
| `500` | Erro interno de inferência |

---

## Variáveis de Ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `MLFLOW_TRACKING_URI` | — | URI do MLflow (ex.: `file:///app/mlruns`) |
| `WEATHEROPS_MODEL_ROOT` | — | Diretório com modelos exportados pelo `promote` |
| `PARQUET_PATH` | `/app/data/spec` | Caminho para arquivos Parquet de features |
| `DEVICE` | `cpu` | Dispositivo de inferência: `cpu` ou `cuda` |
| `CACHE_BACKEND` | `memory` | Backend do cache: `memory` ou `redis` |
| `CACHE_TTL_SECONDS` | `3600` | Tempo de vida das entradas no cache (segundos) |
| `REDIS_URL` | `redis://localhost:6379` | URL Redis (apenas quando `CACHE_BACKEND=redis`) |
| `ACCURACY_DB_PATH` | `/app/data/accuracy.db` | SQLite com histórico de previsões e acurácia |
| `ACCURACY_EVAL_INTERVAL_SECONDS` | `3600` | Intervalo do worker de avaliação de qualidade |
| `GOOGLE_API_KEY` | — | Chave Google AI para o agente (Gemini 2.5 Flash) |
| `AGENT_MAX_MESSAGE_CHARS` | `8000` | Tamanho máximo da mensagem de entrada do agente |
| `KNOWLEDGE_BASE_PATH` | `src/api_agent/knowledge/chroma_db` | ChromaDB para RAG |

---

## Adicionando um Novo Modelo

Adicione uma entrada em `REGISTERED_MODELS` em `config.py` — nenhuma outra alteração é necessária:

```python
ServingModelConfig(
    key="lstm_72",
    experiment_name="weather_lstm_h72",
    model_type="lstm",
    horizon=72,
    sequence_length=168,
    engine_class="sequential",   # "pytorch_forecasting" para TFT/N-BEATS
)
```

---

## Cache

A chave de cache é derivada de `(model_type, horizon, reference_date, group_id)`.
Para múltiplos workers ou pods, use Redis:

```bash
CACHE_BACKEND=redis REDIS_URL=redis://localhost:6379 docker compose ...
```

---

## Monitoramento (Prometheus + Grafana)

### Métricas expostas

| Métrica | Tipo | Descrição |
|---|---|---|
| `http_requests_total` | Counter | Requisições por rota, método e status |
| `http_request_duration_highr_seconds` | Histogram | Latência com buckets finos (p95/p99) |
| `http_requests_in_progress` | Gauge | Requisições em andamento |
| `weatherops_inference_duration_seconds` | Histogram | Latência de inferência por `model_key` |
| `weatherops_cache_hits_total` | Counter | Cache hits |
| `weatherops_cache_misses_total` | Counter | Cache misses |
| `weatherops_model_mae` | Gauge | MAE por modelo e bucket de horizonte |
| `weatherops_model_rmse` | Gauge | RMSE por modelo e bucket de horizonte |
| `weatherops_model_mape` | Gauge | MAPE por modelo e bucket de horizonte |

Buckets de horizonte: `near` (h1–24), `mid` (h25–72), `far` (h73+).

### Queries PromQL úteis

```promql
# Throughput (req/s, última 1 min)
rate(http_requests_total[1m])

# Latência p95 (última 5 min)
histogram_quantile(0.95, sum by (le) (rate(http_request_duration_highr_seconds_bucket{job="weatherops-api"}[5m])))

# MAPE por modelo
weatherops_model_mape{job="weatherops-api"}

# Taxa de erros 5xx
rate(http_requests_total{status=~"5.."}[1m])
```

---

## Estrutura

```
src/api/
├── main.py                  # Fábrica FastAPI e lifespan
├── config.py                # REGISTERED_MODELS, ServingModelConfig, Settings
├── cache.py                 # Cache TTL em memória ou Redis
├── dependencies.py          # Injetores de dependência FastAPI
├── metrics.py               # Instrumentação Prometheus
├── Dockerfile
├── docker-compose.yml       # weatherops-api, mlflow-ui, prometheus, grafana
├── engines/
│   ├── pytorch_forecasting.py  # Engine TFT/N-BEATS
│   └── sequential.py           # Engine LSTM/Transformer
├── routers/
│   ├── forecast.py          # POST /v1/forecast/{horizon}
│   └── health.py            # GET /health · /health/ready
└── services/
    ├── data_service.py      # Carrega Parquet e serve janelas de contexto
    ├── model_registry.py    # Carrega e mantém modelos por chave
    ├── predictor.py         # Orquestra inferência de ponta a ponta
    ├── prediction_logger.py # Persiste previsões em SQLite
    └── accuracy_evaluator.py # Calcula MAE/RMSE/MAPE em background
```

Arquitetura detalhada (engines, strategy pattern, startup sequence): [ARCHITECTURE.md](ARCHITECTURE.md)
