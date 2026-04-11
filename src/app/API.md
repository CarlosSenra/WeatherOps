# WeatherOps Forecast API — Documentação

## Visão Geral

A WeatherOps Forecast API oferece endpoints para consultar o estado dos modelos em
produção e executar inferência (previsão de temperatura) para os horizontes
suportados: 72h, 168h (7 dias) e 336h (14 dias).

Base path: `/api/v1`
OpenAPI UI: `/api/v1/docs`

---

## Quick start

### Local (venv / Poetry)

Com `poetry` (recomendado quando o repositório usa Poetry):

```bash
poetry install
poetry run uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

Alternativa com `pip` (usa `src/app/requirements.txt`):

```bash
pip install -r src/app/requirements.txt
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

A API ficará disponível em `http://localhost:8000/api/v1`.

### Docker

Build da imagem (contexto: raiz do repositório):

```bash
docker build -f src/app/Dockerfile -t weatherops-api .
```

Rodar (exemplo apontando para MLflow remoto e compartilhando `mlruns` local):

```bash
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -v $(pwd)/src/ml_workstation/mlruns:/app/src/ml_workstation/mlruns:ro \
  weatherops-api
```

---

## Variáveis de ambiente importantes

- `MLFLOW_TRACKING_URI` — URI do MLflow tracking server (ex.: `http://mlflow:5000`).
- `DEVICE` — `cpu` ou `cuda` (default: `cpu`).
- `LOG_LEVEL` — nível de log (`INFO`, `DEBUG`, ...).
- `HOST` — host da aplicação (default: `0.0.0.0`).
- `PORT` — porta (default: `8000`).

As variáveis são carregadas por `src/app/core/settings.py`.

---

## Endpoints (prefixo `/api/v1`)

### 1) Health

- Método: `GET`
- URL: `/api/v1/health`
- Descrição: liveness/version check.

Resposta 200:

```json
{
  "status": "ok",
  "version": "dev"
}
```

Códigos:
- `200 OK` — API pronta.

---

### 2) Models — lista

- Método: `GET`
- URL: `/api/v1/models`
- Descrição: lista metadados dos experimentos configurados.

Resposta 200 (exemplo):

```json
{
  "models": [
    {
      "experiment_name": "weather_forecasting_h72",
      "model_name": "lstm_h72_v1",
      "run_id": "14815808b23f47238e2e2379f59f2d2e",
      "mape": 2.45,
      "promoted_at": "2026-03-20",
      "promoted_by": "ci-bot"
    }
  ],
  "total": 1
}
```

Códigos:
- `200 OK` — lista retornada.

---

### 3) Model — detalhe

- Método: `GET`
- URL: `/api/v1/models/{experiment_name}`
- Descrição: metadados do modelo promovido para o experimento.

Resposta 200 (exemplo):

```json
{
  "experiment_name": "weather_forecasting_h72",
  "model_name": "lstm_h72_v1",
  "run_id": "14815808b23f47238e2e2379f59f2d2e",
  "mape": 2.45,
  "promoted_at": "2026-03-20",
  "promoted_by": "ci-bot"
}
```

Códigos:
- `200 OK` — modelo encontrado.
- `404 Not Found` — nenhum modelo promovido para `experiment_name`.

---

### 4) Forecast (inferência)

- Método: `POST`
- URL: `/api/v1/forecast/{experiment_name}`
- Descrição: executa inferência com o modelo em produção.

A API espera um JSON com o campo `features`, um array 3-D com shape `(batch, seq_len, n_features)`.
A API normaliza internamente usando o `scaler.pkl` logado no run do MLflow e desnormaliza a saída
para a escala original.

Request (exemplo simplificado — `batch=1`, `seq_len=24`, `n_features=14`):

```json
{
  "features": [
    [
      [25.3, 72.0, 1013.2, 0.0, 0.866, 0.5, 25.1, 24.8, 25.2, 25.0, 1013.0, 1012.8, 0.2, 0.15],
      [25.4, 71.5, 1013.1, 0.0, 0.87, 0.4, 25.0, 24.9, 25.1, 24.9, 1012.9, 1012.7, 0.25, 0.18]
      /* ... até seq_len ... */
    ]
  ]
}
```

Response 200 (exemplo para horizon=72, n_targets=1):

```json
{
  "experiment_name": "weather_forecasting_h72",
  "predictions": [
    [
      [25.2],
      [25.4],
      /* ... 72 timesteps ... */
    ]
  ],
  "model_meta": {
    "run_id": "14815808b23f47238e2e2379f59f2d2e",
    "mape": 2.45,
    "promoted_at": "2026-03-20",
    "promoted_by": "ci-bot"
  }
}
```

Códigos:
- `200 OK` — inferência bem-sucedida.
- `404 Not Found` — nenhum modelo em produção para `experiment_name`.
- `422 Unprocessable Entity` — payload inválido (shape incorreto).
- `503 Service Unavailable` — falha ao carregar modelo/scaler (MLflow inacessível).

---

## Notas de implementação (onde procurar no código)

- App factory & lifespan: `src/app/main.py`
- Routers: `src/app/routers/health.py`, `src/app/routers/models.py`, `src/app/routers/forecast.py`
- Serviço de inferência (normalização, inferência, desnormalização):
  `src/app/services/prediction_service.py`
- Cache/factory de serviços: `src/app/dependencies/model_cache.py` (usa `@lru_cache`)
- Loader de produção / artefatos: `src/ml_workstation/promotion/loader.py`
- Script de treinamento (gera e loga `scaler.pkl`): `src/ml_workstation/train.py`

---

## Scaler / Treinamento — nota crítica

- Para normalizar/desnormalizar corretamente, cada run treinado deve **logar o `scaler.pkl`**
  (o `StandardScaler` ajustado no conjunto de treino) como artefato do run no MLflow.
- `src/ml_workstation/train.py` foi atualizado para salvar `scaler.pkl` e logá-lo com
  `mlflow.log_artifact("scaler.pkl")` após o treinamento.
- Se houver modelos já promovidos que não possuem `scaler.pkl`, você deve re-treiná-los
  (ou manualmente gerar e subir um `scaler.pkl` compatível) para que a API consiga
  normalizar entradas e desnormalizar saídas corretamente.

---

## Testes rápidos (curl)

Health:

```bash
curl -sS http://localhost:8000/api/v1/health
```

Listar modelos:

```bash
curl -sS http://localhost:8000/api/v1/models
```

Inferência (payload compacto):

```bash
curl -X POST http://localhost:8000/api/v1/forecast/weather_forecasting_h72 \
  -H "Content-Type: application/json" \
  -d '{"features": [[[25.3,72.0,1013.2,0.0,0.866,0.5,25.1,24.8,25.2,25.0,1013.0,1012.8,0.2,0.15]]]}'
```

---

## Segurança & próximos passos

- Adicionar autenticação/autorizações (API Key ou OAuth2) para ambientes públicos.
- Proteger MLflow e Model Registry em rede privada.
- Limitar tamanho de payloads e aplicar rate-limiting.
- Observabilidade: registrar métricas de latência e taxas de erro (Prometheus).
- Integração contínua: pipeline para treinar → logar artefatos → promover → atualizar
  `models_production.yaml` automaticamente.

---

## Contato

Equipe WeatherOps — documentação gerada automaticamente pelo assistente de implementação.
