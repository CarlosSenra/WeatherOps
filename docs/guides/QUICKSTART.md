# Quickstart — WeatherOps API

Guia para ir do clone ao primeiro forecast sem dúvidas. Tempo estimado: **5–10 minutos**.

> Só quer rodar o pipeline do zero (dados → treino → API)?
> Veja [END_TO_END_GUIDE.md](END_TO_END_GUIDE.md).

---

## Pré-requisitos

Antes de começar, verifique que você tem:

- **Docker >= 24** com **Docker Compose V2** instalado e em execução
  - Verifique: `docker compose version` (deve retornar `v2.x.x`)
- **~8 GB de RAM** livres para o container da API
- **~5 GB de espaço em disco** para imagens Docker e modelos
- **`GOOGLE_API_KEY`** — necessária **apenas** para o agente conversacional
  - Obtenha gratuitamente em [aistudio.google.com](https://aistudio.google.com) → "Get API key"
  - Se não precisar do agente, pode deixar em branco

---

## Passo 1 — Clonar o repositório

```bash
git clone <url-do-repositorio>
cd WeatherOps
```

---

## Passo 2 — Configurar o ambiente

Copie o arquivo de variáveis de ambiente:

```bash
cp .env.example .env
```

Abra o `.env` e preencha apenas o que for necessário:

```dotenv
# Obrigatório só para o agente conversacional (/v1/agent/chat)
# Deixe em branco se não for usar o agente
GOOGLE_API_KEY=AIzaSy...sua_chave_aqui

# As variáveis abaixo já têm defaults funcionais para o Docker.
# Só altere se tiver uma configuração personalizada.
# MLFLOW_TRACKING_URI=file:///app/mlruns
# WEATHEROPS_MODEL_ROOT=/app/ml_models
# PARQUET_PATH=/app/data/spec
# DEVICE=cpu
```

> Os caminhos `/app/...` são internos ao container — não precisam ser ajustados
> para uma instalação padrão com Docker Compose.

---

## Passo 3 — Subir o stack

A partir da raiz do repositório:

```bash
# Stack completo: API + MLflow UI + Prometheus + Grafana
docker compose -f src/api/docker-compose.yml --profile api up --build -d

# Ou só a API (mais rápido para testes):
docker compose -f src/api/docker-compose.yml --profile api up --build weatherops-api
```

Na primeira execução o Docker vai baixar as imagens base e instalar dependências
— isso pode levar **3–5 minutos**. Execuções seguintes são muito mais rápidas.

---

## Passo 4 — Aguardar a API ficar pronta

A API carrega os modelos ML no startup. Aguarde e verifique a prontidão:

```bash
curl http://localhost:8888/health/ready
```

| Código HTTP | Situação | O que fazer |
|---|---|---|
| `200` | Pronta — todos os modelos carregados | Pode fazer requests |
| `207` | Degradada — ao menos um modelo falhou | Verifique os logs; outros modelos funcionam |
| `503` | Ainda inicializando | Aguarde ~30–60s e tente novamente |

Resposta esperada quando pronta:

```json
{
  "status": "pronto",
  "data_service": "pronto",
  "models_loaded": ["tft_72", "tft_168", "tft_336"],
  "models_failed": [],
  "row_count": 87648
}
```

---

## Passo 5 — Primeiro forecast

```bash
curl -X POST http://localhost:8888/v1/forecast/72 \
  -H "Content-Type: application/json" \
  -d '{
    "reference_date": "2024-06-01",
    "model_type": "tft",
    "group_id": "station_1"
  }'
```

**Parâmetros:**

| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `reference_date` | `"YYYY-MM-DD"` ou ISO 8601 | obrigatório | Último instante do histórico (inclusivo). A primeira previsão é `reference_date + 1h`. |
| `model_type` | `"tft"` | `"tft"` | Família do modelo. Atualmente `tft` está ativo em produção. |
| `group_id` | string | `"station_1"` | Identificador da estação nos dados Parquet. |
| `horizon` (path) | `72` · `168` · `336` | — | Horas à frente a prever (definido na URL, não no body). |

> **Dica:** use `http://localhost:8888/docs` (Swagger UI) para explorar todos os parâmetros
> interativamente e testar sem precisar de `curl`.

**Resposta:**

```json
{
  "reference_date": "2024-06-01T00:00:00",
  "horizon": 72,
  "model_type": "tft",
  "model_version": "3",
  "mape": 2.47,
  "predictions": [
    {"timestamp": "2024-06-01T01:00:00", "temp_ar_c": 23.1},
    {"timestamp": "2024-06-01T02:00:00", "temp_ar_c": 22.8}
  ],
  "latency_ms": 148.5
}
```

| Campo | Descrição |
|---|---|
| `predictions` | Lista de `horizon` pontos horários com temperatura (°C) |
| `model_version` | Versão do modelo no MLflow Model Registry |
| `mape` | MAPE registrado na promoção do modelo (%) |
| `latency_ms` | Tempo de inferência em ms (exclui cache hits) |

---

## Passo 6 — (Opcional) Agente conversacional

Se você configurou `GOOGLE_API_KEY` no `.env`, o agente já está ativo:

```bash
curl -X POST http://localhost:8888/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Como vai estar a temperatura amanhã à tarde?"}'
```

Para o guia completo do agente, incluindo como ativar o contexto histórico com RAG:
→ [AGENT_QUICKSTART.md](AGENT_QUICKSTART.md)

---

## Serviços disponíveis após `up`

| Serviço | URL | Credenciais |
|---|---|---|
| API | http://localhost:8888 | — |
| Swagger UI | http://localhost:8888/docs | — |
| MLflow UI | http://localhost:5001 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

O datasource do Prometheus é provisionado automaticamente no Grafana.

---

## Troubleshooting

| Sintoma | Causa provável | Solução |
|---|---|---|
| `/health/ready` retorna `503` | Modelos ainda carregando | Aguardar 30–60s e tentar novamente |
| `/health/ready` retorna `207` | Um ou mais modelos falharam | `docker compose logs weatherops-api` para ver o erro |
| `/v1/agent/chat` retorna `400` | `GOOGLE_API_KEY` ausente ou inválida | Adicionar ao `.env` e reiniciar com `--build` |
| `/v1/forecast` retorna `404` | Modelo não encontrado (não promovido) | Verifique `models_loaded` no `/health/ready` |
| `/v1/forecast` retorna `422` | `reference_date` sem dados suficientes no Parquet | Use uma data coberta pelo `data/spec` e verifique `PARQUET_PATH` |
| Porta já em uso | Outro processo usa a porta 8888 | Altere o mapeamento em `src/api/docker-compose.yml`: `"9000:8000"` |
| `/v1/agent/chat` não aparece no `/docs` | Deps do agente não instaladas na imagem | `docker compose -f src/api/docker-compose.yml --profile api up --build` |

---

## Horizontes disponíveis

| Endpoint | Horizonte | Janela de contexto | Modelo ativo |
|---|---|---|---|
| `POST /v1/forecast/72` | 72h (3 dias) | 168h | TFT |
| `POST /v1/forecast/168` | 168h (7 dias) | 336h | TFT |
| `POST /v1/forecast/336` | 336h (14 dias) | 504h | TFT |

---

## Referências

| Tópico | Documento |
|---|---|
| Referência completa da API | [src/api/README.md](../../src/api/README.md) |
| Agente e RAG | [AGENT_QUICKSTART.md](AGENT_QUICKSTART.md) |
| Pipeline completo (dados → treino → API) | [END_TO_END_GUIDE.md](END_TO_END_GUIDE.md) |
