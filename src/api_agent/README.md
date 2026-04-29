# Agente Meteorológico — WeatherOps

Módulo de agente conversacional com **Gemini 2.5 Flash** (LangChain), tool calling e contexto histórico via **RAG** (ChromaDB + Google Embeddings). Responde perguntas de previsão em linguagem natural.

Para configurar e testar o agente: [docs/guides/AGENT_QUICKSTART.md](../../docs/guides/AGENT_QUICKSTART.md)

---

## Como funciona

```
Usuário: "Como vai estar Salvador amanhã à tarde?"
         │
         ▼
  POST /v1/agent/chat
         │
         ▼
  Gemini 2.5 Flash ──► seleciona tool ──► get_forecast_by_period(target_date, "tarde")
                                                     │
                                                     ▼
                                          Predictor → modelo TFT
         │
         ▼ (se RAG ativo)
  get_historical_weather_context("temperatura tarde julho")
         │
         ▼
  ChromaDB → chunks históricos → Resposta em PT-BR
```

---

## Tools disponíveis

| Tool | Trigger típico | Parâmetros principais |
|---|---|---|
| `get_forecast_by_period` | "como vai estar a tarde de amanhã?" | `target_date`, `period` |
| `run_weather_forecast` | "previsão para as próximas 72h" | `horizon`, `reference_date`, `model_type` |
| `summarize_dataset_window` | "como estava o clima ontem?" | `reference_date`, `sequence_length` |
| `list_available_models` | "que modelos existem?" | — |
| `get_historical_weather_context` | "como costuma ser julho?" | `query` — **só ativo com RAG** |

**Períodos do dia reconhecidos:**

| Período | Horas |
|---|---|
| `manha` | 07h–12h |
| `tarde` | 12h–17h |
| `noite` | 17h–00h |
| `madrugada` | 00h–07h |

---

## Endpoint

### `POST /v1/agent/chat`

**Request:**
```json
{"message": "Como vai estar a temperatura em Salvador amanhã à tarde?"}
```

**Response:**
```json
{
  "answer": "Amanhã à tarde em Salvador, a temperatura deve ficar entre 28°C e 32°C...",
  "tool_calls": [
    {"name": "get_forecast_by_period", "arguments": {"target_date": "2026-04-29", "period": "tarde"}}
  ],
  "rag_context_snippets": [
    "Em abril em Salvador: temperatura média de 29.1°C..."
  ]
}
```

- `rag_context_snippets` fica vazio se o RAG não estiver ativo
- `tool_calls` lista todas as ferramentas chamadas na execução

**Erros:**

| Código | Causa |
|---|---|
| `400` | `GOOGLE_API_KEY` não configurada |
| `422` | Mensagem vazia ou acima de `AGENT_MAX_MESSAGE_CHARS` (padrão: 8 000) |
| `500` | Falha interna — verificar logs da API |

---

## RAG — Contexto Histórico

O RAG injeta perfis históricos mensais do município (temperatura, umidade, pressão, precipitação), gerados durante o promote.

### Como gerar

```bash
poetry run python -m src.ml_workstation.promotion.run_promote \
    --experiment-name weather_tft_h72 \
    --export-dir src/api/ml_models \
    --update-knowledge-base
```

Após a próxima reinicialização da API, o log exibirá:
```
INFO FeatureStoreRetriever carregado — RAG ativo no agente.
```

### Arquivos gerados

```
src/api/ml_models/weather_forecasting_h72/feature_store/
├── metadata.json          # município, date_range, row_count
├── monthly_profiles.json  # médias mensais por feature
└── seasonal_extremes.json # percentis p5/p95 por mês

src/api_agent/knowledge/chroma_db/  # banco vetorial (não versionado)
```

---

## Graceful Degradation

| Situação | Comportamento |
|---|---|
| Sem `GOOGLE_API_KEY` | Retorna `400` com mensagem clara |
| Sem ChromaDB | Agente funciona sem a tool `get_historical_weather_context` |
| Modelo não carregado | `run_weather_forecast` retorna erro JSON sem derrubar o agente |

---

## Estrutura do Módulo

```
src/api_agent/
├── routers/
│   └── agent.py                # POST /v1/agent/chat
├── tools/
│   ├── forecast_tool.py        # run_weather_forecast
│   ├── period_forecast_tool.py # get_forecast_by_period
│   ├── dataset_tool.py         # summarize_dataset_window
│   ├── models_tool.py          # list_available_models
│   └── historical_context.py   # get_historical_weather_context (RAG)
├── rag/
│   ├── knowledge_builder.py    # constrói ChromaDB a partir do feature store
│   └── retriever.py            # FeatureStoreRetriever
├── knowledge/
│   └── chroma_db/              # gerado pelo promote (não versionado)
├── schemas.py                  # AgentChatRequest / AgentChatResponse
└── service.py                  # orquestração LangChain + Gemini
```
