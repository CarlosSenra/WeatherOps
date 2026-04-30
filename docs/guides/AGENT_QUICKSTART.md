# Agente WeatherOps — Guia de Início Rápido

Tempo estimado: **10 minutos** (agente básico) · **+5 minutos** (com RAG)

> **Pré-condição:** a API WeatherOps já deve estar rodando e com pelo menos um modelo
> carregado. Se ainda não fez isso, siga o [QUICKSTART.md](QUICKSTART.md) primeiro.

---

## 1. Obter a chave da API Google

O agente usa **Gemini 2.5 Flash** via Google AI. A chave é gratuita para uso pessoal:

1. Acesse [aistudio.google.com](https://aistudio.google.com)
2. Clique em **"Get API key"** → **"Create API key"**
3. Copie a chave gerada (começa com `AIza...`)

---

## 2. Instalar as dependências do agente

As dependências do agente são opcionais e ficam em um grupo separado no `pyproject.toml`:

```bash
# Instalar apenas API + agente (sem ferramentas de treinamento)
poetry install --only api,agent

# Ou, se já instalou tudo antes, apenas adicionar o grupo agent
poetry install --with agent
```

Pacotes instalados pelo grupo `agent`:
- `langchain-google-genai` — integração LangChain com Gemini
- `langchain-chroma` + `chromadb` — banco vetorial para RAG
- `ragas` + `datasets` — avaliação offline (opcional)

---

## 3. Configurar a chave no `.env`

```bash
# Copiar o template se ainda não tiver o .env
cp .env.example .env

# Adicionar a chave (edite o arquivo .env diretamente ou use o comando abaixo)
echo "GOOGLE_API_KEY=AIza...sua_chave_aqui" >> .env
```

Verifique que a linha ficou correta em `.env`:
```env
GOOGLE_API_KEY=AIzaSy...
```

---

## 4. (Re)iniciar a API

Se a API já estava rodando, reinicie para que ela leia a nova variável:

```bash
docker compose -f src/api/docker-compose.yml --profile api up --build -d
```

Confirme que o agente foi carregado no log de startup:
```
INFO     src.api.main — Inicialização da API WeatherOps concluída.
```

(Se o `GOOGLE_API_KEY` não estiver definido, o agente ainda inicia mas retorna 400
nas chamadas — isso é esperado e não impede os endpoints de forecast normais.)

---

## 5. Testar o agente (sem RAG)

```bash
curl -s -X POST http://localhost:8888/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Como vai estar a temperatura amanhã à tarde?"}' \
  | python -m json.tool
```

Resposta esperada:
```json
{
  "answer": "Amanhã à tarde, a temperatura em Salvador deve ficar entre 28°C e 32°C...",
  "tool_calls": [
    {"name": "get_forecast_by_period", "arguments": {"target_date": "...", "period": "tarde"}}
  ],
  "rag_context_snippets": []
}
```

`rag_context_snippets` está vazio — o RAG ainda não foi ativado. Isso é normal.

### Outras perguntas para testar

```bash
# Horizonte explícito (usa run_weather_forecast)
curl -X POST http://localhost:8888/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Previsão para as próximas 72 horas"}'

# Dados históricos observados
curl -X POST http://localhost:8888/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Como estava o clima nas últimas 24 horas?"}'

# Listar modelos disponíveis
curl -X POST http://localhost:8888/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Que horizontes de previsão estão disponíveis?"}'
```

---

## 6. Ativar o RAG (contexto histórico)

O RAG enriquece respostas sobre padrões sazonais com dados históricos reais do município.

### Pré-requisito

- `GOOGLE_API_KEY` configurada no ambiente (passo 3)

### Quick start — gerar a base vetorial (repositório clonado)

O repositório já inclui o feature store do município. Para popular o ChromaDB a partir
dele basta um comando:

```bash
GOOGLE_API_KEY=<SUA_API_KEY> poetry run python scripts/update_kb.py
```

Saída esperada:
```
Construindo knowledge base...
OK — 13 documentos inseridos em src/api_agent/knowledge/chroma_db
```

### Após promover um novo modelo

Se você re-treinou e promoveu um modelo com `run_promote`, use a flag
`--update-knowledge-base` para regenerar a base vetorial com os novos perfis:

```bash
poetry run python -m src.ml_workstation.promotion.run_promote \
    --experiment-name weather_tft_h72 \
    --export-dir src/api/ml_models \
    --update-knowledge-base
```

### Verificar os arquivos gerados

```bash
# Feature store (JSONs com perfis do município)
ls src/api/ml_models/weather_forecasting_h72/feature_store/
# → metadata.json  monthly_profiles.json  seasonal_extremes.json

# Banco vetorial ChromaDB
ls src/api_agent/knowledge/chroma_db/
# → chroma.sqlite3  ...
```

### Reiniciar a API para carregar o RAG

```bash
docker compose -f src/api/docker-compose.yml --profile api up --build -d
```

Log esperado no startup:
```
INFO     src.api.main — FeatureStoreRetriever carregado — RAG ativo no agente.
```

---

## 7. Testar o agente com RAG

Pergunte algo que acione o contexto histórico:

```bash
curl -s -X POST http://localhost:8888/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Como costuma ser o clima em Salvador em julho? É o mês mais frio?"}' \
  | python -m json.tool
```

Resposta com RAG ativo:
```json
{
  "answer": "Julho em Salvador é realmente o mês mais ameno do ano. Historicamente, a temperatura média fica em torno de 24.5°C, com mínimas de 21°C e máximas de 28°C...",
  "tool_calls": [
    {"name": "get_historical_weather_context", "arguments": {"query": "temperatura julho Salvador mês mais frio"}}
  ],
  "rag_context_snippets": [
    "Em julho em salvador: Temperatura média de 24.5°C (mín: 21.1°C, máx: 28.3°C, desvio: 2.1°C). Percentil 5%: 19.8°C, percentil 95%: 27.9°C. Umidade relativa média de 72.3%...",
    "Resumo climático anual de salvador: mês mais quente é fevereiro (média 29.8°C), mês mais frio é julho (média 24.5°C)..."
  ]
}
```

---

## 8. Smoke test automatizado

O script `scripts/smoke_api.py` inclui testes de agente. Para executar com o agente:

```bash
# Variável SMOKE_SKIP_AGENT controla se o agente é testado
WEATHEROPS_API_BASE_URL=http://localhost:8888 \
SMOKE_SKIP_AGENT=false \
poetry run python scripts/smoke_api.py
```

---

## 9. Avaliação de Qualidade

### 9a. RAGAS — qualidade do agente

Avalia o agente contra o golden set e publica os scores no Grafana:

```bash
GOOGLE_API_KEY=sua_chave \
  poetry run python scripts/run_ragas_eval.py \
    --limit 5 \
    --update-prometheus
```

Métricas calculadas: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.

Resultados salvos em `evaluation_results/ragas_results_<timestamp>.json`.

Após rodar, verifique o painel **RAGAS Scores** no Grafana

### 9b. Drift semântico

Mede se as queries reais estão se afastando do golden set de referência e publica no Grafana:

```bash
GOOGLE_API_KEY=sua_chave \
  poetry run python scripts/check_query_drift.py \
    --semantic \
    --update-prometheus
```

Após rodar, verifique o painel **Drift Semântico de Queries** no Grafana

### 9c. Popular painéis manualmente (sem rodar scripts)

Útil para demonstrar os painéis sem aguardar a avaliação completa:

```bash
# RAGAS scores
curl -s -X POST http://localhost:8888/v1/internal/ragas \
  -H "Content-Type: application/json" \
  -d '{"scores": {"faithfulness": 0.85, "answer_relevancy": 0.91, "context_precision": 0.78, "context_recall": 0.82}}' \
  | python -m json.tool

# Drift semântico
curl -s -X POST http://localhost:8888/v1/internal/drift \
  -H "Content-Type: application/json" \
  -d '{"score": 0.08}' \
  | python -m json.tool
```

---

## Troubleshooting

| Sintoma | Causa provável | Solução |
|---------|---------------|---------|
| `400 GOOGLE_API_KEY não configurada` | Variável ausente ou API não reiniciada | Adicione ao `.env` e reinicie |
| `rag_context_snippets` sempre vazio | ChromaDB não gerado | Execute `poetry run python scripts/update_kb.py` |
| `ImportError: No module named 'chromadb'` | Grupo `agent` não instalado | `poetry install --with agent` |
| `Router /v1/agent omitido` no log | Deps do agente ausentes | `poetry install --with agent` e reiniciar |
| `422 Mensagem excede agent_max_message_chars` | Mensagem muito longa | Reduza ou aumente `AGENT_MAX_MESSAGE_CHARS` no `.env` |
| RAG ativo mas chunks irrelevantes | ChromaDB desatualizado | Re-execute o promote com `--update-knowledge-base` |
| `FeatureStoreRetriever: ChromaDB não encontrado` no log | Base vetorial ausente | Normal — o agente funciona sem RAG |

---

## Variáveis de Ambiente do Agente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `GOOGLE_API_KEY` | — | **Obrigatório.** Chave da API Google AI |
| `AGENT_MAX_MESSAGE_CHARS` | `8000` | Tamanho máximo da mensagem de entrada |
| `KNOWLEDGE_BASE_PATH` | `src/api_agent/knowledge/chroma_db` | Caminho do ChromaDB para RAG |

---

## Próximos Passos

- [src/api_agent/README.md](../../src/api_agent/README.md) — arquitetura detalhada do agente
- [PROMOTION.md](../../src/ml_workstation/promotion/PROMOTION.md) — promote com RAG
- [QUICKSTART.md](QUICKSTART.md) — endpoints de forecast sem agente
