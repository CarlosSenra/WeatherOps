# Explicabilidade e Fairness — WeatherOps Agent

Sistema: Agente meteorológico ReAct (`POST /v1/agent/chat`)
Data: 2026-04-28

---

## 1. Lógica de Seleção de Tool

O agente usa Gemini 2.5 Flash com LangChain function calling. A escolha da tool é guiada pelo `_SYSTEM_PROMPT` (definido em [`src/api_agent/service.py`](../../src/api_agent/service.py)) combinado com o raciocínio do próprio LLM. O decision tree documentado abaixo reflete as regras explícitas no prompt — o LLM pode combinar tools ou desviar em casos ambíguos.

### Decision Tree das Tools

```
Query do usuário
│
├─ Menciona período do dia (manhã, tarde, noite, madrugada)?
│   └─ SIM → get_forecast_by_period(target_date, period)
│       • Calcula horizonte automaticamente (72h)
│       • Filtra horas certas: manhã 07-12h, tarde 12-17h, noite 17-24h, madrugada 00-07h
│       • Retorna: min/max/avg temp, pico horário, breakdown por hora
│
├─ Especifica horizonte explícito (72h / 168h / 336h)?
│   └─ SIM → run_weather_forecast(horizon, reference_date, model_type="tft")
│       • Chama o modelo TFT para o horizonte solicitado
│       • Retorna: MAPE, latência, amostra das primeiras 12 previsões
│
├─ Pergunta sobre clima típico / sazonalidade / histórico mensal?
│   └─ SIM → get_historical_weather_context(query)
│       • Busca semântica no ChromaDB (top-3 chunks)
│       • Chunks contêm perfis mensais: temperatura média/min/max, umidade, pressão, precipitação
│       • Retorna: texto em português com contexto sazonal
│
├─ Pergunta sobre dados históricos observados (séries temporais reais)?
│   └─ SIM → summarize_dataset_window(reference_date, sequence_length)
│       • Lê janela do Parquet (últimas N horas antes de reference_date)
│       • Retorna: estatísticas (min/max/mean) por coluna + amostra de linhas
│
└─ Pergunta sobre modelos disponíveis, horizontes, opções?
    └─ SIM → list_available_models()
        • Retorna: horizons [72, 168, 336], model_types ["tft"], configurações de sequência
```

### Combinação de Tools
O agente pode chamar múltiplas tools em sequência. Exemplo típico: query sobre "previsão da tarde de amanhã considerando o histórico de julho" → `get_forecast_by_period` + `get_historical_weather_context`.

### Fronteira e Limitações
- O agente **não tem acesso a dados em tempo real** (nowcasting) — apenas previsões baseadas no modelo TFT treinado.
- Perguntas sobre locais fora do dataset (outras cidades) podem receber resposta com dados do único município configurado, sem aviso explícito.
- O Gemini pode invocar a tool errada em queries ambíguas — o campo `tool_calls` na resposta permite que o consumidor da API audite a decisão.

---

## 2. Transparência da Cadeia de Raciocínio

Todo campo de `AgentChatResponse` ([`src/api_agent/schemas.py`](../../src/api_agent/schemas.py)) contribui para a transparência:

| Campo | Conteúdo | Uso para explicabilidade |
|-------|----------|------------------------|
| `answer` | Resposta final em português | Texto legível com a conclusão do agente |
| `tool_calls` | `[{name, arguments}, ...]` | Lista todas as tools invocadas e com quais argumentos — permite auditoria da cadeia de raciocínio |
| `rag_context_snippets` | Chunks recuperados do ChromaDB | Expõe a base de conhecimento histórico usada — permite verificar se a resposta é fundamentada |

**Exemplo de resposta com transparência total:**
```json
{
  "answer": "Na tarde de amanhã (12h-17h), a temperatura prevista em Salvador será entre 28°C e 31°C, com pico às 14h.",
  "tool_calls": [
    {"name": "get_forecast_by_period", "arguments": {"target_date": "2026-04-29", "period": "tarde"}}
  ],
  "rag_context_snippets": []
}
```

O consumidor da API pode verificar: qual tool foi usada, com quais argumentos, e qual contexto histórico (se algum) embasou a resposta.

---

## 3. Fairness Temporal — Cobertura do Knowledge Base

O ChromaDB é populado por [`src/api_agent/rag/knowledge_builder.py`](../../src/api_agent/rag/knowledge_builder.py) com:
- **12 chunks mensais** (um por mês, janeiro–dezembro) — contendo temperatura média/min/max/desvio, percentis p5/p95, umidade, pressão e precipitação.
- **1 chunk anual** — resumo com mês mais quente, mais frio e intervalo de datas do dataset.

**Distribuição uniforme:** todos os 12 meses são igualmente representados no knowledge base. Não há viés de cobertura temporal no RAG (ex.: meses de verão vs. inverno recebem o mesmo número de chunks).

**Limitação conhecida:** Os perfis refletem apenas os dados históricos disponíveis no dataset INMET. Se o dataset não cobrir um determinado mês (ex.: dados faltantes), o chunk correspondente conterá estatísticas incompletas — isso é documentado no campo `date_range` do `metadata.json` gerado por `feature_store.py`.

---

## 4. Fairness por Horizonte de Previsão

O `AccuracyEvaluator` ([`src/api/services/accuracy_evaluator.py`](../../src/api/services/accuracy_evaluator.py)) avalia a qualidade do modelo em 3 buckets de horizonte:

| Bucket | Horas | Expectativa de Erro |
|--------|-------|---------------------|
| `near` | h1–h24 | MAE menor — próximo ao estado atual |
| `mid` | h25–h72 | MAE intermediário |
| `far` | h73+ | MAE maior — incerteza acumula com o horizonte |

**Princípio de fairness:** A degradação de qualidade em horizontes longos é **esperada e documentada**, não ocultada. O endpoint `/metrics` expõe `weatherops_model_mape{model_key, bucket}` publicamente. O dashboard Grafana ([`docs/grafana_dashboard.json`](../grafana_dashboard.json)) exibe todos os buckets lado a lado.

**Implicação para o usuário:** Ao solicitar uma previsão de 336h (14 dias), o agente retorna os dados com `mape` como parte do payload do tool `run_weather_forecast`. O agente pode (e deve, conforme ajuste futuro do system prompt) comunicar ao usuário o nível de incerteza esperado para horizontes longos.

---

## 5. Limitações Conhecidas e Riscos Residuais

| Limitação | Impacto | Mitigação Atual |
|-----------|---------|-----------------|
| Max 5 iterações no loop ReAct | Queries muito complexas podem ser truncadas | Limite documentado; `AgentRuntimeError` retorna mensagem explicativa |
| Gemini não é especializado em meteorologia | Pode fabricar dados se tool não for chamada | Campo `tool_calls` permite verificar se tools foram usadas; RAGAS `faithfulness` monitora |
| Dataset cobre apenas Salvador (BA) | Queries sobre outras cidades retornam dados do município configurado sem aviso | Limitação documentada aqui e no `src/api_agent/README.md` |
| RAG opcional (ChromaDB pode estar ausente) | Agente responde sem contexto histórico sazonal | Campo `rag_context_snippets: []` sinaliza ausência; resposta degrada graciosamente |
| Embeddings Google (models/embedding-001) | Dependência de serviço externo para build do knowledge base | Knowledge base é pré-compilado offline — a indisponibilidade temporária do embedding API não afeta o serving |
| Padrões de guardrail são regex fixos | False negatives para ataques em outros idiomas ou com ofuscação | Cobertura atual: PT/EN. Expandir para ES e ofuscação unicode em versões futuras |

---

## 6. Auditabilidade

Todos os artefatos de avaliação e segurança podem ser rastreados:

| Artefato | Localização | Frequência |
|----------|-------------|-----------|
| Scores RAGAS | `evaluation_results/ragas_results_*.json` | Sob demanda via `scripts/run_ragas_eval.py` |
| Métricas de acurácia ML | `accuracy.db` + Prometheus | Contínuo (background worker) |
| Bloqueios de guardrail | Prometheus `weatherops_agent_guardrail_blocks_total` + logs | Por requisição |
| Drift semântico | `weatherops_query_semantic_drift_score` (Prometheus) | Sob demanda via `scripts/check_query_drift.py --semantic` |
| Cadeia de raciocínio do agente | `AgentChatResponse.tool_calls` | Por requisição (retornado na API response) |
