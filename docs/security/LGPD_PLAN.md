# Plano LGPD — WeatherOps Agent

Sistema: Agente meteorológico ReAct (`POST /v1/agent/chat`)
Lei aplicável: Lei Geral de Proteção de Dados Pessoais — **Lei nº 13.709/2018 (LGPD)**
Autoridade supervisora: **ANPD** (Autoridade Nacional de Proteção de Dados)
Data de elaboração: 2026-04-28

---

## 1. Inventário de Dados

### 1.1 Dados Processados pelo Sistema

| Dado | Categoria | Onde Flui | Persistido? |
|------|-----------|-----------|-------------|
| Texto da query do usuário | Potencialmente pessoal (se contiver PII) | Memória in-process → Gemini API | Não |
| Resultados de forecast (temperatura, etc.) | Não pessoal (dados de sensores ambientais) | Memória → resposta HTTP | Não |
| Dados históricos do dataset (INMET) | Não pessoal | Parquet em disco → memória | Sim (Parquet) |
| Logs de precisão (`accuracy.db`) | Não pessoal (previsão × real) | SQLite em disco | Sim (90 dias) |
| Métricas Prometheus | Não pessoal (contadores, latências) | Prometheus registry | Sim (configurado no Prometheus) |
| Contexto RAG (ChromaDB) | Não pessoal (resumos climáticos mensais) | ChromaDB em disco | Sim |

### 1.2 Situações Excepcionais com Risco PII
O sistema processa dados meteorológicos, não cadastrais. Contudo, um usuário **pode incluir inadvertidamente** dados pessoais na query:
- CPF, e-mail, telefone como identificador de uma localização ou usuário
- Endereços residenciais precisos
- Dados de saúde associados a condições climáticas (ex.: "minha asma piora quando a umidade cai")

O guardrail de PII em [`src/api_agent/guardrails.py`](../../src/api_agent/guardrails.py) bloqueia CPF, e-mail e telefone BR antes de qualquer transmissão ao Gemini API.

---

## 2. Base Legal (Art. 7 LGPD)

**Base aplicada:** Legítimo interesse do controlador (Art. 7, IX) na prestação do serviço de previsão meteorológica para suporte a decisões operacionais e agrícolas.

**Justificativa:** O tratamento dos dados (queries meteorológicas) é necessário para a finalidade legítima de responder às perguntas do usuário. O interesse do controlador não prevalece sobre direitos e liberdades fundamentais do titular, dado que:
1. O sistema não processa dados sensíveis em condições normais de uso.
2. O guardrail impede o armazenamento ou transmissão de PII.
3. Não há elaboração de perfis de usuário (profiling).

**Dados do dataset histórico (INMET):** Base legal de pesquisa científica/meteorológica (Art. 7, IV) — dados de estações meteorológicas públicas, sem identificação de pessoas físicas.

---

## 3. Minimização e Limitação de Finalidade (Art. 6, I e III)

| Princípio | Implementação |
|-----------|--------------|
| **Necessidade** | Queries limitadas a 8.000 caracteres (`agent_max_message_chars`). |
| **Minimização** | Guardrail bloqueia PII antes da transmissão ao LLM externo. |
| **Finalidade** | O agente é instruído via system prompt a responder apenas sobre meteorologia — recusas a perguntas fora do escopo são comportamento esperado. |
| **Não persistência de queries** | Nenhum log de texto de mensagem é gravado em disco. Prometheus armazena apenas contadores e durações. |
| **Transmissão a terceiros** | Apenas ao Gemini API (Google) — ver seção 6 (DPA). Sem transmissão a outros terceiros. |

---

## 4. Retenção e Descarte (Art. 15 LGPD)

| Dado | Prazo de Retenção | Mecanismo de Descarte |
|------|------------------|-----------------------|
| Queries do usuário | **Não persistido** — existe apenas durante o ciclo de vida da requisição HTTP | N/A |
| `accuracy.db` (previsões × real) | 90 dias | `DELETE FROM prediction_log WHERE created_at < datetime('now', '-90 days')` — executar via cron ou ao inicializar o AccuracyEvaluator |
| Logs de aplicação | 30 dias (definir na política de rotação de logs do servidor) | `logrotate` ou equivalente com `compress` + `dateext` |
| ChromaDB (perfis climáticos) | Indefinido — não contém PII | Rebuild automático a cada `run_promote.py --update-knowledge-base` |
| Métricas Prometheus | Definido pela retenção do servidor Prometheus (recomendado: 15 dias) | Configurar `--storage.tsdb.retention.time=15d` no Prometheus |

---

## 5. Direitos do Titular (Art. 18 LGPD)

### 5.1 Situação Normal (sem PII persistida)
Como o sistema **não persiste** o texto das queries, os seguintes direitos são atendidos por design:

| Direito | Resposta |
|---------|----------|
| Acesso (Art. 18, I) | "Não armazenamos o conteúdo das suas consultas." |
| Correção (Art. 18, III) | N/A — dado não armazenado. |
| Eliminação (Art. 18, VI) | N/A — dado não armazenado. |
| Portabilidade (Art. 18, V) | N/A — dado não armazenado. |
| Oposição ao tratamento (Art. 18, IX) | Usuário pode deixar de usar o serviço; nenhum dado residual persiste. |

### 5.2 Situação Excepcional (PII em logs de debug)
Se logs de debug foram habilitados e capturaram texto de queries:
1. Localizar entradas no arquivo de log pelo timestamp da sessão.
2. Anonimizar ou deletar as linhas relevantes.
3. Informar ao titular o prazo de conclusão (até 15 dias corridos, Art. 18 §3º).

---

## 6. Responsabilidades DPA (Controlador × Processador)

| Papel | Entidade | Instrumento |
|-------|----------|-------------|
| **Controlador** | Operador do WeatherOps (organização que hospeda a API) | — |
| **Processador** | Google LLC (Gemini API) | Google Cloud DPA / API Terms of Service |
| **Subprocessador** | Google LLC (Google Embedding API — `models/embedding-001`) | Incluído no Cloud DPA |

**Obrigações do controlador com relação ao Google como processador:**
1. Assinar o **Google Cloud Data Processing Addendum (DPA)** vigente.
2. Verificar e documentar que o projeto Google Cloud utilizado está configurado com **opt-out de uso de dados para treinamento de modelos** (configuração disponível no Google AI Studio / Vertex AI — consultar `x-goog-user-project` e política de uso de dados da API).
3. Manter registro atualizado das atividades de tratamento (ROPA) conforme Art. 37 LGPD.

---

## 7. Resposta a Incidentes (Art. 48 LGPD)

### 7.1 Cenário: Query com PII transmitida ao Gemini antes da ativação dos guardrails

**Quando pode ocorrer:** Deploy sem `guardrails.py` ou com versão defeituosa; rollback acidental.

**Protocolo de resposta:**

| Passo | Ação | Prazo |
|-------|------|-------|
| 1 | Identificar escopo: período de exposição, estimativa de queries afetadas (via logs de acesso HTTP) | Imediato |
| 2 | Notificar DPO interno com relatório preliminar | Até 24h após detecção |
| 3 | Acionar Google DPA/Suporte para verificar política de retenção de dados de API para o tier de conta usado | Até 48h |
| 4 | Notificar ANPD (portal gov.br/anpd) se houver risco relevante aos titulares (Art. 48, §1º) | Até 72h após detecção |
| 5 | Comunicar titulares afetados se identificados, descrevendo o incidente e medidas tomadas | Conforme orientação ANPD |
| 6 | Implementar correção, documentar lições aprendidas | Até 15 dias |

### 7.2 Cenário: Dados históricos INMET comprometidos

Dados são públicos e não contêm PII — classificar como incidente de segurança operacional, sem obrigação de notificação ANPD (Art. 48 não se aplica a dados não pessoais).

---

## 8. Encarregado de Proteção de Dados (DPO)

Designar encarregado conforme Art. 41 LGPD. Contato público deve estar disponível na política de privacidade do serviço.

> **Ação requerida:** Preencher nome e e-mail do DPO e publicar na página do serviço antes de go-live em produção.

---

## Checklist de Conformidade

- [x] Guardrail de PII bloqueia CPF, e-mail e telefone BR na entrada
- [x] Queries não são persistidas em disco
- [x] `accuracy.db` limitada a dados não pessoais (previsão × temperatura real)
- [x] Prometheus armazena apenas métricas agregadas sem conteúdo de mensagens
- [ ] DPA com Google assinado e documentado
- [ ] Opt-out de treinamento de modelo configurado no projeto Google Cloud
- [ ] DPO designado e contato publicado
- [ ] ROPA (Registro de Atividades de Tratamento) elaborado
- [ ] Política de rotação de logs configurada (30 dias)
- [ ] Cron de expurgo do `accuracy.db` configurado (90 dias)
