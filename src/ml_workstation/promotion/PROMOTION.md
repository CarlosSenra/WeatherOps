# Model Promotion Pipeline

Módulo responsável por selecionar, validar e promover modelos treinados para produção, com
integração ao **MLflow Model Registry** como única fonte de verdade sobre o estado de produção.

---

## Visão Geral

```
Experimentos MLflow
  (h72 | h168 | h336)
        │
        ▼
  select_best_run()          ← ranking por menor MAPE (ou outra métrica)
        │
        ▼
  promote_run()              ← verifica regressão de métricas (via Registry tags)
        │                       registra no Model Registry
        │                       atribui alias "production"
        │                       grava metadados como model version tags
        ▼
  MLflow Model Registry      ← ÚNICA fonte de verdade (alias + tags)
        │
        ▼
  load_production_model()    ← usado pela API / inferência
```

Cada horizonte (`h72`, `h168`, `h336`) é uma entrada **independente** no Registry —
promover o modelo `h72` não afeta os demais.

---

## Fonte de Verdade — MLflow Model Registry

O estado de produção é armazenado **exclusivamente** no MLflow Model Registry.
Não existe mais o `models_production.yaml`.

Cada model version promovida recebe as seguintes **tags**:

| Tag | Descrição |
|---|---|
| `mape` | MAPE do run no momento da promoção (`str(float)` ou `""` se ausente) |
| `promoted_at` | Data ISO da promoção (ex.: `"2026-04-14"`) |
| `promoted_by` | `"auto"` (seleção por métrica) ou `"manual"` (run_id explícito) |
| `experiment_name` | Nome do experimento MLflow para rastreabilidade reversa |
| `run_id` | Run MLflow de origem (usado para carregar artefatos como `scaler.pkl`) |

O alias `production` aponta para a model version ativa em cada modelo registrado.

---

## Estrutura de Arquivos

```
src/ml_workstation/promotion/
├── __init__.py          # exporta todos os símbolos públicos
├── promote.py           # lógica de seleção e promoção
├── export_local.py      # exportação do artefato `model` para disco (ex.: src/api/ml_models)
├── loader.py            # carregamento do modelo em produção
├── run_promote.py       # CLI (entry point)
└── PROMOTION.md         # esta documentação
```

### Exportação para disco (`--export-dir`)

Depois de uma promoção **bem-sucedida** no Registry, podes opcionalmente copiar o artefato MLflow `model` (PyTorch flavour) para um diretório no disco. Isso gera pastas prontas para montar na API de serving (por exemplo `src/api/ml_models`) sem expor o `mlruns` completo.

**Como usar**

| Forma | Descrição |
|---|---|
| `--export-dir DIR` | Passa o diretório **base** na linha de comando. Se for relativo, é resolvido pela raiz do repositório. Cada modelo promovido passa a existir em `DIR/<nome_no_registry>/` (o nome no Registry é o `experiment_name` ou o `--model-name` se o usares). |
| `WEATHEROPS_EXPORT_MODELS_DIR` | Variável de ambiente com o mesmo efeito que `--export-dir`. Se ambos estiverem definidos, **a CLI ganha** (`--export-dir` tem prioridade). |
| `--strict-export / --no-strict-export` | Em modo estrito (default), falha o comando se o export local não gerar artefatos válidos (`MLmodel` + `data/`). |

O diretório base deve existir ou ser criável; as subpastas por modelo são substituídas em cada exportação após validação da estrutura exportada.

**Conteúdo gerado**

- Pasta do modelo no formato MLflow PyTorch (`MLmodel`, `data`, etc.).
- `manifest.json` na raiz dessa pasta: `registry_version`, `run_id`, `mape`, `promoted_at`, `experiment_name`, etc.

Em modo estrito (default), se a exportação falhar (I/O, permissões ou layout incompleto), o comando retorna erro para evitar estado local inconsistente. Usa `--no-strict-export` para manter o comportamento permissivo.

**Exemplos (a partir da raiz do repositório `WeatherOps/`)**

```bash
# Promover o melhor run e exportar para a pasta usada pelo Docker da API
poetry run python -m src.ml_workstation.promotion.run_promote \
    --experiment-name weather_tft_h72 \
    --export-dir src/api/ml_models

# Equivalente sem flag (útil em CI): exportar para o mesmo destino
export WEATHEROPS_EXPORT_MODELS_DIR=src/api/ml_models
poetry run python -m src.ml_workstation.promotion.run_promote \
    --experiment-name weather_tft_h72

# Promover um run explícito com exportação
poetry run python -m src.ml_workstation.promotion.run_promote \
    --experiment-name weather_tft_h72 \
    --run-id abc123def456 \
    --export-dir src/api/ml_models
```

**API de serving**

Na API, define `WEATHEROPS_MODEL_ROOT=/app/ml_models` e monta o volume (ver `src/api/docker-compose.yml`). A inferência tenta carregar desse diretório primeiro; se a pasta do modelo não existir, usa o Model Registry como antes.

**Python** — o parâmetro `export_dir` em `promote_run` / `promote_best` aceita `str` ou `Path` e corresponde ao mesmo destino base.

---

## API Pública

### `select_best_run(experiment_name, metric="mape", tracking_uri=None) -> Run`

Busca o run com menor valor de `metric` entre os runs finalizados do experimento.

```python
from src.ml_workstation.promotion import select_best_run

run = select_best_run("weather_forecasting_h72")
print(run.info.run_id, run.data.metrics["mape"])
```

---

### `promote_run(run_id, experiment_name, model_name=None, tracking_uri=None, force=False) -> str`

Registra o run no MLflow Model Registry, atribui o alias `production` e grava
metadados como tags na model version.

| Parâmetro | Descrição |
|---|---|
| `run_id` | ID do run a promover |
| `experiment_name` | Nome do experimento MLflow |
| `model_name` | Nome no Registry (padrão: `experiment_name`) |
| `tracking_uri` | URI do MLflow (auto-detectado se `None`) |
| `force` | Permite promover mesmo que o MAPE seja pior que o atual |

Lança `PromotionRejectedError` se o candidato tiver MAPE pior que o modelo vigente em
produção (lido das tags da version atual) e `force=False`.

```python
from src.ml_workstation.promotion import promote_run

version = promote_run(
    run_id="abc123",
    experiment_name="weather_forecasting_h72",
)
print(f"Versão promovida: {version}")
```

---

### `promote_best(experiment_name, metric="mape", model_name=None, tracking_uri=None, force=False) -> str`

Combina `select_best_run` + `promote_run` em uma única chamada.

```python
from src.ml_workstation.promotion import promote_best

version = promote_best("weather_forecasting_h72")
```

---

### `load_production_model(experiment_name, tracking_uri=None, device="cpu", model_name=None) -> torch.nn.Module`

Carrega o modelo em produção via alias `production` no MLflow Model Registry.

```python
from src.ml_workstation.promotion import load_production_model

model = load_production_model("weather_forecasting_h72", device="cpu")
model.eval()
```

Lança `ModelNotInProductionError` se não houver alias `production` no Registry ou se
o carregamento falhar.

---

### `get_production_info(experiment_name, tracking_uri=None, model_name=None) -> dict`

Retorna os metadados do modelo atualmente em produção lidos das tags da model version.

```python
from src.ml_workstation.promotion import get_production_info

info = get_production_info("weather_forecasting_h72")
# {
#   'model_name': 'weather_forecasting_h72',
#   'version': '3',
#   'run_id': 'abc123...',
#   'mape': 4.80,
#   'promoted_at': '2026-04-14',
#   'promoted_by': 'auto',
#   'experiment_name': 'weather_forecasting_h72',
# }
```

---

### `load_production_scaler(experiment_name, tracking_uri=None, model_name=None) -> StandardScaler`

Carrega o `StandardScaler` do run promovido em produção.

Aplicável **apenas** a modelos **LSTM e Transformer**, que normalizam externamente via
`StandardScaler`. Modelos **TFT e N-BEATS** incorporam normalização internamente via
`GroupNormalizer` do pytorch-forecasting e não requerem scaler externo.

```python
from src.ml_workstation.promotion import load_production_scaler

scaler = load_production_scaler("weather_forecasting_h72")
```

---

## CLI

```
usage: python -m src.ml_workstation.promotion.run_promote
       --experiment-name EXPERIMENT_NAME
       [--run-id RUN_ID]
       [--metric METRIC]
       [--model-name MODEL_NAME]
       [--tracking-uri TRACKING_URI]
       [--force]
       [--export-dir DIR]
       [--strict-export | --no-strict-export]
       [--update-knowledge-base]
       [--knowledge-base-path DIR]
```

### Exemplos

```bash
# Seleciona e promove automaticamente o melhor MAPE (deve ser melhorado para o TFT e Nbeast)
poetry run python -m src.ml_workstation.promotion.run_promote \
    --experiment-name weather_forecasting_h72 \
    --export-dir src/api/ml_models

# Promove um run específico
poetry run python -m src.ml_workstation.promotion.run_promote \
    --experiment-name weather_forecasting_h72 \
    --run-id abc123def456 \
    --export-dir src/api/ml_models

# Força promoção ignorando proteção de regressão
poetry run python -m src.ml_workstation.promotion.run_promote \
    --experiment-name weather_forecasting_h72 \
    --run-id abc123def456 \
    --force \
    --export-dir src/api/ml_models

# Promove + exporta + atualiza base vetorial do agente (RAG)
poetry run python -m src.ml_workstation.promotion.run_promote \
    --experiment-name weather_tft_h72 \
    --export-dir src/api/ml_models \
    --update-knowledge-base
```

### Códigos de saída

| Código | Significado |
|---|---|
| `0` | Promoção realizada com sucesso |
| `1` | Promoção rejeitada (`PromotionRejectedError`) |
| `2` | Erro inesperado |

---

## Atualizar a Base Vetorial do Agente (RAG)

O flag `--update-knowledge-base` ativa a geração automática do **feature store** e da
**base vetorial ChromaDB** usada pelo agente conversacional para responder perguntas
sobre padrões históricos de clima.

### O que acontece

```
promote --update-knowledge-base
    │
    ├─ 1. Feature store snapshot
    │       Lê o Parquet de serving exportado em serving_data/
    │       → detecta o município pelo diretório pai do source_file
    │       → calcula perfis mensais (média, min, max, std) e percentis (p5/p95)
    │       → salva em <model_dir>/feature_store/
    │           ├── metadata.json
    │           ├── monthly_profiles.json
    │           └── seasonal_extremes.json
    │
    └─ 2. Rebuild do ChromaDB
            Converte os JSONs em chunks de texto em PT-BR
            → embeda com Google Embedding (models/embedding-001)
            → persiste em src/api_agent/knowledge/chroma_db/
            (idempotente: remove documentos do município antes de reinserir)
```

### Pré-requisitos

- `--export-dir` deve estar definido e o promote deve ter gerado `serving_data/`
- `GOOGLE_API_KEY` disponível no ambiente
- Dependências do grupo `agent` instaladas: `poetry install --only agent`

### Flags

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--update-knowledge-base` | `False` | Ativa a geração do feature store + rebuild do ChromaDB |
| `--knowledge-base-path DIR` | `src/api_agent/knowledge/chroma_db` | Diretório de persistência do ChromaDB. Também lido de `KNOWLEDGE_BASE_PATH` |

### Exemplo completo

```bash
GOOGLE_API_KEY=AIza... \
poetry run python -m src.ml_workstation.promotion.run_promote \
    --experiment-name weather_tft_h72 \
    --export-dir src/api/ml_models \
    --update-knowledge-base

# Saída esperada:
# Melhor run promovido com sucesso.
#   Experimento : weather_tft_h72
#   Métrica     : mape
#   Versão      : 4
# Base vetorial atualizada: 13 chunks em .../src/api_agent/knowledge/chroma_db
```

### Reiniciar a API

Após gerar o ChromaDB, reinicie a API para que o `FeatureStoreRetriever` seja carregado:

```bash
# Log esperado no startup:
# INFO  src.api.main — FeatureStoreRetriever carregado — RAG ativo no agente.
```

### Graceful degradation

- Se `GOOGLE_API_KEY` não estiver definida: o flag é ignorado com aviso no log
- Se `serving_data/` não existir no modelo exportado: base vetorial não é gerada (aviso no log)
- Se o ChromaDB não existir quando a API sobe: o agente funciona normalmente, sem a tool `get_historical_weather_context`

---

## Proteção contra Regressão de Métricas

Antes de promover, `promote_run` consulta o MLflow Model Registry para obter o MAPE
da versão atualmente em produção (via tag `mape` da model version com alias `production`).
Se o candidato for **pior** (MAPE maior), a promoção é bloqueada:

```
PromotionRejectedError: Promoção rejeitada: candidato MAPE=0.1200 é pior que o atual
em produção MAPE=0.0842 (delta=+0.0358). Use --force para sobrescrever.
```

A guarda é inerte nos seguintes casos (promoção prossegue normalmente):
- Primeira promoção: nenhum alias `production` existe ainda no Registry.
- Run sem métrica `mape` registrada: guarda incompleta não bloqueia.

Use `force=True` / `--force` para sobrescrever a proteção quando necessário.

---

## Exceções

| Exceção | Módulo | Quando é lançada |
|---|---|---|
| `PromotionRejectedError` | `promote.py` | Candidato tem MAPE pior e `force=False` |
| `ModelNotInProductionError` | `loader.py` | Alias `production` ausente no Registry ou falha de carregamento |
| `ValueError` | `promote.py` | Experimento MLflow não encontrado ou sem runs finalizados |
