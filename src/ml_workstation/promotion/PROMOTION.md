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
├── loader.py            # carregamento do modelo em produção
├── run_promote.py       # CLI (entry point)
└── PROMOTION.md         # esta documentação
```

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
```

### Exemplos

```bash
# Seleciona e promove automaticamente o melhor MAPE
poetry run python -m src.ml_workstation.promotion.run_promote \
    --experiment-name weather_forecasting_h72

# Promove um run específico
poetry run python -m src.ml_workstation.promotion.run_promote \
    --experiment-name weather_forecasting_h72 \
    --run-id abc123def456

# Força promoção ignorando proteção de regressão
poetry run python -m src.ml_workstation.promotion.run_promote \
    --experiment-name weather_forecasting_h72 \
    --run-id abc123def456 \
    --force
```

### Códigos de saída

| Código | Significado |
|---|---|
| `0` | Promoção realizada com sucesso |
| `1` | Promoção rejeitada (`PromotionRejectedError`) |
| `2` | Erro inesperado |

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
