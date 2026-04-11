# Model Promotion Pipeline

Módulo responsável por selecionar, validar e promover modelos treinados para produção, com
integração ao MLflow Model Registry e rastreamento local via YAML.

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
  promote_run()              ← verifica regressão de métricas
        │                       registra no Model Registry
        │                       atribui alias "production"
        ▼
  models_production.yaml     ← fonte da verdade (versionada no Git)
        │
        ▼
  load_production_model()    ← usado pela API / inferência
```

Cada horizonte (`h72`, `h168`, `h336`) é uma entrada **independente** no Registry e no YAML —
promover o modelo `h72` não afeta os demais.

---

## Estrutura de Arquivos

```
src/ml_workstation/promotion/
├── __init__.py          # exporta todos os símbolos públicos
├── promote.py           # lógica de seleção e promoção
├── loader.py            # carregamento do modelo em produção
├── run_promote.py       # CLI (entry point)
└── PROMOTION.md         # esta documentação

src/ml_workstation/
└── models_production.yaml   # fonte de verdade com estado atual de produção
```

---

## Fonte de Verdade — `models_production.yaml`

Arquivo YAML versionado no Git. Armazena o estado de produção para cada experimento.

```yaml
experiments:
  weather_forecasting_h72:
    model_name: weather_forecasting_h72   # nome no MLflow Model Registry
    run_id: <run_id>                      # run promovido
    mape: 0.0842                          # MAPE do run promovido
    promoted_at: "2026-04-11"            # data da promoção
    promoted_by: auto                     # "auto" (por métrica) ou "manual" (por run_id)
  weather_forecasting_h168:
    ...
  weather_forecasting_h336:
    ...
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

Registra o run no MLflow Model Registry, atribui o alias `production` e atualiza
`models_production.yaml`.

| Parâmetro | Descrição |
|---|---|
| `run_id` | ID do run a promover |
| `experiment_name` | Nome do experimento MLflow |
| `model_name` | Nome no Registry (padrão: `experiment_name`) |
| `tracking_uri` | URI do MLflow (auto-detectado se `None`) |
| `force` | Permite promover mesmo que o MAPE seja pior que o atual |

Lança `PromotionRejectedError` se o candidato tiver MAPE pior que o modelo vigente em
produção e `force=False`.

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

### `load_production_model(experiment_name, tracking_uri=None, device="cpu") -> torch.nn.Module`

Carrega o modelo em produção para inferência. Tenta o Model Registry primeiro (alias
`production`); se falhar, usa o `runs:/` URI registrado no YAML como fallback.

```python
from src.ml_workstation.promotion import load_production_model

model = load_production_model("weather_forecasting_h72", device="cpu")
model.eval()
```

Lança `ModelNotInProductionError` se não houver modelo promovido ou se ambas as fontes
falharem.

---

### `get_production_info(experiment_name) -> dict`

Retorna os metadados do modelo atualmente em produção.

```python
from src.ml_workstation.promotion import get_production_info

info = get_production_info("weather_forecasting_h72")
# {'model_name': '...', 'run_id': '...', 'mape': 0.0842, 'promoted_at': '2026-04-11', ...}
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

Antes de promover, `promote_run` compara o MAPE do candidato com o MAPE do modelo
atualmente em produção (lido do YAML). Se o candidato for **pior** (MAPE maior), a promoção
é bloqueada:

```
PromotionRejectedError: Promoção rejeitada: candidato MAPE=0.1200 é pior que o atual
em produção MAPE=0.0842 (delta=+0.0358). Use --force para sobrescrever.
```

Use `force=True` / `--force` para sobrescrever essa proteção quando necessário
(ex.: re-treinamento com dados novos que ainda não convergiu completamente).

---

## Exceções

| Exceção | Módulo | Quando é lançada |
|---|---|---|
| `PromotionRejectedError` | `promote.py` | Candidato tem MAPE pior e `force=False` |
| `ModelNotInProductionError` | `loader.py` | Nenhum modelo promovido ou falha de carregamento |
| `ValueError` | `promote.py` | Experimento MLflow não encontrado ou sem runs finalizados |
