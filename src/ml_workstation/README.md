# ML Workstation

Módulo de treinamento de modelos de séries temporais para previsão meteorológica usando PyTorch + MLflow.

---

## Estrutura

```
src/ml_workstation/
├── Dockerfile
├── docker-compose.yml
│
├── config/
│   └── training_config.py       # DataConfig, ModelConfig, TrainingConfig (Pydantic)
│
├── data/
│   ├── dataset.py               # WeatherSequenceDataset
│   └── loader.py                # ParquetDataLoader — lê, normaliza e divide os dados
│
├── models/
│   ├── interface.py             # ITimeSeriesModel (ABC)
│   ├── lstm.py                  # WeatherLSTM
│   └── transformer.py           # WeatherTransformer
│
├── training/
│   ├── metrics.py               # compute_metrics() — MAE, RMSE, MAPE
│   └── trainer.py               # Trainer — loop, early stopping, checkpointing
│
├── tracking/
│   └── mlflow_tracker.py        # MLflowTracker — wrapper MLflow
│
├── train.py                     # Entrypoint de treinamento
│
├── experiments/                 # Configs JSON dos runs (volume)
│   ├── lstm_baseline.json
│   └── transformer_baseline.json
│
├── artifacts/                   # Checkpoints do melhor modelo (volume)
└── mlruns/                      # Experimentos MLflow (volume)
```

---

## Volumes

| Host (`src/ml_workstation/`) | Container | Conteúdo |
|---|---|---|
| `../../data/spec` | `/app/data/spec` | Parquet de entrada — somente leitura |
| `./artifacts` | `/app/artifacts` | Checkpoints `.pt` do melhor modelo |
| `./mlruns` | `/app/mlruns` | Runs e métricas do MLflow |
| `./experiments` | `/app/experiments` | Arquivos JSON de configuração |

---

## Como usar

Todos os comandos a partir de `src/ml_workstation/`.

### Build

```bash
docker compose --profile train build
```

> **Atenção:** o serviço `trainer` usa o profile `train`. Sem `--profile train` o Docker não encontra serviços para build.

### Treinar

```bash
# Smoke test — 3 épocas, rápido para validar o setup
docker compose --profile train run --rm trainer --config //app/experiments/lstm_smoke_test.json

# LSTM baseline — 50 épocas
docker compose --profile train run --rm trainer --config //app/experiments/lstm_baseline.json

# Transformer baseline
docker compose --profile train run --rm trainer --config //app/experiments/transformer_baseline.json
```

> **Atenção (Git Bash / Windows):** use `//app/...` (duas barras) ao passar caminhos do container.
> O Git Bash expande `/app/...` para `C:/Program Files/Git/app/...` causando `FileNotFoundError`.

### Visualizar experimentos

Em outro terminal:

```bash
docker compose --profile ui up mlflow-ui
```

Acesse `http://localhost:5000` no navegador. Os runs ficam salvos em `mlruns/` e persistem entre execuções.

---

## Criar um novo experimento

Crie um arquivo JSON em `experiments/` baseado nos exemplos existentes e passe via `--config`:

```json
{
    "experiment_name": "weather_forecasting",
    "run_name": "meu_experimento",
    "epochs": 50,
    "batch_size": 64,
    "data": {
        "parquet_path": "/app/data/spec",
        "feature_columns": [
            "temp_ar_c", "umidade_rel_ar_percent", "pressao_atm_estacao_mb",
            "precipitacao_total_mm", "hora_sin", "hora_cos",
            "temp_lag_1h", "temp_lag_24h",
            "temp_ma_6h", "temp_ma_12h",
            "pressao_ma_6h", "pressao_ma_12h",
            "pressao_tendencia_1h", "temp_tendencia_1h"
        ],
        "target_columns": ["temp_ar_c"],
        "sequence_length": 24,
        "horizon": 1
    },
    "model": {
        "model_type": "lstm",
        "hidden_size": 256,
        "num_layers": 3
    }
}
```

> As colunas em `feature_columns` devem bater exatamente com as colunas do parquet em `data/spec/`.
