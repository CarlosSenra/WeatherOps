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

# LSTM 
docker compose --profile train run --rm trainer --config //app/experiments/lstm_baseline.json
docker compose --profile train run --rm trainer --config //app/experiments/lstm_h72.json
docker compose --profile train run --rm trainer --config //app/experiments/lstm_h72_v2.json
docker compose --profile train run --rm trainer --config //app/experiments/lstm_h168.json
docker compose --profile train run --rm trainer --config //app/experiments/lstm_h168_v2.json
docker compose --profile train run --rm trainer --config //app/experiments/lstm_h336.json
docker compose --profile train run --rm trainer --config //app/experiments/lstm_h336_v2.json

# Transformer 
docker compose --profile train run --rm trainer --config //app/experiments/transformer_baseline.json
docker compose --profile train run --rm trainer --config //app/experiments/transformer_h72.json
docker compose --profile train run --rm trainer --config //app/experiments/transformer_h72_v2.json
docker compose --profile train run --rm trainer --config //app/experiments/transformer_h168.json
docker compose --profile train run --rm trainer --config //app/experiments/transformer_h168_v2.json
docker compose --profile train run --rm trainer --config //app/experiments/transformer_h336.json
docker compose --profile train run --rm trainer --config //app/experiments/transformer_h336_v2.json
```

### Treinar com CUDA (GPU)

Use o serviço `trainer-gpu` com profile `train-gpu`.

1. Teste de detecção de GPU no container:

```bash
docker compose --profile train-gpu run --rm --entrypoint python trainer-gpu -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

2. Smoke test em CUDA:

```bash
docker compose --profile train-gpu run --rm trainer-gpu --config //app/experiments/lstm_smoke_test_cuda.json
docker compose --profile train-gpu run --rm trainer-gpu --config //app/experiments/transformer_baseline.json
docker compose --profile train-gpu run --rm trainer-gpu --config //app/experiments/lstm_h72.json
docker compose --profile train-gpu run --rm trainer-gpu --config //app/experiments/lstm_h72_v2.json
docker compose --profile train-gpu run --rm trainer-gpu --config //app/experiments/lstm_h168.json
docker compose --profile train-gpu run --rm trainer-gpu --config //app/experiments/lstm_h168_v2.json
docker compose --profile train-gpu run --rm trainer-gpu --config //app/experiments/lstm_h336.json
docker compose --profile train-gpu run --rm trainer-gpu --config //app/experiments/lstm_h336_v2.json

```

3. Para executar qualquer experimento em CUDA, use `"device": "cuda"` no JSON e rode via `trainer-gpu`.

Pré-requisitos do host para CUDA:

- Driver NVIDIA instalado e funcional (`nvidia-smi`).
- Docker Desktop com suporte a GPU habilitado.
- Ambiente com acesso à GPU (Windows/WSL2 configurado para CUDA).

> **Atenção (Git Bash / Windows):** use `//app/...` (duas barras) ao passar caminhos do container.
> O Git Bash expande `/app/...` para `C:/Program Files/Git/app/...` causando `FileNotFoundError`.

### Governança no MLflow

Cada treino agora registra metadados de governança em dois formatos:

- **Tags MLflow** (`model_name`, `model_version`, `model_type`, `training_data_version`, `owner`, `risk_level`, `fairness_checked`, `git_sha`)
- **Parâmetros MLflow** com prefixo `governance.*` (mesmos campos, para facilitar filtros)

Campos de entrada manual no JSON do experimento (bloco `governance`):

```json
"governance": {
    "model_name": "weather_lstm_h72",
    "model_version": "1.0.0",
    "model_type": "regression",
    "risk_level": "medium",
    "fairness_checked": false
}
```

Campos preenchidos automaticamente no runtime:

- `owner`: lido de `EMAIL` no ambiente (arquivo `.env` na raiz). Se ausente, usa `unknown`.
- `training_data_version`: hash `md5` lido de `data.dvc`.
- `git_sha`: commit atual via `git rev-parse HEAD`.

Precedência de resolução para `git_sha` e `training_data_version`:

1. Valor explícito no JSON do experimento (`governance.git_sha` / `governance.training_data_version`).
2. Variáveis de ambiente no container (`GIT_SHA` / `TRAINING_DATA_VERSION`).
3. Runtime local (`git rev-parse HEAD` e leitura de `data.dvc`).
4. Fallback final: `unknown`.

No Docker Compose de treino, `.git` e `data.dvc` são montados em modo leitura para permitir a captura automática desses metadados.

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
