# ML Workstation — Arquitetura e Fluxo de Treinamento

Documentação técnica do fluxo completo, da definição de configuração até a execução do treinamento.

---

## Visão geral

O entrypoint é `train.py`. Ele recebe um `TrainingConfig` (via defaults ou JSON) e instancia, em ordem, quatro componentes independentes — dados, modelo, tracker e trainer — que se comunicam através de interfaces bem definidas.

```
TrainingConfig
      │
      ├──► ParquetDataLoader  ──► DataLoaders (train / val / test)
      │
      ├──► build_model()      ──► WeatherLSTM | WeatherTransformer
      │
      ├──► MLflowTracker      ──► MLflow run (parâmetros + métricas)
      │
      └──► Trainer            ──► loop de épocas + early stopping + checkpoint
```

---

## 1. Configuração (`config/training_config.py`)

Tudo começa em três modelos Pydantic compostos:

```
TrainingConfig
├── experiment_name, run_name
├── epochs, batch_size, learning_rate, weight_decay
├── early_stopping_patience, checkpoint_dir, device
│
├── data: DataConfig
│   ├── parquet_path          → onde estão os arquivos .parquet
│   ├── feature_columns       → lista de colunas de entrada do modelo
│   ├── target_columns        → o que prever (ex: ["temp_ar_c"])
│   ├── sequence_length = 24  → janela de look-back em horas
│   ├── horizon = 1           → quantos passos à frente prever
│   ├── train_ratio = 0.80    → proporção para treino
│   └── val_ratio   = 0.10    → proporção para validação (restante = teste)
│
├── governance: GovernanceConfig
│   ├── model_name, model_version, model_type
│   ├── owner              → e-mail do responsável (config ou .env)
│   ├── risk_level, fairness_checked
│   ├── training_data_version → hash/versionamento dos dados
│   └── git_sha            → commit do código
│
└── model: ModelConfig
    ├── model_type: "lstm" | "transformer"
    ├── hidden_size, num_layers, dropout
    └── num_heads, ffn_dim   (somente Transformer)
```

O `TrainingConfig` pode ser instanciado de duas formas:

```python
# 1. Defaults embutidos
config = TrainingConfig()

# 2. A partir de um arquivo JSON
config = TrainingConfig.model_validate(json.load(open("experiments/lstm_baseline.json")))
```

O JSON é validado pelo Pydantic antes de qualquer execução — erros de configuração são detectados antes de tocar os dados.

---

## 2. Dados (`data/`)

### `ParquetDataLoader`

Recebe um `DataConfig` e executa o pipeline de dados em quatro etapas:

**Etapa 1 — Leitura**
Lê um arquivo `.parquet` ou todos os `.parquet` de um diretório. Os dados vêm da pipeline `data/spec/`, que já contém as features de engenharia (lags, médias móveis, features cíclicas). O DataFrame é ordenado pelo índice temporal e as colunas desnecessárias são descartadas.

**Etapa 2 — Split temporal**
O split é feito em ordem cronológica, sem embaralhar:

```
─────────────────────────────────────────────────────── tempo
│       80% treino        │  10% val  │   10% teste   │
```

Isso evita data leakage — o modelo nunca vê dados futuros durante o treino ou a validação.

**Etapa 3 — Normalização**
O `StandardScaler` é ajustado (`fit`) apenas no conjunto de treino e aplicado (`transform`) nos três conjuntos. Isso garante que a média e o desvio padrão usados na normalização venham exclusivamente dos dados de treino.

**Etapa 4 — DataLoaders**
Cada fatia normalizada é passada para `WeatherSequenceDataset`, que gera pares `(X, y)`:

```
X: (sequence_length, n_features)   ← janela de entrada de 24h
y: (horizon, n_targets)            ← valor a prever
```

O índice `i` do dataset corresponde a:
- `X = dados[i : i + 24]`
- `y = dados[i + 24 : i + 24 + horizon]` (somente colunas alvo)

Os três `DataLoader` PyTorch resultantes são retornados junto com metadados (`n_features`, `n_targets`, tamanhos dos conjuntos).

---

## 3. Modelo (`models/`)

### Interface

Todos os modelos implementam `ITimeSeriesModel` (ABC):

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # entrada:  (batch, seq_len, n_features)
    # saída:    (batch, horizon, n_targets)
```

### `WeatherLSTM`

```
entrada (batch, 24, n_features)
    │
    └──► nn.LSTM(hidden_size, num_layers, batch_first=True)
              │
         último estado oculto h_n[-1]  → shape (batch, hidden_size)
              │
         nn.Linear(hidden_size → horizon * n_targets)
              │
         reshape → (batch, horizon, n_targets)
```

O dropout entre camadas LSTM só é aplicado quando `num_layers > 1`.

### `WeatherTransformer`

```
entrada (batch, 24, n_features)
    │
    └──► nn.Linear(n_features → hidden_size)        # projeção de entrada
              │
         _PositionalEncoding(sinusoidal)              # codificação de posição
              │
         nn.TransformerEncoder(num_layers, num_heads) # atenção multi-cabeça
              │
         mean-pool na dimensão temporal               # (batch, hidden_size)
              │
         nn.Linear(hidden_size → horizon * n_targets)
              │
         reshape → (batch, horizon, n_targets)
```

O mean-pool agrega todos os 24 passos de forma igual, mais estável que pegar apenas o último token para sequências curtas.

### Factory `build_model()`

```python
model = build_model(
    n_features=data_output.n_features,
    n_targets=data_output.n_targets,
    horizon=config.data.horizon,
    config=config.model,       # model_type decide qual classe instanciar
)
```

---

## 4. Rastreamento (`tracking/mlflow_tracker.py`)

O `MLflowTracker` isola todas as chamadas MLflow em um único lugar. Os demais módulos não importam `mlflow` diretamente.

**Ciclo de vida do run:**

```python
tracker.start_run()
# └── mlflow.set_experiment(experiment_name)
# └── mlflow.start_run(run_name)
# └── mlflow.log_params(config.model_dump())  ← config completo de uma vez
# └── mlflow.set_tags(governance)              ← metadados de governança
# └── mlflow.log_params(governance.*)          ← governança também como params

    # a cada época:
    tracker.log_epoch_metrics({"train_loss": ..., "val_loss": ..., "mae": ..., "rmse": ...}, epoch)

# ao final:
tracker.log_governance_metrics(...)       # snapshot final prefixado como governance_metric.*
tracker.log_artifact(checkpoint_path)   # arquivo .pt do melhor modelo
tracker.log_model(model)                # modelo PyTorch registrado no MLflow
tracker.end_run()
```

O `TrainingConfig` é serializado via `model_dump(mode="json")` e achatado antes de ser logado, garantindo que todos os hiperparâmetros apareçam no MLflow como parâmetros individuais (ex: `data.sequence_length`, `model.hidden_size`).

---

## 5. Treinamento (`training/trainer.py`)

O `Trainer` recebe objetos já construídos e os orquestra. Não constrói nada.

### Loop por época

```
para cada época:
    1. _train_epoch()   → forward + backward + optimizer.step() em cada batch
                          retorna: avg train_loss (MSE normalizado por amostra)

    2. _validate()      → forward sem gradientes em todo o val_loader
                          retorna: val_loss + mae + rmse + mape

    3. tracker.log_epoch_metrics(métricas, época)

    4. se val_loss melhorou:
           salva state_dict em artifacts/<run_name>/best_model.pt
           reseta contador de patience

       senão:
           incrementa contador de patience
           se patience >= early_stopping_patience:
               para o loop
```

### Saída

`Trainer.fit()` retorna um `TrainerOutput` (Pydantic):

```python
TrainerOutput(
    best_epoch=12,
    best_val_loss=0.0043,
    checkpoint_path="artifacts/lstm_baseline/best_model.pt",
    total_epochs_run=19,
    final_metrics={"train_loss": ..., "val_loss": ..., "mae": ..., "rmse": ..., "mape": ...},
)
```

### Métricas (`training/metrics.py`)

```
MAE  = mean(|y_pred - y_true|)
RMSE = sqrt(mean((y_pred - y_true)²))
MAPE = mean(|y_pred - y_true| / max(|y_true|, epsilon)) × 100
```

Observações importantes:
- `compute_metrics()` calcula MAE/RMSE na escala recebida.
- `compute_mape()` usa `epsilon` no denominador para estabilizar quando `y_true` está próximo de zero.
- No `Trainer`, o MAPE principal (`mape`) é recalculado após desfazer a normalização do alvo (escala original).
- O MAPE na escala normalizada é mantido como métrica auxiliar `mape_scaled`.

---

## 6. Fluxo completo em `train.py`

```python
# 1. Config
config = TrainingConfig.model_validate(json.load(...))

# 2. Dados
loader = ParquetDataLoader(config.data)
train_loader, val_loader, test_loader, data_output = loader.build(batch_size=config.batch_size)
target_indices = [config.data.feature_columns.index(c) for c in config.data.target_columns]
target_mean = loader.scaler.mean_[target_indices]
target_scale = loader.scaler.scale_[target_indices]

# 3. Modelo
model = build_model(data_output.n_features, data_output.n_targets, config.data.horizon, config.model)

# 4. Tracker
tracker = MLflowTracker(config)
tracker.start_run()

# 5. Treino
trainer = Trainer(
    model,
    train_loader,
    val_loader,
    config,
    tracker,
    target_mean=target_mean,
    target_scale=target_scale,
)
result  = trainer.fit()

# 6. Artefatos
tracker.log_artifact(result.checkpoint_path)
tracker.log_model(model)
tracker.end_run()
```

---

## Fluxo de dependências (sem ciclos)

```
train.py
  ├── config/training_config.py        (sem dependências internas)
  ├── data/loader.py        → config + data/dataset.py
  ├── models/__init__.py    → lstm.py | transformer.py → config
  ├── tracking/mlflow_tracker.py → config
  └── training/trainer.py   → metrics.py + mlflow_tracker.py + config
```
