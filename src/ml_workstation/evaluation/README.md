# Evaluation

Módulo para avaliar modelos treinados no MLflow usando run_id, gerar predições no conjunto de teste e salvar um gráfico Plotly com Real vs Predito.

> **Modelos suportados: LSTM e Transformer.**
> A avaliação interativa via este módulo não é suportada para runs de TFT e NBEATS.
> Veja a seção [Limitação para TFT e NBEATS](#limitacao-para-tft-e-nbeats) ao final deste documento.

## Arquivos

```text
src/ml_workstation/evaluation/
├── README.md
├── __init__.py
├── core.py
├── mlflow_helpers.py
└── run_evaluation.py
```

- run_evaluation.py: somente CLI (parse de argumentos + chamada da avaliação)
- core.py: pipeline de avaliação (inferência, validações, pós-processamento e gráfico)
- mlflow_helpers.py: integração com MLflow (leitura de params, resolução de paths e load de modelo)

## O que o script faz

1. Lê os parâmetros do run no MLflow.
2. Reconstroi a configuração de dados usada no treino.
3. Tenta carregar o modelo via URI runs:/<run_id>/model.
4. Se a URI falhar, usa fallback para best_model.pt local no mlruns.
5. Executa inferência no split de teste.
6. Faz inverse transform dos targets para escala original.
7. Gera 1 gráfico Plotly de linhas com Real vs Predito.
8. Salva um arquivo HTML no diretório de saída.

## Uso rápido com Poetry (Windows)

Execute a partir da raiz do projeto:

```bash
poetry run python -m src.ml_workstation.evaluation.run_evaluation --run-id <RUN_ID>
```

Exemplo com um run real:

```bash
poetry run python -m src.ml_workstation.evaluation.run_evaluation --run-id 14815808b23f47238e2e2379f59f2d2e
```

## Parâmetros disponíveis

- --run-id: run id do MLflow (obrigatório)
- --target-index: índice do target a plotar (default: 0)
- --horizon-step: passo do horizonte a plotar, base 0 (default: 0)
- --max-points: limita a quantidade de pontos no gráfico (default: 2000, 0 desliga)
- --device: dispositivo de inferência, exemplo cpu ou cuda (default: cpu)
- --output-dir: diretório de saída do html (default: evaluation_results)
- --tracking-uri: URI opcional do MLflow

## Associação com parâmetros de treino

Para escolher os argumentos corretos no evaluate, use os parâmetros que foram logados no treino para aquele run_id.

| No treino (TrainingConfig) | Onde aparece no MLflow | No evaluate | Regra prática |
|---|---|---|---|
| data.horizon | param data.horizon | --horizon-step | Válidos de 0 até data.horizon - 1 |
| data.target_columns | param data.target_columns | --target-index | Válidos de 0 até len(target_columns) - 1 |
| batch_size | param batch_size | (interno) | O script reaproveita automaticamente |
| data.feature_columns | param data.feature_columns | (interno) | O script reaproveita automaticamente |

Regra do horizonte:

- Se no treino data.horizon = 1, então apenas --horizon-step 0 é válido.
- Se no treino data.horizon = 3, então --horizon-step 0, 1 e 2 são válidos.

Conversão mental:

- horizonte t+1 no gráfico = --horizon-step 0
- horizonte t+2 no gráfico = --horizon-step 1
- horizonte t+3 no gráfico = --horizon-step 2

Resumo de fórmula:

- limite superior de horizon-step = data.horizon - 1
- limite superior de target-index = len(data.target_columns) - 1

## Exemplos

Target 0, horizonte t+1:

```bash
poetry run python -m src.ml_workstation.evaluation.run_evaluation \
  --run-id 14815808b23f47238e2e2379f59f2d2e \
  --target-index 0 \
  --horizon-step 0
```

Mesmo run, horizonte t+3 (somente se esse run tiver data.horizon >= 3):

```bash
poetry run python -m src.ml_workstation.evaluation.run_evaluation \
  --run-id 14815808b23f47238e2e2379f59f2d2e \
  --target-index 0 \
  --horizon-step 2
```

Com saída customizada:

```bash
poetry run python -m src.ml_workstation.evaluation.run_evaluation \
  --run-id 14815808b23f47238e2e2379f59f2d2e \
  --output-dir src/ml_workstation/evaluation_results
```

## Saída

O script imprime o caminho do arquivo gerado, por exemplo:

```text
Grafico salvo em: evaluation_results/14815808b23f47238e2e2379f59f2d2e_target-temp_ar_c_h-1.html
```

## Observações

- Alguns runs antigos podem ter metadados de artifact repository incompatíveis localmente. Nesses casos, o fallback para best_model.pt é usado automaticamente.
- Se o run foi treinado em container com caminho /app/data/spec, o script resolve para o caminho local do workspace quando possível.
- Se target-index ou horizon-step estiver fora do range do modelo, o script falha com mensagem explícita.

## Limitacao para TFT e NBEATS

O módulo de avaliação atual (`core.py`) foi projetado para a interface `ITimeSeriesModel`, que recebe tensores crus `(batch, seq_len, n_features)` e retorna `(batch, horizon, n_targets)`.

Modelos TFT e NBEATS são `LightningModule` do pytorch-forecasting e **não implementam essa interface**. A inferência desses modelos exige um batch no formato de dicionário produzido pelo `TimeSeriesDataSet`, que não está disponível no pipeline de avaliação atual.

O que funciona para runs TFT/NBEATS:

- O modelo é salvo normalmente no MLflow via `mlflow.pytorch.log_model`.
- O carregamento via `runs:/<run_id>/model` não falha.
- Métricas de validação por época e o snapshot final ficam registrados no MLflow e podem ser consultados normalmente pela UI.

O que **não funciona** para runs TFT/NBEATS via este módulo:

- Inferência no split de teste.
- Geração do gráfico HTML Real vs Predito.

Para avaliar visualmente um modelo TFT ou NBEATS, use diretamente as APIs do pytorch-forecasting após recarregar o modelo e reconstruir o `TimeSeriesDataSet` de teste.
