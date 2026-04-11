# Troubleshooting

## Erro: FileNotFoundError para config no Docker (Windows Git Bash)

Sintoma:
- O container nao encontra arquivo em `/app/...`.

Causa comum:
- Git Bash pode reescrever `/app/...` para caminho local incorreto.

Correcao:
- Use `//app/...` nos comandos `--config`.

Exemplo:

```bash
docker compose --profile train run --rm trainer --config //app/experiments/transformer/transformer_h72_v1.json
```

## Erro: experimento nao encontrado

Checklist:
1. Confirmar nome com sufixo `_v` no arquivo.
2. Confirmar pasta correta (`lstm` ou `transformer`).
3. Confirmar versao existente (ex.: `v1..v36`).

## Erro: CUDA indisponivel no container

Checklist:
1. Validar host com `nvidia-smi`.
2. Confirmar suporte GPU habilitado no Docker Desktop.
3. Testar dentro do container:

```bash
docker compose --profile train-gpu run --rm --entrypoint python trainer-gpu -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## Erro: run de avaliacao nao carrega modelo

Causa comum:
- Runs antigos com metadados de artifact path diferentes.

Comportamento esperado:
- O modulo de avaliacao tenta fallback para `best_model.pt` no `mlruns` local.

Acao:
1. Garantir que `src/ml_workstation/mlruns` esta montado/presente.
2. Reexecutar avaliacao com `--run-id` valido.

## Erro: metricas estranhas apos alteracao de dados

Causas comuns:
- `feature_columns` desatualizado.
- Mudanca de distribuicao sem ajustar config/modelo.

Acao:
1. Comparar schema de `data/spec` com JSON do experimento.
2. Rodar smoke test antes de treino completo.
3. Revisar normalizacao e colunas alvo.

## Erro: Airflow nao sobe completamente

Checklist:
1. Rodar `airflow-init` antes do `up -d`.
2. Verificar variaveis do `.env`.
3. Verificar portas 8080, 5432 e 6379 livres.
4. Inspecionar logs:

```bash
docker compose -f docker-compose-airflow.yaml logs -f airflow-webserver
docker compose -f docker-compose-airflow.yaml logs -f airflow-scheduler
```
