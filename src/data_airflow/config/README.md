# Data Airflow Config

Diretorio para configuracoes do Airflow compartilhadas com os containers.

## Conteudo esperado

- `airflow.cfg` customizado (quando necessario).
- Arquivos auxiliares de configuracao referenciados por DAGs.

## Como e usado

No `docker-compose-airflow.yaml`, este diretorio e montado como:

- Host: `./src/data_airflow/config`
- Container: `/opt/airflow/config`

A variavel `AIRFLOW_CONFIG` aponta para:

- `/opt/airflow/config/airflow.cfg`

## Boas praticas

1. Nao commitar secrets em texto puro.
2. Preferir variaveis de ambiente para credenciais.
3. Versionar somente parametros nao sensiveis e defaults.
4. Documentar qualquer override relevante no README do modulo.
