"""CLI para selecionar e promover o melhor modelo de um experimento MLflow."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from src.ml_workstation.evaluation.mlflow_helpers import workspace_root
from src.ml_workstation.promotion.promote import PromotionRejectedError, promote_best, promote_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seleciona e promove o melhor modelo de um experimento MLflow para produção "
            "no Model Registry (alias production). Opcionalmente exporta o artefato para disco."
        )
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Nome do experimento MLflow (ex.: weather_forecasting_h72)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Run ID explícito a promover. "
            "Se omitido, seleciona automaticamente pelo melhor valor da métrica."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mape",
        help="Métrica para seleção automática do melhor run (default: mape).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Nome do modelo no Model Registry. "
            "Se omitido, usa o experiment_name como nome do modelo."
        ),
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI. Se omitido, detecta automaticamente via mlruns/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Força a promoção mesmo que o candidato tenha métrica pior "
            "que o modelo atual em produção."
        ),
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Após promover, exporta o artefato 'model' para DIR/<nome_no_registry>. "
            "Predefinição: variável de ambiente WEATHEROPS_EXPORT_MODELS_DIR "
            "(ex.: src/api/ml_models relativo ao repositório)."
        ),
    )
    parser.add_argument(
        "--strict-export",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Falha o comando quando a exportação local não gerar artefatos válidos. "
            "Use --no-strict-export para manter comportamento permissivo."
        ),
    )
    return parser


def _resolve_export_dir(export_dir: str | None) -> str | None:
    if not export_dir:
        return None
    export_path = Path(export_dir)
    if export_path.is_absolute():
        return str(export_path.resolve())
    return str((workspace_root() / export_path).resolve())


def main() -> None:
    args = _build_parser().parse_args()
    export_dir_arg = getattr(args, "export_dir", None) or os.environ.get(
        "WEATHEROPS_EXPORT_MODELS_DIR"
    )
    export_dir = _resolve_export_dir(export_dir_arg)
    if export_dir:
        logger.info("Diretório efetivo de exportação: %s", export_dir)

    try:
        if args.run_id:
            version = promote_run(
                run_id=args.run_id,
                experiment_name=args.experiment_name,
                model_name=args.model_name,
                tracking_uri=args.tracking_uri,
                force=args.force,
                export_dir=export_dir,
                strict_export=args.strict_export,
            )
            print(
                f"Modelo promovido com sucesso.\n"
                f"  Experimento : {args.experiment_name}\n"
                f"  Run ID      : {args.run_id}\n"
                f"  Versão      : {version}"
            )
        else:
            version = promote_best(
                experiment_name=args.experiment_name,
                metric=args.metric,
                model_name=args.model_name,
                tracking_uri=args.tracking_uri,
                force=args.force,
                export_dir=export_dir,
                strict_export=args.strict_export,
            )
            print(
                f"Melhor run promovido com sucesso.\n"
                f"  Experimento : {args.experiment_name}\n"
                f"  Métrica     : {args.metric}\n"
                f"  Versão      : {version}"
            )

    except PromotionRejectedError as exc:
        logger.error("Promoção rejeitada: %s", exc)
        print(f"ERRO: {exc}", file=sys.stderr)
        sys.exit(1)

    except Exception as exc:
        logger.error("Falha na promoção: %s", exc, exc_info=True)
        print(f"ERRO inesperado: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
