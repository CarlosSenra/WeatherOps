"""CLI para selecionar e promover o melhor modelo de um experimento MLflow."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

from src.ml_workstation.evaluation.mlflow_helpers import workspace_root
from src.ml_workstation.promotion.promote import PromotionRejectedError, promote_best, promote_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
_WINDOWS_ABS_PATH_RE = re.compile(r"^[a-zA-Z]:[\\/]")
_WINDOWS_UNC_PATH_RE = re.compile(r"^\\\\[^\\]+\\[^\\]+")


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
    parser.add_argument(
        "--update-knowledge-base",
        action="store_true",
        default=False,
        help=(
            "Após exportar, reconstrói a base vetorial ChromaDB do agente com perfis "
            "históricos do município detectado no serving_data. "
            "Requer GOOGLE_API_KEY no ambiente e --export-dir definido."
        ),
    )
    parser.add_argument(
        "--knowledge-base-path",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Diretório de persistência do ChromaDB. "
            "Predefinição: variável de ambiente KNOWLEDGE_BASE_PATH ou "
            "src/api_agent/knowledge/chroma_db relativo ao repositório."
        ),
    )
    return parser


def _resolve_export_dir(export_dir: str | None) -> str | None:
    if not export_dir:
        return None
    raw_path = export_dir.strip()
    export_path = Path(raw_path)
    if export_path.is_absolute():
        return str(export_path.resolve())
    if _WINDOWS_ABS_PATH_RE.match(raw_path) or _WINDOWS_UNC_PATH_RE.match(raw_path):
        return os.path.normpath(raw_path)
    return str((workspace_root() / export_path).resolve())


def _resolve_knowledge_base_path(kb_path_arg: str | None) -> Path:
    raw = kb_path_arg or os.environ.get("KNOWLEDGE_BASE_PATH", "")
    if raw:
        p = Path(raw.strip())
        return p.resolve() if p.is_absolute() else (workspace_root() / p).resolve()
    return (workspace_root() / "src" / "api_agent" / "knowledge" / "chroma_db").resolve()


def _update_knowledge_base(export_dir: str, model_name: str, kb_path: Path) -> None:
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not google_api_key:
        logger.warning(
            "--update-knowledge-base ignorado: GOOGLE_API_KEY não definida no ambiente."
        )
        return

    feature_store_dir = Path(export_dir) / model_name / "feature_store"
    if not feature_store_dir.is_dir():
        logger.warning(
            "--update-knowledge-base: feature_store não encontrado em %s. "
            "Verifique se o export gerou serving_data.",
            feature_store_dir,
        )
        return

    try:
        from src.api_agent.rag.knowledge_builder import build_knowledge_base
        n = build_knowledge_base(
            feature_store_dir=feature_store_dir,
            chroma_path=kb_path,
            google_api_key=google_api_key,
        )
        print(f"Base vetorial atualizada: {n} chunks em {kb_path}")
    except Exception as exc:
        logger.error("Falha ao atualizar base vetorial: %s", exc, exc_info=True)
        print(f"AVISO: base vetorial não atualizada — {exc}", file=sys.stderr)


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

    if args.update_knowledge_base and export_dir:
        effective_model_name = args.model_name or args.experiment_name
        kb_path = _resolve_knowledge_base_path(getattr(args, "knowledge_base_path", None))
        _update_knowledge_base(export_dir, effective_model_name, kb_path)


if __name__ == "__main__":
    main()
