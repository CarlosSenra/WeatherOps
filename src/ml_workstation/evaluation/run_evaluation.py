"""CLI para avaliar um modelo MLflow por run_id e gerar grafico Plotly."""

from __future__ import annotations

import argparse
import logging

from src.ml_workstation.evaluation.core import evaluate_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Avalia modelo do MLflow por run_id e gera 1 grafico Plotly (real vs predito)."
    )
    parser.add_argument("--run-id", type=str, required=True, help="Run ID do MLflow")
    parser.add_argument(
        "--target-index",
        type=int,
        default=0,
        help="Indice do target para plotar (0-based)",
    )
    parser.add_argument(
        "--horizon-step",
        type=int,
        default=0,
        help="Passo do horizonte para plotar (0-based)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Numero maximo de pontos no grafico (0 desliga downsampling)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Dispositivo para inferencia (ex.: cpu, cuda)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Diretorio de saida para o HTML do grafico",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="Tracking URI opcional do MLflow",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_file = evaluate_run(
        run_id=args.run_id,
        target_index=args.target_index,
        horizon_step=args.horizon_step,
        max_points=args.max_points,
        device_name=args.device,
        output_dir=args.output_dir,
        tracking_uri=args.tracking_uri,
    )
    print(f"Grafico salvo em: {output_file}")


if __name__ == "__main__":
    main()
