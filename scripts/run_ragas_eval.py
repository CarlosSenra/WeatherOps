#!/usr/bin/env python3
"""Avaliação offline RAGAS para o golden set do agente WeatherOps.

Calcula 4 métricas: faithfulness, answer_relevancy, context_precision, context_recall.
Requer GOOGLE_API_KEY e o grupo de dependências do agente instalado:
    poetry install --with agent

Uso:
    GOOGLE_API_KEY=... poetry run python scripts/run_ragas_eval.py
    GOOGLE_API_KEY=... poetry run python scripts/run_ragas_eval.py --limit 5 --output-dir results/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "src" / "api_agent" / "eval" / "golden_set.jsonl"


# ---------------------------------------------------------------------------
# Carregamento do golden set
# ---------------------------------------------------------------------------


def load_golden_set(path: Path, limit: int | None = None) -> list[dict]:
    """Lê o golden set e mapeia para o formato esperado pelo RAGAS."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append(
                {
                    "user_input": row.get("user_input", ""),
                    "response": row.get("response", ""),
                    "retrieved_contexts": row.get("retrieved_contexts") or [],
                    "reference": row.get("reference", ""),
                }
            )
    if limit:
        records = records[:limit]
    return records


# ---------------------------------------------------------------------------
# Avaliação RAGAS
# ---------------------------------------------------------------------------


def run_evaluation(records: list[dict], api_key: str) -> dict[str, float]:
    """Executa as 4 métricas RAGAS usando Gemini como judge LLM."""
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

    llm_wrapper = LangchainLLMWrapper(llm)
    emb_wrapper = LangchainEmbeddingsWrapper(embeddings)

    samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
        )
        for r in records
    ]
    dataset = EvaluationDataset(samples=samples)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm_wrapper,
        embeddings=emb_wrapper,
    )

    df = result.to_pandas()
    return {col: float(df[col].mean()) for col in df.columns if df[col].dtype.kind == "f"}


# ---------------------------------------------------------------------------
# Persistência de resultados
# ---------------------------------------------------------------------------


def write_results(scores: dict[str, float], n_examples: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    json_path = output_dir / f"ragas_results_{ts}.json"
    payload = {
        "timestamp": ts,
        "n_examples": n_examples,
        "scores": scores,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultados JSON: {json_path}")

    csv_path = output_dir / f"ragas_results_{ts}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "score"])
        for metric, score in scores.items():
            writer.writerow([metric, f"{score:.4f}"])
    print(f"Resultados CSV:  {csv_path}")


def update_prometheus_gauges(scores: dict[str, float], api_base_url: str = "http://localhost:8888") -> None:
    import requests

    try:
        r = requests.post(f"{api_base_url}/v1/internal/ragas", json={"scores": scores}, timeout=5)
        r.raise_for_status()
        print(f"Gauges Prometheus atualizados via API ({api_base_url}).")
    except Exception as exc:
        print(f"Aviso: não foi possível atualizar gauges via API — {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "evaluation_results",
        help="Diretório de saída (default: evaluation_results/).",
    )
    p.add_argument("--limit", type=int, default=None, help="Avaliar apenas os primeiros N exemplos.")
    p.add_argument("--update-prometheus", action="store_true", help="Atualizar gauges weatherops_agent_ragas_score via API.")
    p.add_argument("--api-url", default="http://localhost:8888", help="URL base da API WeatherOps (default: http://localhost:8888).")
    args = p.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("ERRO: GOOGLE_API_KEY não definida.", file=sys.stderr)
        return 1

    if not GOLDEN.exists():
        print(f"ERRO: golden set não encontrado em {GOLDEN}", file=sys.stderr)
        return 1

    records = load_golden_set(GOLDEN, limit=args.limit)
    print(f"Avaliando {len(records)} exemplo(s) com RAGAS...")

    scores = run_evaluation(records, api_key)

    print("\nResultados RAGAS:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")

    write_results(scores, n_examples=len(records), output_dir=args.output_dir)

    if args.update_prometheus:
        update_prometheus_gauges(scores, api_base_url=args.api_url)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
