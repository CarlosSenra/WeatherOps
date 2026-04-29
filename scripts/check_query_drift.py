#!/usr/bin/env python3
"""Verifica deriva de queries em relação ao golden set baseline.

Dois modos:
- Comprimento (padrão): compara estatísticas de tamanho de texto
- Semântico (--semantic): compara centróides de embeddings Google (requer GOOGLE_API_KEY)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "src" / "api_agent" / "eval" / "golden_set.jsonl"


def _load_golden() -> list[dict]:
    rows: list[dict] = []
    with GOLDEN.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _baseline_lengths(rows: list[dict]) -> list[int]:
    return [len(row.get("user_input", "")) for row in rows]


# ---------------------------------------------------------------------------
# Drift semântico
# ---------------------------------------------------------------------------


def _embed_texts(texts: list[str], api_key: str) -> list[list[float]]:
    import google.generativeai as genai  # type: ignore[import-untyped]

    genai.configure(api_key=api_key)
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=texts,
        task_type="RETRIEVAL_QUERY",
    )
    embeddings = result["embedding"]
    # API retorna lista única quando content é lista de 1 elemento
    if isinstance(embeddings[0], float):
        return [embeddings]
    return embeddings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _centroid(embeddings: list[list[float]]) -> list[float]:
    n = len(embeddings)
    dim = len(embeddings[0])
    return [sum(e[i] for e in embeddings) / n for i in range(dim)]


def compute_semantic_drift(baseline_texts: list[str], current_texts: list[str], api_key: str) -> float:
    """Calcula deriva semântica comparando centróides de embeddings.

    Retorna score em [0, 1]: 0 = idêntico, 1 = máxima divergência.
    Usa abordagem de centróide O(N+M) — adequada para CI.
    """
    baseline_embs = _embed_texts(baseline_texts, api_key)
    current_embs = _embed_texts(current_texts, api_key)
    b_cent = _centroid(baseline_embs)
    c_cent = _centroid(current_embs)
    sim = _cosine_similarity(b_cent, c_cent)
    return round(1.0 - sim, 4)


def _update_prometheus(score: float) -> None:
    try:
        from src.api.metrics import query_semantic_drift_score  # type: ignore[import]

        query_semantic_drift_score.set(score)
        print(f"Prometheus gauge atualizado: weatherops_query_semantic_drift_score={score}")
    except ImportError:
        print("Aviso: não foi possível atualizar gauge Prometheus (src não está no path).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries-file", type=Path, default=None, help="Um texto por linha (queries reais).")
    p.add_argument("--semantic", action="store_true", help="Ativa deriva semântica (requer GOOGLE_API_KEY).")
    p.add_argument("--threshold", type=float, default=0.3, help="Limiar de alerta para deriva semântica (default 0.3).")
    p.add_argument("--update-prometheus", action="store_true", help="Atualiza gauge weatherops_query_semantic_drift_score.")
    args = p.parse_args()

    golden_rows = _load_golden()
    base = _baseline_lengths(golden_rows)

    print("Golden set (baseline):")
    p95_idx = int(0.95 * (len(base) - 1))
    print(f"  n={len(base)} mean={statistics.mean(base):.1f} stdev={statistics.pstdev(base):.1f} p95={sorted(base)[p95_idx]}")

    current_texts: list[str] | None = None
    if args.queries_file and args.queries_file.is_file():
        lines = [ln.strip() for ln in args.queries_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        lens = [len(x) for x in lines]
        print("Ficheiro de queries:")
        p95q_idx = int(0.95 * (len(lens) - 1))
        print(f"  n={len(lens)} mean={statistics.mean(lens):.1f} stdev={statistics.pstdev(lens):.1f} p95={sorted(lens)[p95q_idx]}")
        current_texts = lines
    else:
        print("(Sem --queries-file: apenas baseline.)")

    if args.semantic:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            print("ERRO: GOOGLE_API_KEY não definida. Ignorando drift semântico.", file=sys.stderr)
            return 1

        baseline_texts = [row["user_input"] for row in golden_rows]
        eval_texts = current_texts if current_texts is not None else baseline_texts
        print("\nCalculando drift semântico...")
        score = compute_semantic_drift(baseline_texts, eval_texts, api_key)
        print(f"Drift semântico: {score:.4f} (limiar={args.threshold})")

        if args.update_prometheus:
            _update_prometheus(score)

        if score > args.threshold:
            print(f"ALERTA: drift semântico {score:.4f} excede limiar {args.threshold}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
