#!/usr/bin/env python3
"""Compara comprimentos de queries a um ficheiro opcional vs golden set baseline."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "src" / "api_agent" / "eval" / "golden_set.jsonl"


def _baseline_lengths() -> list[int]:
    out: list[int] = []
    with GOLDEN.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(len(row.get("user_input", "")))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--queries-file", type=Path, default=None, help="Um texto por linha (queries reais).")
    args = p.parse_args()

    base = _baseline_lengths()
    print("Golden set (baseline):")
    print(f"  n={len(base)} mean={statistics.mean(base):.1f} stdev={statistics.pstdev(base):.1f} p95={sorted(base)[int(0.95 * (len(base)-1))]}")

    if args.queries_file and args.queries_file.is_file():
        lines = [ln.strip() for ln in args.queries_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        lens = [len(x) for x in lines]
        print("Ficheiro de queries:")
        print(f"  n={len(lens)} mean={statistics.mean(lens):.1f} stdev={statistics.pstdev(lens):.1f} p95={sorted(lens)[int(0.95 * (len(lens)-1))]}")
    else:
        print("(Sem --queries-file: apenas baseline.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
