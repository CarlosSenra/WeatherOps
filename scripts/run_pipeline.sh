#!/usr/bin/env bash
set -euo pipefail

MODE="bootstrap"
DEVICE="auto"
START_YEAR="2024"
END_YEAR="2026"
SKIP_API=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --start-year)
      START_YEAR="$2"
      shift 2
      ;;
    --end-year)
      END_YEAR="$2"
      shift 2
      ;;
    --skip-api)
      SKIP_API="--skip-api"
      shift
      ;;
    *)
      echo "Argumento desconhecido: $1"
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "Executando user pipeline..."
poetry run python -m src.tools.user_pipeline \
  --mode "$MODE" \
  --device "$DEVICE" \
  --start-year "$START_YEAR" \
  --end-year "$END_YEAR" \
  $SKIP_API

