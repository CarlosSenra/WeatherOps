"""Popula (ou atualiza) o ChromaDB a partir do feature store existente."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api_agent.rag.knowledge_builder import build_knowledge_base

FEATURE_STORE = Path("src/api/ml_models/weather_forecasting_h72/feature_store")
CHROMA_PATH = Path("src/api_agent/knowledge/chroma_db")


def main() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Erro: variável GOOGLE_API_KEY não definida.")
        sys.exit(1)

    if not FEATURE_STORE.exists():
        print(f"Erro: feature store não encontrado em {FEATURE_STORE}")
        sys.exit(1)

    print("Construindo knowledge base...")
    n = build_knowledge_base(
        feature_store_dir=FEATURE_STORE,
        chroma_path=CHROMA_PATH,
        google_api_key=api_key,
    )
    print(f"OK — {n} documentos inseridos em {CHROMA_PATH}")


if __name__ == "__main__":
    main()
