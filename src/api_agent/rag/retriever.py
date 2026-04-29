"""Retriever da base vetorial ChromaDB para o agente WeatherOps.

Carrega o ChromaDB gerado pelo promote e fornece busca semântica por chunks
históricos de clima. Inicializado de forma lazy — retorna None se a base ainda
não foi construída, permitindo que o agente funcione sem RAG (graceful degradation).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "weatherops_features"
_DEFAULT_CHROMA_SUBPATH = "src/api_agent/knowledge/chroma_db"


class FeatureStoreRetriever:
    """Consulta semântica sobre perfis históricos de clima por município."""

    def __init__(self, chroma_path: Path, google_api_key: str) -> None:
        from langchain_chroma import Chroma
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=google_api_key,
        )
        self._db = Chroma(
            collection_name=_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(chroma_path),
        )
        logger.info("FeatureStoreRetriever inicializado a partir de %s.", chroma_path)

    def search(self, query: str, k: int = 3) -> list[str]:
        """Retorna os k chunks de texto mais semanticamente próximos à query."""
        docs = self._db.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    @classmethod
    def from_env(cls, google_api_key: str) -> "FeatureStoreRetriever | None":
        """Cria o retriever a partir da variável KNOWLEDGE_BASE_PATH ou do caminho padrão.

        Retorna None se o diretório ChromaDB não existir, sem lançar exceção.
        """
        raw_path = os.environ.get("KNOWLEDGE_BASE_PATH", "")
        if raw_path:
            chroma_path = Path(raw_path).resolve()
        else:
            workspace = _find_workspace_root()
            chroma_path = (workspace / _DEFAULT_CHROMA_SUBPATH).resolve()

        if not chroma_path.exists():
            logger.info(
                "FeatureStoreRetriever: ChromaDB não encontrado em %s. "
                "Execute o promote com --update-knowledge-base para gerar.",
                chroma_path,
            )
            return None

        try:
            return cls(chroma_path=chroma_path, google_api_key=google_api_key)
        except Exception as exc:
            logger.warning("Falha ao carregar FeatureStoreRetriever: %s", exc)
            return None


def _find_workspace_root() -> Path:
    """Localiza a raiz do repositório subindo a partir deste arquivo."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()
