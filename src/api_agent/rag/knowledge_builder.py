"""Constrói a base vetorial ChromaDB a partir do feature store gerado pelo promote.

Uso típico (linha de comando via run_promote.py --update-knowledge-base):
    build_knowledge_base(feature_store_dir, chroma_path, google_api_key)

Cada promote gera chunks de texto em português descrevendo os perfis mensais
e extremos sazonais do município. Os chunks são incorporados com Google Embeddings
(models/text-embedding-004) e persistidos no ChromaDB.

A operação é idempotente: documentos do mesmo município são removidos antes de
reinserir, permitindo re-execução sem duplicatas.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_MONTH_NAMES = {
    1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril",
    5: "maio", 6: "junho", 7: "julho", 8: "agosto",
    9: "setembro", 10: "outubro", 11: "novembro", 12: "dezembro",
}

_COLLECTION_NAME = "weatherops_features"


def _generate_monthly_chunk(
    month: int,
    municipio: str,
    profiles: dict,
    extremes: dict,
) -> str:
    month_name = _MONTH_NAMES.get(month, str(month))
    parts = [f"Em {month_name} em {municipio}:"]

    temp = profiles.get("temp_ar_c")
    temp_ext = extremes.get("temp_ar_c", {})
    if temp:
        parts.append(
            f"Temperatura média de {temp['mean']}°C "
            f"(mín: {temp['min']}°C, máx: {temp['max']}°C, desvio: {temp['std']}°C). "
            f"Percentil 5%: {temp_ext.get('p5', '?')}°C, percentil 95%: {temp_ext.get('p95', '?')}°C."
        )

    umidade = profiles.get("umidade_rel_ar_percent")
    if umidade:
        parts.append(
            f"Umidade relativa média de {umidade['mean']}% "
            f"(mín: {umidade['min']}%, máx: {umidade['max']}%)."
        )

    pressao = profiles.get("pressao_atm_estacao_mb")
    if pressao:
        parts.append(f"Pressão atmosférica média de {pressao['mean']} mb.")

    chuva = profiles.get("precipitacao_total_mm")
    if chuva:
        parts.append(
            f"Precipitação: média de {chuva['mean']} mm, máximo de {chuva['max']} mm."
        )

    return " ".join(parts)


def _generate_annual_summary(municipio: str, monthly_profiles: dict, metadata: dict) -> str:
    temp_by_month = {
        int(k): v["temp_ar_c"]["mean"]
        for k, v in monthly_profiles.items()
        if "temp_ar_c" in v
    }
    if not temp_by_month:
        return ""

    hottest = max(temp_by_month, key=lambda m: temp_by_month[m])
    coldest = min(temp_by_month, key=lambda m: temp_by_month[m])
    date_range = metadata.get("date_range", {})

    return (
        f"Resumo climático anual de {municipio}: "
        f"mês mais quente é {_MONTH_NAMES.get(hottest, str(hottest))} "
        f"(média {temp_by_month[hottest]}°C), "
        f"mês mais frio é {_MONTH_NAMES.get(coldest, str(coldest))} "
        f"(média {temp_by_month[coldest]}°C). "
        f"Histórico de {date_range.get('start', '?')[:7]} a {date_range.get('end', '?')[:7]}, "
        f"total de {metadata.get('row_count', 0):,} observações horárias."
    )


def build_knowledge_base(
    feature_store_dir: Path,
    chroma_path: Path,
    google_api_key: str,
) -> int:
    """Constrói (ou atualiza) a base vetorial ChromaDB a partir do feature store.

    Args:
        feature_store_dir: Diretório com ``metadata.json``, ``monthly_profiles.json``
                           e ``seasonal_extremes.json`` gerados por ``snapshot_features``.
        chroma_path:       Diretório de persistência do ChromaDB.
        google_api_key:    Chave da API Google para GoogleGenerativeAIEmbeddings.

    Returns:
        Número de documentos inseridos.
    """
    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    with (feature_store_dir / "metadata.json").open(encoding="utf-8") as f:
        metadata = json.load(f)
    with (feature_store_dir / "monthly_profiles.json").open(encoding="utf-8") as f:
        monthly_profiles = json.load(f)
    with (feature_store_dir / "seasonal_extremes.json").open(encoding="utf-8") as f:
        seasonal_extremes = json.load(f)

    municipio: str = metadata["municipio"]

    texts: list[str] = []
    doc_metadatas: list[dict] = []

    for month_str, profiles in monthly_profiles.items():
        month = int(month_str)
        extremes = seasonal_extremes.get(month_str, {})
        chunk = _generate_monthly_chunk(month, municipio, profiles, extremes)
        texts.append(chunk)
        doc_metadatas.append({
            "municipio": municipio,
            "mes": month,
            "mes_nome": _MONTH_NAMES.get(month, month_str),
        })

    summary = _generate_annual_summary(municipio, monthly_profiles, metadata)
    if summary:
        texts.append(summary)
        doc_metadatas.append({"municipio": municipio, "mes": 0, "mes_nome": "anual"})

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key,
    )

    chroma_path.mkdir(parents=True, exist_ok=True)
    db = Chroma(
        collection_name=_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(chroma_path),
    )

    existing = db.get(where={"municipio": municipio})
    if existing["ids"]:
        db.delete(ids=existing["ids"])
        logger.info(
            "Removidos %d documentos existentes para '%s' antes de reinserir.",
            len(existing["ids"]),
            municipio,
        )

    db.add_texts(texts=texts, metadatas=doc_metadatas)

    logger.info(
        "Base vetorial atualizada: %d chunks inseridos para '%s' em %s.",
        len(texts),
        municipio,
        chroma_path,
    )
    return len(texts)
