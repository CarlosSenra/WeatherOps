"""Endpoints internos para atualização de métricas Prometheus via scripts externos."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from src.api.metrics import agent_ragas_score, query_semantic_drift_score

router = APIRouter(prefix="/v1/internal", tags=["internal"])


class RagasScoresPayload(BaseModel):
    scores: dict[str, float]


class DriftScorePayload(BaseModel):
    score: float


@router.post("/ragas", summary="Atualiza gauges RAGAS no registry Prometheus da API")
async def update_ragas_scores(payload: RagasScoresPayload) -> dict:
    for metric_name, value in payload.scores.items():
        agent_ragas_score.labels(metric_name=metric_name).set(value)
    return {"updated": list(payload.scores.keys())}


@router.post("/drift", summary="Atualiza gauge de drift semântico no registry Prometheus da API")
async def update_drift_score(payload: DriftScorePayload) -> dict:
    query_semantic_drift_score.set(payload.score)
    return {"updated": "weatherops_query_semantic_drift_score", "value": payload.score}
