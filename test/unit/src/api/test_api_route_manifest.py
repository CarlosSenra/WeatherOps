"""Guard: API exposes the same HTTP surface as the production factory (routers + metrics + docs)."""
from __future__ import annotations

from collections.abc import Iterable

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.routing import Mount, Route

from src.api.metrics import setup_metrics
from src.api.routers import forecast, health


def _build_mirror_app(*, include_agent: bool) -> FastAPI:
    """Mirrors router registration in ``src.api.main.create_app`` (without lifespan or CORS)."""
    app = FastAPI(docs_url="/docs", redoc_url="/redoc")
    app.include_router(health.router)
    app.include_router(forecast.router)
    if include_agent:
        from src.api_agent.routers.agent import router as agent_router  # noqa: PLC0415

        app.include_router(agent_router)
    setup_metrics(app)
    return app


def _agent_router_available() -> bool:
    """Same try/import as ``src.api.main.create_app`` for the agent router."""
    try:
        from src.api_agent.routers.agent import router as _agent_router  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def _method_path_pairs(app: FastAPI) -> set[tuple[str, str]]:
    """Collect (method, path) including routes mounted under ``Mount`` (e.g. ``/metrics``)."""
    out: set[tuple[str, str]] = set()

    def visit(route: object, prefix: str) -> None:
        if isinstance(route, Mount):
            mount_path = getattr(route, "path", "") or ""
            nested = prefix + mount_path
            for sub in getattr(route, "routes", []) or []:
                visit(sub, nested)
            return
        if isinstance(route, APIRoute):
            path = prefix + (route.path or "")
            for method in route.methods:
                if method in ("HEAD", "OPTIONS"):
                    continue
                out.add((method, path))
            return
        if isinstance(route, Route) and route.methods:
            path = prefix + (route.path or "")
            for method in route.methods:
                if method in ("HEAD", "OPTIONS"):
                    continue
                out.add((method, path))

    for top in app.routes:
        visit(top, "")

    return out


def _frozen_pairs(pairs: Iterable[tuple[str, str]]) -> frozenset[tuple[str, str]]:
    return frozenset(pairs)


def test_api_route_manifest_matches_production_surface() -> None:
    include_agent = _agent_router_available()
    app = _build_mirror_app(include_agent=include_agent)
    actual = _method_path_pairs(app)

    required: list[tuple[str, str]] = [
        ("GET", "/health"),
        ("GET", "/health/ready"),
        ("POST", "/v1/forecast/{horizon}"),
        ("GET", "/metrics"),
        ("GET", "/docs"),
        ("GET", "/redoc"),
        ("GET", "/openapi.json"),
    ]
    if include_agent:
        required.append(("POST", "/v1/agent/chat"))

    missing = _frozen_pairs(required) - actual
    assert not missing, f"Missing routes on mirror app: {sorted(missing)}"
