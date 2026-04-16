"""Unit tests for src/api/tracing.py."""
from __future__ import annotations

import builtins
import uuid

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.api.tracing import TRACE_ID_HEADER, TraceIDMiddleware, setup_tracing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app() -> Starlette:
    async def homepage(request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/", homepage)])
    app.add_middleware(TraceIDMiddleware)
    return app


# ---------------------------------------------------------------------------
# TraceIDMiddleware
# ---------------------------------------------------------------------------

def test_trace_id_header_is_added_to_response() -> None:
    client = TestClient(_make_app())
    response = client.get("/")
    assert TRACE_ID_HEADER in response.headers


def test_generated_trace_id_is_valid_uuid() -> None:
    client = TestClient(_make_app())
    response = client.get("/")
    trace_id = response.headers[TRACE_ID_HEADER]
    uuid.UUID(trace_id)  # raises ValueError if invalid


def test_incoming_trace_id_is_propagated() -> None:
    client = TestClient(_make_app())
    sent_id = "my-custom-trace-id-9876"
    response = client.get("/", headers={TRACE_ID_HEADER: sent_id})
    assert response.headers[TRACE_ID_HEADER] == sent_id


def test_different_requests_get_different_trace_ids() -> None:
    client = TestClient(_make_app())
    r1 = client.get("/")
    r2 = client.get("/")
    assert r1.headers[TRACE_ID_HEADER] != r2.headers[TRACE_ID_HEADER]


# ---------------------------------------------------------------------------
# setup_tracing
# ---------------------------------------------------------------------------

def test_setup_tracing_without_endpoint_does_not_raise() -> None:
    setup_tracing(service_name="test-service", otlp_endpoint=None)


def test_setup_tracing_degrades_gracefully_without_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ImportError("mocked: opentelemetry not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)
    # Must not raise — graceful degradation
    setup_tracing(service_name="test-svc", otlp_endpoint="http://otel:4318")
