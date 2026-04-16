"""Configuração de rastreamento distribuído.

Fornece duas camadas de observabilidade:

1. **Cabeçalho Trace ID** — sempre ativo. Cada resposta carrega um cabeçalho
   ``X-Trace-Id`` UUID, permitindo correlação de logs mesmo sem um coletor
   OTel completo.

2. **OpenTelemetry** — ativado quando a variável de ambiente
   ``OTEL_EXPORTER_OTLP_ENDPOINT`` está definida. Requer os pacotes opcionais:

       pip install opentelemetry-sdk \\
                   opentelemetry-instrumentation-fastapi \\
                   opentelemetry-exporter-otlp-proto-http

   Caso esses pacotes não estejam instalados, a API inicia normalmente apenas
   com o cabeçalho Trace ID ativo (degradação graciosa).

Middleware
----------
``TraceIDMiddleware`` é um middleware ASGI que:
- Gera um UUID de trace por requisição (ou reutiliza ``X-Trace-Id`` do
  cabeçalho de entrada, permitindo propagação upstream).
- Anexa o trace ID aos cabeçalhos da resposta.
- Armazena o trace ID em ``request.state.trace_id`` para uso em logs.
"""
from __future__ import annotations

import logging
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

TRACE_ID_HEADER = "X-Trace-Id"


# ---------------------------------------------------------------------------
# Middleware de Trace ID — sem dependências externas
# ---------------------------------------------------------------------------


class TraceIDMiddleware(BaseHTTPMiddleware):
    """Injeta um trace ID único em cada ciclo de requisição/resposta."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        trace_id = request.headers.get(TRACE_ID_HEADER) or str(uuid.uuid4())
        request.state.trace_id = trace_id

        response = await call_next(request)
        response.headers[TRACE_ID_HEADER] = trace_id
        return response


# ---------------------------------------------------------------------------
# Configuração do OpenTelemetry (opcional, degradação graciosa se não instalado)
# ---------------------------------------------------------------------------


def setup_tracing(service_name: str, otlp_endpoint: str | None) -> None:
    """Configura o rastreamento OpenTelemetry se o endpoint estiver definido.

    Seguro de chamar quando os pacotes OTel não estão instalados — registra
    um aviso e retorna sem levantar exceção.

    Argumentos:
        service_name:   Identifica este serviço na interface do coletor.
        otlp_endpoint:  Endpoint OTLP HTTP, ex.:
                        ``http://otel-collector:4318``. Quando ``None``,
                        o OTel não é configurado.
    """
    if not otlp_endpoint:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT não definido — rastreamento desativado.")
        return

    try:
        from opentelemetry import trace  # type: ignore[import]
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import]
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore[import]
        from opentelemetry.sdk.resources import Resource  # type: ignore[import]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import]
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import]
    except ImportError:
        logger.warning(
            "Pacotes OpenTelemetry não instalados. "
            "Instale opentelemetry-sdk, opentelemetry-instrumentation-fastapi e "
            "opentelemetry-exporter-otlp-proto-http para habilitar o rastreamento completo. "
            "Continuando apenas com o cabeçalho X-Trace-Id."
        )
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument()

    logger.info(
        "Rastreamento OpenTelemetry habilitado — service='%s', endpoint='%s'",
        service_name,
        otlp_endpoint,
    )
