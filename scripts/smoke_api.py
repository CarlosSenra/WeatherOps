#!/usr/bin/env python3
"""HTTP smoke checks against a running WeatherOps API.

Requires ``httpx`` (dev dependency). Run after ``uvicorn`` or ``docker compose``::

    set WEATHEROPS_API_BASE_URL=http://127.0.0.1:8888
    python scripts/smoke_api.py

Optional: ``SMOKE_REFERENCE_DATE`` (YYYY-MM-DD or ISO), ``SMOKE_FORECAST_HORIZON`` (int).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import httpx


def _base_url(cli_base: str | None) -> str:
    raw = (cli_base or os.environ.get("WEATHEROPS_API_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
    return raw


def _wait_ready(client: httpx.Client, base: str, *, max_wait_s: float = 120.0, interval_s: float = 2.0) -> None:
    deadline = time.monotonic() + max_wait_s
    last_status: int | None = None
    while time.monotonic() < deadline:
        r = client.get(f"{base}/health/ready", timeout=30.0)
        last_status = r.status_code
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "pronto":
                return
        time.sleep(interval_s)
    raise RuntimeError(
        f"Readiness timeout after {max_wait_s:.0f}s (last /health/ready status={last_status}). "
        "Ensure models and Parquet finished loading."
    )


def run_smoke(base: str, *, reference_date: str, horizon: int, skip_agent: bool) -> None:
    """Raises ``RuntimeError`` on failure (for use from tests and ``main()``)."""
    base = base.rstrip("/")
    with httpx.Client(timeout=120.0) as client:
        r_health = client.get(f"{base}/health")
        if r_health.status_code != 200 or r_health.json().get("status") != "ok":
            raise RuntimeError(f"/health unexpected: {r_health.status_code} {r_health.text[:300]}")

        _wait_ready(client, base)

        r_metrics = client.get(f"{base}/metrics")
        if r_metrics.status_code != 200:
            raise RuntimeError(f"/metrics HTTP {r_metrics.status_code}")
        ct = r_metrics.headers.get("content-type", "")
        if "text/plain" not in ct:
            raise RuntimeError(f"/metrics content-type unexpected: {ct!r}")
        if "http_requests_total" not in r_metrics.text and "http_request_duration" not in r_metrics.text:
            raise RuntimeError("/metrics body missing expected Prometheus series")

        body = {
            "reference_date": reference_date,
            "model_type": "tft",
            "group_id": "station_1",
        }
        fc = client.post(f"{base}/v1/forecast/{horizon}", json=body)
        if fc.status_code != 200:
            raise RuntimeError(
                f"POST /v1/forecast/{horizon} -> HTTP {fc.status_code}: {fc.text[:800]}. "
                "Try SMOKE_REFERENCE_DATE within your Parquet range."
            )
        payload = fc.json()
        if "predictions" not in payload or not isinstance(payload["predictions"], list):
            raise RuntimeError("Forecast response missing non-empty predictions list")
        if len(payload["predictions"]) != horizon:
            raise RuntimeError(
                f"Expected {horizon} prediction points, got {len(payload['predictions'])}"
            )

        if skip_agent:
            print("Skipping /v1/agent/chat (--skip-agent or SMOKE_SKIP_AGENT=1).")
            return

        ag = client.post(f"{base}/v1/agent/chat", json={"message": "Responde só: ok."})
        if ag.status_code == 200:
            print("Agent /v1/agent/chat: 200")
            return
        if ag.status_code == 503:
            try:
                detail = ag.json().get("detail", "")
            except Exception:
                detail = ag.text
            if "GEMINI" in str(detail).upper() or "desativado" in str(detail).lower():
                print("Agent disabled (503) — acceptable for smoke.")
                return
        if ag.status_code == 422:
            raise RuntimeError(f"/v1/agent/chat 422: {ag.text[:500]}")
        raise RuntimeError(f"/v1/agent/chat unexpected HTTP {ag.status_code}: {ag.text[:500]}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Smoke test a running WeatherOps API over HTTP.")
    p.add_argument(
        "--base-url",
        default=None,
        help="Override WEATHEROPS_API_BASE_URL (default http://127.0.0.1:8000).",
    )
    p.add_argument(
        "--reference-date",
        default=os.environ.get("SMOKE_REFERENCE_DATE", "2024-06-01"),
        help="Forecast reference_date (default SMOKE_REFERENCE_DATE or 2024-06-01).",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=int(os.environ.get("SMOKE_FORECAST_HORIZON", "72")),
        help="Forecast horizon path segment (default 72 or SMOKE_FORECAST_HORIZON).",
    )
    p.add_argument(
        "--skip-agent",
        action="store_true",
        help="Do not call /v1/agent/chat.",
    )
    args = p.parse_args(argv)

    skip_agent = args.skip_agent or os.environ.get("SMOKE_SKIP_AGENT", "").strip() in ("1", "true", "yes")
    base = _base_url(args.base_url)
    try:
        run_smoke(base, reference_date=args.reference_date, horizon=args.horizon, skip_agent=skip_agent)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    print("Smoke OK:", base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
