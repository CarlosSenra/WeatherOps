"""Live HTTP smoke against a running API (set ``WEATHEROPS_API_BASE_URL``).

Run::

    pytest test/integration/test_live_api_smoke.py -m live_api -s

Or use ``python scripts/smoke_api.py`` (same checks).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live_api

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_smoke_module():
    path = _REPO_ROOT / "scripts" / "smoke_api.py"
    spec = importlib.util.spec_from_file_location("smoke_api", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_live_api_smoke_http() -> None:
    base = os.environ.get("WEATHEROPS_API_BASE_URL", "").strip()
    if not base:
        pytest.skip("Set WEATHEROPS_API_BASE_URL to run live HTTP smoke (see scripts/smoke_api.py).")

    mod = _load_smoke_module()
    ref = os.environ.get("SMOKE_REFERENCE_DATE", "2024-06-01")
    horizon = int(os.environ.get("SMOKE_FORECAST_HORIZON", "72"))
    skip_agent = os.environ.get("SMOKE_SKIP_AGENT", "").strip() in ("1", "true", "yes")

    mod.run_smoke(
        base,
        reference_date=ref,
        horizon=horizon,
        skip_agent=skip_agent,
    )
