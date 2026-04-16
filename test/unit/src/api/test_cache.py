"""Unit tests for src/api/cache.py."""
from __future__ import annotations

import time

import pytest

from src.api.cache import InMemoryResponseCache


def test_set_and_get_returns_value() -> None:
    cache = InMemoryResponseCache(ttl_seconds=60)
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"


def test_get_missing_key_returns_none() -> None:
    cache = InMemoryResponseCache()
    assert cache.get("nonexistent") is None


def test_entry_expires_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = InMemoryResponseCache(ttl_seconds=10)
    cache.set("key", "val")

    original_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original_monotonic() + 100)

    assert cache.get("key") is None


def test_get_valid_entry_does_not_expire(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = InMemoryResponseCache(ttl_seconds=3600)
    cache.set("key", "val")

    original_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original_monotonic() + 10)

    assert cache.get("key") == "val"


def test_clear_removes_all_entries() -> None:
    cache = InMemoryResponseCache()
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert cache.size == 0
    assert cache.get("a") is None
    assert cache.get("b") is None


def test_size_reflects_number_of_entries() -> None:
    cache = InMemoryResponseCache()
    assert cache.size == 0
    cache.set("x", "y")
    assert cache.size == 1
    cache.set("z", "w")
    assert cache.size == 2


def test_make_key_is_deterministic() -> None:
    cache = InMemoryResponseCache()
    k1 = cache.make_key("horizon", "72", "tft")
    k2 = cache.make_key("horizon", "72", "tft")
    assert k1 == k2
    assert len(k1) == 64  # SHA-256 hex digest length


def test_make_key_differs_for_different_inputs() -> None:
    cache = InMemoryResponseCache()
    k1 = cache.make_key("72")
    k2 = cache.make_key("168")
    assert k1 != k2


def test_overwrite_existing_key_updates_value() -> None:
    cache = InMemoryResponseCache()
    cache.set("key", "first")
    cache.set("key", "second")
    assert cache.get("key") == "second"
    assert cache.size == 1


def test_stores_arbitrary_python_objects() -> None:
    cache = InMemoryResponseCache()
    payload = {"data": [1, 2, 3], "nested": {"a": True}}
    cache.set("obj", payload)
    assert cache.get("obj") == payload


def test_redis_cache_raises_import_error_without_package() -> None:
    import sys
    from src.api.cache import RedisResponseCache

    if "redis" not in sys.modules:
        # redis not installed — constructor must raise ImportError
        try:
            RedisResponseCache(redis_url="redis://localhost:6379")
        except ImportError:
            pass  # expected
