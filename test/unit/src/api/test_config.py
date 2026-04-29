"""Unit tests for src/api/config.py."""
from __future__ import annotations

from src.api.config import (
    ALL_SERVING_COLUMNS,
    DEFAULT_FEATURES,
    DEFAULT_TARGETS,
    REGISTERED_MODELS,
    VALID_HORIZONS,
    VALID_MODEL_TYPES,
    ServingModelConfig,
    Settings,
    get_settings,
)


def test_default_features_has_14_columns() -> None:
    assert len(DEFAULT_FEATURES) == 14


def test_default_features_contains_temp() -> None:
    assert "temp_ar_c" in DEFAULT_FEATURES


def test_default_targets_is_temperature() -> None:
    assert DEFAULT_TARGETS == ["temp_ar_c"]


def test_registered_models_not_empty() -> None:
    assert len(REGISTERED_MODELS) >= 1


def test_all_registered_models_are_serving_model_config() -> None:
    for m in REGISTERED_MODELS:
        assert isinstance(m, ServingModelConfig)


def test_registered_models_have_required_fields() -> None:
    for m in REGISTERED_MODELS:
        assert m.key
        assert m.experiment_name
        assert m.horizon > 0
        assert m.sequence_length > 0
        assert m.engine_class in {"pytorch_forecasting", "sequential"}
        assert m.model_type in {"tft", "nbeats", "lstm", "transformer"}


def test_valid_horizons_matches_registered_model_horizons() -> None:
    expected = frozenset(m.horizon for m in REGISTERED_MODELS)
    assert VALID_HORIZONS == expected


def test_valid_model_types_contains_all_registered_types() -> None:
    for m in REGISTERED_MODELS:
        assert m.model_type in VALID_MODEL_TYPES


def test_all_serving_columns_includes_features_and_targets() -> None:
    for m in REGISTERED_MODELS:
        for col in m.feature_columns:
            assert col in ALL_SERVING_COLUMNS
        for col in m.target_columns:
            assert col in ALL_SERVING_COLUMNS


def test_all_serving_columns_has_no_duplicates() -> None:
    assert len(ALL_SERVING_COLUMNS) == len(set(ALL_SERVING_COLUMNS))


def test_serving_model_config_defaults_features_and_targets() -> None:
    m = ServingModelConfig(
        key="test_72",
        experiment_name="test_exp",
        model_type="lstm",
        horizon=72,
        sequence_length=168,
        engine_class="sequential",
    )
    assert m.feature_columns == list(DEFAULT_FEATURES)
    assert m.target_columns == list(DEFAULT_TARGETS)
    assert m.train_ratio == 0.8
    assert m.val_ratio == 0.1
    assert m.registry_model_name is None


def test_serving_model_config_registry_model_name_explicit() -> None:
    m = ServingModelConfig(
        key="tft_72",
        experiment_name="weather_tft_h72",
        model_type="tft",
        horizon=72,
        sequence_length=168,
        engine_class="pytorch_forecasting",
        registry_model_name="weather_forecasting_h72",
    )
    assert m.registry_model_name == "weather_forecasting_h72"


def test_settings_defaults() -> None:
    s = Settings()
    assert s.device == "cpu"
    assert s.cache_backend == "memory"
    assert s.cache_ttl_seconds == 3600
    assert s.parquet_path == "/app/data/spec"


def test_settings_registered_models_returns_global_list() -> None:
    s = Settings()
    assert s.registered_models is REGISTERED_MODELS


def test_get_settings_is_cached() -> None:
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
