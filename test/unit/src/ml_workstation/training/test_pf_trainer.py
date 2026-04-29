from __future__ import annotations

from src.ml_workstation.training.pf_trainer import _canonicalize_tft_mape


def test_canonicalize_tft_mape_prefers_existing_mape() -> None:
    metrics = {"mape": 9.5, "val_MAPE": 5.0, "val_loss": 0.2}

    result = _canonicalize_tft_mape(metrics)

    assert result["mape"] == 9.5
    assert result["val_MAPE"] == 5.0


def test_canonicalize_tft_mape_uses_val_mape_aliases() -> None:
    result_upper = _canonicalize_tft_mape({"val_MAPE": 7.3, "val_loss": 0.1})
    result_lower = _canonicalize_tft_mape({"val_mape": 6.1})

    assert result_upper["mape"] == 7.3
    assert result_lower["mape"] == 6.1


def test_canonicalize_tft_mape_falls_back_to_mape_scaled() -> None:
    result = _canonicalize_tft_mape({"mape_scaled": 8.8})

    assert result["mape"] == 8.8
