import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.ml_workstation.config.training_config import ModelConfig, TrainingConfig


def test_training_config_defaults_are_stable() -> None:
    config = TrainingConfig()

    assert config.data.sequence_length > 0
    assert config.data.horizon > 0
    assert config.model.model_type in {"lstm", "transformer"}
    assert config.device in {"cpu", "cuda"}


def test_model_config_rejects_unknown_model_type() -> None:
    with pytest.raises(ValidationError):
        _ = ModelConfig(model_type="xgboost")


def test_training_config_loads_real_experiment_json() -> None:
    path = Path("src/ml_workstation/experiments/lstm/lstm_h72_v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    config = TrainingConfig.model_validate(payload)

    assert config.run_name == "lstm_h72_v1"
    assert config.data.horizon == 72
    assert config.model.model_type == "lstm"
