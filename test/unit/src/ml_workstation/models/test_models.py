import pytest
import torch

from src.ml_workstation.config.training_config import ModelConfig
from src.ml_workstation.models import build_model


@pytest.mark.parametrize(
    "model_config",
    [
        ModelConfig(model_type="lstm", hidden_size=32, num_layers=2, dropout=0.1),
        ModelConfig(
            model_type="transformer",
            hidden_size=32,
            num_layers=2,
            dropout=0.1,
            num_heads=4,
            ffn_dim=64,
        ),
    ],
)
def test_model_forward_shape(model_config: ModelConfig) -> None:
    model = build_model(n_features=14, n_targets=2, horizon=12, config=model_config)
    x = torch.randn(5, 24, 14)

    y = model(x)

    assert y.shape == (5, 12, 2)


@pytest.mark.parametrize(
    "model_config",
    [
        ModelConfig(model_type="lstm", hidden_size=16, num_layers=1, dropout=0.0),
        ModelConfig(
            model_type="transformer",
            hidden_size=16,
            num_layers=1,
            dropout=0.0,
            num_heads=4,
            ffn_dim=32,
        ),
    ],
)
def test_model_backward_pass(model_config: ModelConfig) -> None:
    model = build_model(n_features=6, n_targets=1, horizon=4, config=model_config)
    x = torch.randn(3, 10, 6)
    target = torch.randn(3, 4, 1)

    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_model_eval_mode_is_deterministic_when_dropout_zero() -> None:
    config = ModelConfig(
        model_type="transformer",
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        num_heads=4,
        ffn_dim=32,
    )
    model = build_model(n_features=4, n_targets=1, horizon=3, config=config)
    model.eval()
    x = torch.randn(2, 8, 4)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1, out2)


def test_transformer_invalid_heads_raises() -> None:
    config = ModelConfig(
        model_type="transformer",
        hidden_size=30,
        num_layers=1,
        dropout=0.0,
        num_heads=8,
        ffn_dim=64,
    )

    with pytest.raises(Exception):
        _ = build_model(n_features=4, n_targets=1, horizon=2, config=config)
