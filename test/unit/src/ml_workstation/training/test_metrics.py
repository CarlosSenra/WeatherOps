import numpy as np

from src.ml_workstation.training.metrics import compute_mape, compute_metrics


def test_compute_metrics_zero_error_returns_zeros() -> None:
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = y_true.copy()

    metrics = compute_metrics(y_pred, y_true)

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mape"] == 0.0


def test_compute_mape_handles_near_zero_targets() -> None:
    y_true = np.array([0.0, 1e-12, -1e-12, 2.0])
    y_pred = np.array([1.0, 1.0, 1.0, 1.0])

    mape = compute_mape(y_pred, y_true)

    assert np.isfinite(mape)
    assert mape > 0


def test_compute_metrics_flattens_arbitrary_shapes() -> None:
    y_true = np.ones((2, 3, 1), dtype=np.float64)
    y_pred = np.zeros((2, 3, 1), dtype=np.float64)

    metrics = compute_metrics(y_pred, y_true)

    assert metrics["mae"] == 1.0
    assert metrics["rmse"] == 1.0
    assert metrics["mape"] == 100.0


def test_compute_metrics_high_error_case() -> None:
    y_true = np.array([10.0, 10.0, 10.0])
    y_pred = np.array([30.0, -10.0, 50.0])

    metrics = compute_metrics(y_pred, y_true)

    assert metrics["mae"] > 20.0
    assert metrics["rmse"] > 20.0
    assert metrics["mape"] > 200.0
