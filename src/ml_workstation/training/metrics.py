import numpy as np


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """
    Calcula métricas de avaliação para previsão de séries temporais.

    Args:
        y_pred: Previsões do modelo, shape arbitrário.
        y_true: Valores reais, mesmo shape de y_pred.

    Returns:
        Dicionário com mae, rmse e mape.
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    mask = y_true != 0
    if mask.any():
        mape = float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100)
    else:
        mape = float("nan")

    return {"mae": mae, "rmse": rmse, "mape": mape}
