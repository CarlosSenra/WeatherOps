import numpy as np


def compute_mape(y_pred: np.ndarray, y_true: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Calcula MAPE em porcentagem com estabilizacao no denominador.

    Args:
        y_pred: Previsoes do modelo, shape arbitrario.
        y_true: Valores reais, mesmo shape de y_pred.
        epsilon: Valor minimo para o denominador, evitando explosao perto de zero.

    Returns:
        MAPE em porcentagem.
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    denominator = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs(y_pred - y_true) / denominator) * 100)


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

    mape = compute_mape(y_pred, y_true)

    return {"mae": mae, "rmse": rmse, "mape": mape}
