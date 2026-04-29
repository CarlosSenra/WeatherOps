"""Utilities de avaliacao para modelos treinados no MLflow."""

__all__ = ["evaluate_run"]


def __getattr__(name: str):
    if name == "evaluate_run":
        from src.ml_workstation.evaluation.core import evaluate_run  # noqa: PLC0415

        return evaluate_run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
