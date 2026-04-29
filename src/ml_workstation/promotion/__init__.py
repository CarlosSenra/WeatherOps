__all__ = [
    "export_promoted_model_to_disk",
    "promote_best",
    "promote_run",
    "select_best_run",
    "load_production_model",
    "load_production_scaler",
    "get_production_info",
    "PromotionRejectedError",
    "ModelNotInProductionError",
]

_LAZY: dict[str, tuple[str, str]] = {
    "export_promoted_model_to_disk": ("src.ml_workstation.promotion.export_local", "export_promoted_model_to_disk"),
    "ModelNotInProductionError": ("src.ml_workstation.promotion.loader", "ModelNotInProductionError"),
    "get_production_info": ("src.ml_workstation.promotion.loader", "get_production_info"),
    "load_production_model": ("src.ml_workstation.promotion.loader", "load_production_model"),
    "load_production_scaler": ("src.ml_workstation.promotion.loader", "load_production_scaler"),
    "PromotionRejectedError": ("src.ml_workstation.promotion.promote", "PromotionRejectedError"),
    "promote_best": ("src.ml_workstation.promotion.promote", "promote_best"),
    "promote_run": ("src.ml_workstation.promotion.promote", "promote_run"),
    "select_best_run": ("src.ml_workstation.promotion.promote", "select_best_run"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib  # noqa: PLC0415

        mod_path, attr = _LAZY[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
