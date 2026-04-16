from .dataset import WeatherSequenceDataset

__all__ = ["WeatherSequenceDataset", "ParquetDataLoader"]


def __getattr__(name: str):
    if name == "ParquetDataLoader":
        from .loader import ParquetDataLoader  # noqa: PLC0415

        return ParquetDataLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
