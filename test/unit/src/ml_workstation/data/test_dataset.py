import numpy as np
import torch
from torch.utils.data import DataLoader

from src.ml_workstation.config.training_config import DataConfig
from src.ml_workstation.data.dataset import WeatherSequenceDataset


def test_dataset_length_and_shapes() -> None:
    data = np.random.rand(100, 5).astype(np.float32)
    config = DataConfig(sequence_length=24, horizon=10)
    ds = WeatherSequenceDataset(data=data, config=config, target_indices=[0])

    assert len(ds) == 67

    x, y = ds[0]
    assert x.shape == (24, 5)
    assert y.shape == (10, 1)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_dataset_last_valid_index() -> None:
    data = np.random.rand(50, 4).astype(np.float32)
    config = DataConfig(sequence_length=12, horizon=6)
    ds = WeatherSequenceDataset(data=data, config=config, target_indices=[1, 3])

    x, y = ds[len(ds) - 1]

    assert x.shape == (12, 4)
    assert y.shape == (6, 2)


def test_dataset_dataloader_batching() -> None:
    data = np.random.rand(64, 6).astype(np.float32)
    config = DataConfig(sequence_length=8, horizon=4)
    ds = WeatherSequenceDataset(data=data, config=config, target_indices=[0, 2])

    loader = DataLoader(ds, batch_size=16, shuffle=False)
    batch_x, batch_y = next(iter(loader))

    assert batch_x.shape == (16, 8, 6)
    assert batch_y.shape == (16, 4, 2)
