from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from drl_system.config import DatasetConfig
from drl_system.data.dataset_builder import SyntheticDatasetBuilder


def test_dataset_builder_generates(tmp_path: Path):
    config = DatasetConfig(root=str(tmp_path), version="test")
    builder = SyntheticDatasetBuilder(config)
    root = builder.generate(num_samples=10, obs_dim=2)
    obs, actions, rewards = builder.load()
    assert obs.shape == (10, 2)
    assert actions.shape[0] == 10
    assert rewards.shape[0] == 10
    assert root.exists()
