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


def test_dataset_builder_chunking(tmp_path: Path):
    config = DatasetConfig(root=str(tmp_path), version="chunked", chunk_size=4, chunk_overlap=2)
    builder = SyntheticDatasetBuilder(config)
    builder.generate(num_samples=10, obs_dim=2)
    chunks = list(builder.iter_chunks())
    # Overlap ensures more than two chunks for 10 samples when chunk size is 4 and overlap 2
    assert len(chunks) >= 3
    for obs_chunk, action_chunk, reward_chunk in chunks:
        assert obs_chunk.shape[0] <= 4
        assert action_chunk.shape[0] == obs_chunk.shape[0]
        assert reward_chunk.shape[0] == obs_chunk.shape[0]
