"""Synthetic dataset builder for offline experimentation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

from ..config import DatasetConfig


@dataclass
class SyntheticDatasetBuilder:
    config: DatasetConfig

    def generate(self, num_samples: int = 1000, obs_dim: int = 8) -> Path:
        root = Path(self.config.root) / self.config.version
        root.mkdir(parents=True, exist_ok=True)
        observations = np.random.randn(num_samples, obs_dim).astype("float32")
        actions = np.random.randint(0, 4, size=(num_samples, 1)).astype("int64")
        rewards = np.random.randn(num_samples, 1).astype("float32")
        np.save(root / "observations.npy", observations)
        np.save(root / "actions.npy", actions)
        np.save(root / "rewards.npy", rewards)
        return root

    def load(self) -> List[np.ndarray]:
        root = Path(self.config.root) / self.config.version
        return [np.load(root / name) for name in ["observations.npy", "actions.npy", "rewards.npy"]]


__all__ = ["SyntheticDatasetBuilder"]
