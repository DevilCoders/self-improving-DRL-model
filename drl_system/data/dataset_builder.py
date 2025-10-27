"""Synthetic dataset builder for offline experimentation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

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
        self._materialize_chunks(root, observations, actions, rewards)
        return root

    def load(self) -> List[np.ndarray]:
        root = Path(self.config.root) / self.config.version
        return [np.load(root / name) for name in ["observations.npy", "actions.npy", "rewards.npy"]]

    # Chunking -------------------------------------------------------------
    def _chunk_array(self, array: np.ndarray) -> List[np.ndarray]:
        chunk_size = max(1, self.config.chunk_size)
        overlap = max(0, min(self.config.chunk_overlap, chunk_size - 1))
        chunks: List[np.ndarray] = []
        start = 0
        while start < array.shape[0]:
            end = min(start + chunk_size, array.shape[0])
            chunks.append(array[start:end])
            if end == array.shape[0]:
                break
            start = end - overlap
        return chunks

    def _materialize_chunks(
        self,
        root: Path,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> None:
        chunk_root = root / "chunks"
        (chunk_root / "observations").mkdir(parents=True, exist_ok=True)
        (chunk_root / "actions").mkdir(parents=True, exist_ok=True)
        (chunk_root / "rewards").mkdir(parents=True, exist_ok=True)

        for idx, chunk in enumerate(self._chunk_array(observations)):
            np.save(chunk_root / "observations" / f"chunk_{idx:04d}.npy", chunk)
        for idx, chunk in enumerate(self._chunk_array(actions)):
            np.save(chunk_root / "actions" / f"chunk_{idx:04d}.npy", chunk)
        for idx, chunk in enumerate(self._chunk_array(rewards)):
            np.save(chunk_root / "rewards" / f"chunk_{idx:04d}.npy", chunk)

    def iter_chunks(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        root = Path(self.config.root) / self.config.version / "chunks"
        observation_files = sorted((root / "observations").glob("chunk_*.npy"))
        action_files = sorted((root / "actions").glob("chunk_*.npy"))
        reward_files = sorted((root / "rewards").glob("chunk_*.npy"))
        for obs_file, act_file, rew_file in zip(observation_files, action_files, reward_files):
            yield np.load(obs_file), np.load(act_file), np.load(rew_file)


__all__ = ["SyntheticDatasetBuilder"]
