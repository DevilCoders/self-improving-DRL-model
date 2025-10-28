"""Data preprocessing utilities for offline and online pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List

import numpy as np


def normalize_observation(obs: np.ndarray) -> np.ndarray:
    mean = obs.mean(axis=0, keepdims=True)
    std = obs.std(axis=0, keepdims=True) + 1e-6
    return (obs - mean) / std


def clip_rewards(rewards: np.ndarray, min_value: float = -10.0, max_value: float = 10.0) -> np.ndarray:
    return np.clip(rewards, min_value, max_value)


@dataclass
class PreprocessingPipeline:
    steps: List[Callable[[np.ndarray], np.ndarray]]

    def run(self, data: np.ndarray) -> np.ndarray:
        transformed = data
        for step in self.steps:
            transformed = step(transformed)
        return transformed

    @classmethod
    def offline_default(cls) -> "PreprocessingPipeline":
        return cls(steps=[normalize_observation])

    @classmethod
    def online_default(cls) -> "PreprocessingPipeline":
        return cls(steps=[normalize_observation, lambda x: clip_rewards(x, -5.0, 5.0)])


def build_text_corpus(observations: Iterable[str]) -> List[str]:
    return [obs.strip().lower() for obs in observations if obs]


__all__ = [
    "normalize_observation",
    "clip_rewards",
    "PreprocessingPipeline",
    "build_text_corpus",
]
