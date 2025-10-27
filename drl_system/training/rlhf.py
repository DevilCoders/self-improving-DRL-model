"""Human feedback aggregation and integration utilities."""
from __future__ import annotations

import queue
from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np

from ..config import RLHFConfig


@dataclass
class HumanFeedback:
    observation: np.ndarray
    action: np.ndarray
    score: float


@dataclass
class FeedbackBuffer:
    config: RLHFConfig
    _buffer: queue.Queue = field(init=False)

    def __post_init__(self) -> None:
        self._buffer = queue.Queue(maxsize=self.config.human_buffer_size)

    def push(self, feedback: HumanFeedback) -> None:
        if self._buffer.full():
            _ = self._buffer.get()
        self._buffer.put(feedback)

    def collect(self, limit: int | None = None) -> List[HumanFeedback]:
        items: List[HumanFeedback] = []
        while not self._buffer.empty() and (limit is None or len(items) < limit):
            items.append(self._buffer.get())
        return items


def aggregate_feedback(feedback: Iterable[HumanFeedback], config: RLHFConfig) -> float:
    scores = np.array([fb.score for fb in feedback], dtype=np.float32)
    if scores.size == 0:
        return 0.0
    if config.aggregation == "mean":
        return float(scores.mean() * config.reward_scale)
    if config.aggregation == "median":
        return float(np.median(scores) * config.reward_scale)
    raise ValueError(f"Unknown aggregation strategy: {config.aggregation}")


__all__ = ["HumanFeedback", "FeedbackBuffer", "aggregate_feedback"]
