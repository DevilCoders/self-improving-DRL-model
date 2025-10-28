"""Replay buffer implementations for off-policy and on-policy workflows."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Tuple

import numpy as np
import torch


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any]


class ReplayBuffer:
    """Standard replay buffer with optional prioritization hooks."""

    def __init__(self, capacity: int, prioritized: bool = False) -> None:
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.prioritized = prioritized
        self.priorities: Deque[float] = deque(maxlen=capacity) if prioritized else None

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buffer)

    def push(self, transition: Transition, priority: float = 1.0) -> None:
        self.buffer.append(transition)
        if self.prioritized and self.priorities is not None:
            self.priorities.append(priority)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough samples to draw from the buffer")

        if not self.prioritized:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        else:
            priorities = np.array(self.priorities, dtype=np.float64)
            priorities = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=priorities)

        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in indices))
        return (
            torch.from_numpy(np.stack(states)).float(),
            torch.from_numpy(np.stack(actions)).float(),
            torch.from_numpy(np.asarray(rewards, dtype=np.float32)).unsqueeze(-1),
            torch.from_numpy(np.stack(next_states)).float(),
            torch.from_numpy(np.asarray(dones, dtype=np.float32)).unsqueeze(-1),
        )

    def iterate(self, batch_size: int) -> Iterable[Tuple[torch.Tensor, ...]]:
        for start in range(0, len(self.buffer), batch_size):
            end = min(start + batch_size, len(self.buffer))
            if end - start < batch_size:
                break
            yield self.sample(batch_size)


class EpisodicMemory:
    """Stores complete episodes for long-term meta learning."""

    def __init__(self, max_episodes: int = 1_000) -> None:
        self.max_episodes = max_episodes
        self.episodes: Deque[List[Transition]] = deque(maxlen=max_episodes)

    def add_episode(self, transitions: List[Transition]) -> None:
        if not transitions:
            raise ValueError("Episode must contain at least one transition")
        self.episodes.append(list(transitions))

    def sample_episode(self) -> List[Transition]:
        if not self.episodes:
            raise ValueError("No episodes stored")
        idx = np.random.randint(0, len(self.episodes))
        return list(self.episodes[idx])

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.episodes)


__all__ = ["Transition", "ReplayBuffer", "EpisodicMemory"]
