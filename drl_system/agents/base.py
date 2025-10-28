"""Protocol definitions for reinforcement learning agents."""
from __future__ import annotations

from typing import Iterable, Protocol

import torch

from ..memory.replay_buffer import Transition
from .ppo_agent import PPOBatch


class AgentProtocol(Protocol):
    """Interface required by the high-level trainer."""

    device: torch.device
    model: torch.nn.Module

    def compute_advantages(
        self,
        transitions: Iterable[Transition],
        gamma: float,
        lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def prepare_batch(
        self,
        transitions: Iterable[Transition],
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> PPOBatch:
        ...

    def update(self, batch: PPOBatch, **kwargs) -> dict:
        ...


__all__ = ["AgentProtocol"]
