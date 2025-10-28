"""Self-adaptive optimizer wrapper supporting meta-updates."""
from __future__ import annotations

from typing import Dict

from torch import optim


class AdaptiveOptimizer:
    """Wraps a PyTorch optimizer with lightweight meta-learning behavior."""

    def __init__(self, optimizer: optim.Optimizer, meta_lr: float = 1e-4) -> None:
        self.optimizer = optimizer
        self.meta_lr = meta_lr
        self._step_counter = 0

    def zero_grad(self) -> None:  # pragma: no cover - trivial
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()
        self._step_counter += 1
        if self._step_counter % 100 == 0:
            for group in self.optimizer.param_groups:
                group["lr"] = group["lr"] * (1.0 - self.meta_lr)

    def state_dict(self) -> Dict[str, object]:  # pragma: no cover - passthrough
        return self.optimizer.state_dict()

    def load_state_dict(self, state: Dict[str, object]) -> None:  # pragma: no cover - passthrough
        self.optimizer.load_state_dict(state)


__all__ = ["AdaptiveOptimizer"]
