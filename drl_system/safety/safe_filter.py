"""Safety filtering for action selection with human-in-the-loop overrides."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from ..config import SafeFilterConfig


@dataclass
class SafeActionsFilter:
    config: SafeFilterConfig
    custom_checks: Iterable[Callable[[np.ndarray], bool]] = ()

    def clamp(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.config.min_action, self.config.max_action)

    def is_safe(self, action: np.ndarray) -> bool:
        if self.config.forbidden_zones:
            for low, high in self.config.forbidden_zones:
                if np.all(action >= low) and np.all(action <= high):
                    return False
        for check in self.custom_checks:
            if not check(action):
                return False
        return True

    def filter(self, action: np.ndarray) -> np.ndarray:
        action = self.clamp(action)
        if self.is_safe(action):
            return action
        return np.zeros_like(action)


__all__ = ["SafeActionsFilter"]
