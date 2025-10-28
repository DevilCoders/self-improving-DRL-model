"""Self-improvement loop utilities for continual learning."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from ..config import MetaLearningConfig
from ..optimization.adaptive_optimizer import AdaptiveOptimizer


@dataclass
class SelfImprovementLoop:
    config: MetaLearningConfig
    optimizer: AdaptiveOptimizer
    history: Dict[str, float] = None
    evaluations: Dict[str, float] = None

    def __post_init__(self) -> None:
        self.history = {}
        self.evaluations = {}

    def step(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = value
            else:
                self.history[key] = 0.9 * self.history[key] + 0.1 * value

    def record_evaluation(self, evaluation: Dict[str, float]) -> None:
        for key, value in evaluation.items():
            self.evaluations[key] = value

    def checkpoint(self, step: int, metrics: Dict[str, float], directory: str = "checkpoints") -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        payload = {
            "step": step,
            "metrics": metrics,
            "history": self.history,
            "evaluations": self.evaluations,
        }
        path = Path(directory) / f"self_improvement_{step}.json"
        path.write_text(json.dumps(payload, indent=2))


__all__ = ["SelfImprovementLoop"]
