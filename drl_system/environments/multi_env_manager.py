"""Management utilities for multi-environment and asynchronous execution."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np


@dataclass
class EnvironmentFactory:
    make_env: Callable[[], "Environment"]


class Environment:
    """Minimal environment protocol used for abstract integration."""

    def reset(self) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class MultiEnvManager:
    factory: EnvironmentFactory
    num_envs: int
    asynchronous: bool = True
    _threads: List[threading.Thread] = field(init=False, default_factory=list)
    _queues: List[queue.Queue] = field(init=False, default_factory=list)
    _results: List[queue.Queue] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        for _ in range(self.num_envs):
            self._queues.append(queue.Queue(maxsize=1))
            self._results.append(queue.Queue(maxsize=1))
            if self.asynchronous:
                thread = threading.Thread(target=self._worker, args=(self.factory.make_env(), len(self._queues) - 1))
                thread.daemon = True
                thread.start()
                self._threads.append(thread)

    def _worker(self, env: Environment, index: int) -> None:
        obs = env.reset()
        self._results[index].put((obs, 0.0, False, {}))
        while True:
            action = self._queues[index].get()
            if action is None:
                obs = env.reset()
                self._results[index].put((obs, 0.0, False, {"reset": True}))
                continue
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
            self._results[index].put((obs, reward, done, info))

    def step(self, actions: List[np.ndarray]) -> List[tuple[np.ndarray, float, bool, dict]]:
        if not self.asynchronous:
            env = self.factory.make_env()
            return [env.step(action) for action in actions]

        for q, action in zip(self._queues, actions):
            q.put(action)
        return [result.get() for result in self._results]

    def reset(self) -> List[np.ndarray]:
        if not self.asynchronous:
            env = self.factory.make_env()
            return [env.reset() for _ in range(self.num_envs)]
        resets = []
        for idx, q in enumerate(self._queues):
            q.put(None)
            obs, *_ = self._results[idx].get()
            resets.append(obs)
        return resets


__all__ = ["MultiEnvManager", "Environment", "EnvironmentFactory"]
