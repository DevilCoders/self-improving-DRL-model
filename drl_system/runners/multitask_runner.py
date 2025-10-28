"""Threaded multi-task runner providing cooperative scheduling."""
from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, List, Optional


@dataclass
class ScheduledResult:
    description: str
    duration: float
    result: object


@dataclass
class MultiTaskScheduler:
    """Runs callables concurrently and aggregates their results."""

    max_workers: int = 4
    _executor: ThreadPoolExecutor = field(init=False)
    _futures: List[Future] = field(init=False, default_factory=list)
    _lock: Lock = field(init=False, default_factory=Lock)

    def __post_init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="scheduler")

    def schedule(
        self,
        fn: Callable[..., object],
        *args: object,
        description: str = "task",
        **kwargs: object,
    ) -> Future:
        def wrapped() -> ScheduledResult:
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            duration = time.perf_counter() - start
            return ScheduledResult(description=description, duration=duration, result=result)

        future = self._executor.submit(wrapped)
        with self._lock:
            self._futures.append(future)
        return future

    def gather(self, timeout: Optional[float] = None) -> List[ScheduledResult]:
        with self._lock:
            futures = list(self._futures)
            self._futures.clear()
        if not futures:
            return []
        done, pending = wait(futures, timeout=timeout)
        results = [future.result() for future in done]
        with self._lock:
            self._futures.extend(pending)
        return results

    def shutdown(self, wait_for_tasks: bool = True) -> None:
        self._executor.shutdown(wait=wait_for_tasks)


__all__ = ["MultiTaskScheduler", "ScheduledResult"]
