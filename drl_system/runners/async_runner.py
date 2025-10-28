"""Asynchronous training runner supporting distributed style orchestration."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable


@dataclass
class AsyncRunner:
    target: Callable[[], None]

    def run(self, num_workers: int = 4) -> None:
        threads = []
        for _ in range(num_workers):
            thread = threading.Thread(target=self.target)
            thread.daemon = True
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join(timeout=0.1)


__all__ = ["AsyncRunner"]
