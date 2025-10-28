"""Deliberation and thinking utilities supporting deep reasoning traces."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config import ThinkingConfig


@dataclass
class DeliberationStep:
    """Single entry in a deliberation trace."""

    content: str
    timestamp: float
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class DeliberationTrace:
    """Stores the structured thinking process for a task."""

    trace_id: str
    task: str
    mode: str
    steps: List[DeliberationStep] = field(default_factory=list)
    summary: Optional[str] = None

    def add_step(self, content: str, **metadata: object) -> None:
        self.steps.append(DeliberationStep(content=content, timestamp=time.time(), metadata=metadata))


class DeliberationEngine:
    """Manages thinking traces in multiple modes (extensive, deep, extended)."""

    def __init__(self, config: ThinkingConfig) -> None:
        self.config = config
        self._traces: Dict[str, DeliberationTrace] = {}
        self._lock = threading.Lock()

    def start_trace(self, task: str, mode: Optional[str] = None) -> DeliberationTrace:
        if not self.config.enable:
            # Traces are disabled; return a lightweight placeholder.
            trace = DeliberationTrace(trace_id=str(uuid.uuid4()), task=task, mode=mode or self.config.default_mode)
            with self._lock:
                self._traces[trace.trace_id] = trace
            return trace

        selected_mode = mode or self.config.default_mode
        if selected_mode not in self.config.modes:
            selected_mode = self.config.default_mode

        trace = DeliberationTrace(trace_id=str(uuid.uuid4()), task=task, mode=selected_mode)
        with self._lock:
            self._traces[trace.trace_id] = trace
        trace.add_step(f"Trace started for task '{task}' in {selected_mode} mode.", stage="start")
        return trace

    def add_step(self, trace_id: str, content: str, **metadata: object) -> None:
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                return
            if self.config.enable and len(trace.steps) >= self.config.max_steps:
                return
            trace.add_step(content, **metadata)

    def summarize(self, trace_id: str) -> Optional[str]:
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                return None
            if not trace.steps:
                trace.summary = ""
                return trace.summary
            summary = "; ".join(step.content for step in trace.steps[-5:])
            trace.summary = summary
            return summary

    def record_metrics(self, metrics: Dict[str, float], description: str = "metrics") -> Optional[str]:
        trace = self.start_trace(description, mode="deep")
        for key, value in metrics.items():
            self.add_step(trace.trace_id, f"{key}={value:.6f}", stage="metric")
        if self.config.auto_summarize:
            return self.summarize(trace.trace_id)
        return None

    def get_trace(self, trace_id: str) -> Optional[DeliberationTrace]:
        with self._lock:
            return self._traces.get(trace_id)

    def all_traces(self) -> List[DeliberationTrace]:
        with self._lock:
            return list(self._traces.values())


__all__ = ["DeliberationEngine", "DeliberationTrace", "DeliberationStep"]
