"""Utility helpers for selecting and managing computation devices."""
from __future__ import annotations

import os
from typing import Optional

import torch


class DeviceManager:
    """Selects CPU/GPU targets at runtime and exposes helper utilities."""

    def __init__(self, preferred: str = "auto") -> None:
        self._preferred = preferred
        self._device: Optional[torch.device] = None

    def select(self) -> torch.device:
        if self._device is not None:
            return self._device

        if self._preferred == "cpu":
            self._device = torch.device("cpu")
        elif self._preferred == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            self._device = torch.device("cuda")
        else:
            # Automatic selection
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return self._device

    @property
    def name(self) -> str:
        return self.select().type

    def seed_everything(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)


__all__ = ["DeviceManager"]
