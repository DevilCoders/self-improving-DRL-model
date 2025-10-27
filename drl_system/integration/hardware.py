"""Hardware integration helpers for Raspberry Pi and Arduino targets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class HardwareInterface(Protocol):
    def write(self, command: str) -> None:  # pragma: no cover - interface
        ...

    def read(self) -> str:  # pragma: no cover - interface
        ...


@dataclass
class RaspberryPiController:
    interface: HardwareInterface

    def send_pwm(self, value: float) -> None:
        self.interface.write(f"PWM:{value:.3f}")

    def read_sensor(self) -> float:
        response = self.interface.read()
        try:
            return float(response)
        except ValueError:
            return 0.0


@dataclass
class ArduinoController:
    interface: HardwareInterface

    def send_command(self, command: str) -> None:
        self.interface.write(f"CMD:{command}")

    def read_state(self) -> str:
        return self.interface.read()


__all__ = ["RaspberryPiController", "ArduinoController", "HardwareInterface"]
