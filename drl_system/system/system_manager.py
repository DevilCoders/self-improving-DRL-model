"""Cross-platform system automation utilities for Linux and Windows."""
from __future__ import annotations

import os
import platform
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from ..config import SystemManagementConfig


@dataclass
class CommandResult:
    command: List[str]
    returncode: int
    stdout: str
    stderr: str
    elevated: bool


class SystemManager:
    """Lightweight automation orchestrator for Linux and Windows environments."""

    def __init__(self, config: SystemManagementConfig, log_directory: str = "system_logs") -> None:
        self.config = config
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.platform = platform.system().lower()

    # Platform helpers -----------------------------------------------------
    def detect_platform(self) -> str:
        return self.platform

    def has_admin_rights(self) -> bool:
        if self.platform == "windows":
            try:
                import ctypes  # local import to avoid platform issues

                return bool(ctypes.windll.shell32.IsUserAnAdmin())
            except Exception:
                return False
        return os.geteuid() == 0 if hasattr(os, "geteuid") else False

    # Command execution ----------------------------------------------------
    def _should_use_admin(self, elevated: bool) -> bool:
        if not elevated:
            return False
        if self.platform == "windows":
            return self.config.enable_windows_admin
        return self.config.enable_linux_admin

    def _should_use_standard(self) -> bool:
        if self.platform == "windows":
            return self.config.enable_windows_standard
        return self.config.enable_linux_standard

    def _prepare_command(self, command: Iterable[str], elevated: bool) -> List[str]:
        base_cmd = list(command)
        if not self._should_use_standard():
            raise PermissionError("Standard user execution disabled by configuration")
        if self._should_use_admin(elevated) and not self.config.dry_run:
            if self.platform == "windows":
                return [
                    "powershell",
                    "-NoProfile",
                    "Start-Process",
                    base_cmd[0],
                    "-ArgumentList",
                    shlex.join(base_cmd[1:]),
                    "-Verb",
                    "RunAs",
                    "-Wait",
                ]
            else:
                return ["sudo", "-n", *base_cmd]
        return base_cmd

    def run_command(self, command: Iterable[str], elevated: bool = False) -> CommandResult:
        prepared = self._prepare_command(command, elevated)
        if self.config.dry_run:
            return CommandResult(list(prepared), 0, "", "", elevated)
        process = subprocess.run(
            prepared,
            capture_output=True,
            text=True,
            check=False,
        )
        result = CommandResult(list(prepared), process.returncode, process.stdout, process.stderr, elevated)
        if self.config.audit_commands:
            self._log_result(result)
        return result

    def _log_result(self, result: CommandResult) -> None:
        path = self.log_directory / "automation.log"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"command={shlex.join(result.command)} | returncode={result.returncode} | elevated={result.elevated}\n"
            )
            if result.stdout:
                handle.write(f"stdout: {result.stdout}\n")
            if result.stderr:
                handle.write(f"stderr: {result.stderr}\n")

    # Automation helpers ---------------------------------------------------
    def create_directory(self, path: str, elevated: bool = False) -> CommandResult:
        if self.platform == "windows":
            command = ["powershell", "-NoProfile", "New-Item", "-ItemType", "Directory", "-Path", path]
        else:
            command = ["mkdir", "-p", path]
        return self.run_command(command, elevated=elevated)

    def manage_service(self, name: str, action: str, elevated: bool = True) -> CommandResult:
        action = action.lower()
        if self.platform == "windows":
            command = ["powershell", "-NoProfile", "Set-Service", name, f"-Status", action]
        else:
            command = ["systemctl", action, name]
        return self.run_command(command, elevated=elevated)

    def schedule_task(
        self,
        name: str,
        command: str,
        when: str,
        elevated: bool = False,
    ) -> CommandResult:
        if self.platform == "windows":
            powershell_cmd = (
                "powershell",
                "-NoProfile",
                "Register-ScheduledTask",
                f"-TaskName",
                name,
                "-Trigger",
                f"(New-ScheduledTaskTrigger -Once -At {when})",
                "-Action",
                f"(New-ScheduledTaskAction -Execute '{command}')",
                "-RunLevel",
                "Highest" if elevated else "Limited",
            )
            return self.run_command(powershell_cmd, elevated=elevated)
        cron_expression = when
        cron_entry = f"{cron_expression} {command}"
        cron_file = self.log_directory / f"{name}.cron"
        cron_file.write_text(cron_entry, encoding="utf-8")
        return CommandResult(["cron"], 0, f"Scheduled: {cron_entry}", "", elevated)


__all__ = ["SystemManager", "CommandResult"]
