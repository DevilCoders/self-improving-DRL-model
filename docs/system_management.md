# System Management and Automation

The `drl_system.system` package provides cross-platform automation tools to bootstrap
environments, schedule maintenance tasks, and manage services on Linux and Windows.

## Command Execution

```python
from drl_system import SystemConfig, SystemManager

config = SystemConfig()
config.system_management.dry_run = True
manager = SystemManager(config.system_management)

result = manager.run_command(["echo", "hello"], elevated=False)
print(result.stdout)
```

- **Standard vs Admin** – Configuration toggles (`enable_*`) control whether admin or
  standard user execution is allowed per platform.
- **Dry Run** – When `dry_run` is enabled the manager returns immediately without
  launching the command, which is helpful for testing automation plans.
- **Auditing** – Command invocations are appended to `system_logs/automation.log` when
  `audit_commands` is `True`.

## Service Management

```python
manager.manage_service("docker", action="restart", elevated=True)
```

On Linux this maps to `systemctl restart docker`. On Windows it uses PowerShell's
`Set-Service` cmdlet.

## Scheduling Tasks

- **Linux** – Writes a cron-style entry into `system_logs/<name>.cron` which can be
  imported into `crontab`.
- **Windows** – Uses `Register-ScheduledTask` with configurable run level.

Example:

```python
manager.schedule_task(
    name="nightly-backup",
    command="python backup.py",
    when="0 2 * * *",  # Cron expression for 2 AM daily
    elevated=True,
)
```

## Directory Provisioning

Ensure experiment directories exist on both platforms:

```python
manager.create_directory("C:/experiments/run01", elevated=False)
manager.create_directory("/opt/drl/runs", elevated=True)
```

## Admin Rights Detection

`SystemManager.has_admin_rights()` uses `os.geteuid()` on Linux and `IsUserAnAdmin` on
Windows to report whether the current process already has elevated privileges.

## Integration Tips

- Call `detect_platform()` during startup to branch into OS-specific deployment logic.
- Pair automation scripts with the ROS bridge to orchestrate sensor calibration or robot
  bring-up sequences end-to-end.
- Use the log files to audit automation steps performed on production agents.
