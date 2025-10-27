# System Operations Playbook

This playbook complements `docs/system_management.md` by detailing
platform-specific automation flows, escalation procedures, and offline/online
switching strategies using the new agent portfolio.

## Linux

- **Admin Mode** – Enable privileged automation via `SystemConfig.system_management.enable_linux_admin`.
  - Use `sudo systemctl` recipes for deploying environment runners.
  - Pair with `config.agent.type = "a3c"` to exploit multi-core rollouts.
- **Standard Mode** – Disable admin flags to run inside restricted containers.
  - Cron-based dataset refresh: `crontab -e` → `0 */3 * * * python tools/offline_refresh.py`.

## Windows

- **Admin Mode** – Toggle `enable_windows_admin` and use PowerShell scripts under
  an elevated prompt (`Start-Process PowerShell -Verb RunAs`).
- **Standard Mode** – Schedule tasks with `schtasks /Create` using non-admin
  credentials. Prefer `config.agent.type = "ppo"` for deterministic evaluation.

## Offline vs Online

| Mode | Configuration | Notes |
| --- | --- | --- |
| Offline | `config.training.modes = ["offline"]` | Utilises dataset chunks and replay memory exclusively. |
| Online | Includes `"online"`, `"parallel"`, or `"distributed"` | Streams live rollouts; combine with SAC for exploratory phases. |
| Hybrid | `config.training.modes = ["offline", "parallel", "evaluation"]` | Seeds replay with synthetic data then fine-tunes across environments. |

## Automation Checklist

1. Provision dataset directories using `SyntheticDatasetBuilder.generate`.
2. Select agent type (PPO, A3C, SAC) and tune `SystemConfig.agent` knobs.
3. Run `Trainer.train` in offline mode to validate.
4. Activate system managers (Linux/Windows tasks) for production scheduling.
5. Monitor metrics exported by `self_improvement` checkpoints and
   `agent.update` dictionaries.
