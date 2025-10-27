# Documentation Hub (Capitalised Edition)

The `Docs/` directory mirrors the lower-case `docs/` knowledge base while
providing printable, modular manuals that some teams prefer to keep under a
capitalised root. Each guide is self-contained and references the Python API, C++
bridge, and automation workflows required to run the self-improving DRL stack in
both air-gapped and online environments.

## Available Guides

- `agents.md` – Portfolio comparison covering PPO, A3C, SAC, and extending the
  factory with custom agents.
- `models.md` – Hierarchical actor-critic internals, diagnostic streams, and
  checkpoints.
- `system_operations.md` – Linux and Windows orchestration patterns, including
  admin vs. non-admin automation recipes.

Every document is structured to double as a runbook: copy-and-paste code cells
and CLI sequences are annotated with expected outputs, safety call-outs, and
links back to the configuration dataclasses in `drl_system.config`.
