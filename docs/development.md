# Development Guide

This guide outlines recommended development practices, testing strategies, and extension
points for the self-improving DRL project.

## Environment Setup

1. Install dependencies from `requirements.txt` (optionally extend with PyTorch or ROS
   packages suited to your hardware).
2. Build the ROS C++ example with CMake if you require bridge integration.
3. Configure system automation policies by editing `SystemConfig.system_management`.

## Running Tests

```bash
pytest
```

Some optional dependencies (e.g., `numpy`) may not be present in minimal environments.
Tests gracefully skip when a dependency is unavailable; install missing libraries to
unlock the full suite.

## Extending Agents

- Modify `ActorCritic` to introduce alternative activations or additional auxiliary heads.
- Adapt `PPOAgent.update` to incorporate bespoke loss terms, such as KL penalties or
  behaviour cloning regularisers.
- Register new agents through `drl_system.__init__` for ergonomic imports.

## Adding Training Modes

- Extend `Trainer.collect_rollout` for new modes (e.g., hierarchical control, imitation
  warm starts).
- Append the mode identifier to `SystemConfig.training.modes`.
- Document the mode in `docs/training_and_modes.md` for discoverability.

## Dataset Pipelines

- Implement richer dataset ingestion (e.g., ROS bag parsing) by writing adapters that
  emit NumPy arrays, then save them using the chunk schema described in
  `docs/data_management.md`.

## Automation and Deployment

- Use `SystemManager.schedule_task` for recurring maintenance (log rotation, backup
  scripts).
- Combine ROS integration with automation commands to orchestrate sensor bring-up,
  firmware updates, or experiment resets.

## Contribution Checklist

- [ ] Update relevant documentation in `docs/`.
- [ ] Add or adjust tests, especially for new data-handling or safety features.
- [ ] Run `pytest` (expect skips where optional deps are absent).
- [ ] Ensure the README highlights new capabilities.
