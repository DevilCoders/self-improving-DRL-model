# Training Modes and Execution Paths

This guide explains each training mode, curriculum behaviour, and how to orchestrate
parallel or distributed rollouts using the trainer utilities.

## Configuring Modes

```python
import numpy as np
from drl_system import SystemConfig, Trainer

config = SystemConfig()
config.training.modes = ["offline", "parallel", "evaluation"]
config.training.parallel_workers = 6
config.agent.type = "a3c"  # Switch to asynchronous updates for heavier parallelism

trainer = Trainer(config, env_factory=lambda: np.random.randn(8).astype("float32"))
trainer.train()
```

### Agent-Aware Scheduling

Different agents shine in different modes:

- **PPO** – Default for balanced offline/online curricula.
- **A3C** – Pair with `parallel` or `distributed` modes to leverage asynchronous
  rollouts while keeping the trainer API unchanged.
- **SAC** – Combine with `online` exploration windows to encourage higher
  entropy policies during discovery phases.
- **DQN** – Excellent for discrete automation in `offline` or hybrid schedules where
  dataset chunks bootstrap the replay buffer before online fine-tuning.
- **DDPG** – Works best with curriculum stages that end in deterministic policies;
  pair with ROS integrations for robotics control.
- **TD3** – Select when delayed, conservative updates are required; suits
  high-precision evaluation phases following broader exploration.

Available modes:

| Mode | Description |
| --- | --- |
| `offline` | Consumes pre-generated dataset chunks and continues fine-tuning online. |
| `online` | Default live environment interaction. |
| `parallel` | Uses batched rollouts with multiple workers on a single node. |
| `distributed` | Mirrors `parallel` execution while tagging transitions for downstream distributed pipelines. |
| `curriculum` | Gradually increases rollout length per stage for curriculum learning. |
| `evaluation` | Runs lightweight evaluation sweeps and records the metrics in the self-improvement history. |

## Parallel & Distributed Execution

- Rollouts are computed in batches using `parallel_workers`. Each worker maintains its
  own observation stream, and safety filtering is applied independently per worker.
- Setting `distributed` in `config.training.modes` enables rollout tagging that can be
  consumed by downstream distributed learners or loggers.
- The trainer automatically adapts to CPU/GPU placement based on `DeviceManager` and
  can interleave modes (e.g., `["parallel", "evaluation"]`).

## Curriculum Learning

When `curriculum` is included:

1. The trainer tracks a `curriculum_stage` counter.
2. Every `update_interval * curriculum_stages` steps the stage increments (bounded by
   `curriculum_stages - 1`).
3. Rollout length grows by `128 * stage`, encouraging longer horizons over time.

## Evaluation Workflow

Calling `Trainer.evaluate()` explicitly or including `evaluation` mode produces a
dictionary with `mean_reward`, `max_reward`, and `num_transitions`. The
`SelfImprovementLoop` caches this data in checkpoints for later inspection.

## Safe Actions and RLHF Integration

- The `SafeActionsFilter` clamps actions and removes unsafe commands before they are sent
  to environments, maintaining safety even in distributed runs.
- RLHF feedback gathered via `FeedbackBuffer.collect()` is aggregated and blended into
  the latest transitions to accelerate alignment with human preferences.

## Extending Modes

To add a custom mode:

1. Append the new mode label to `config.training.modes`.
2. Extend `Trainer.collect_rollout` to branch on the new label.
3. Optionally customise the curriculum logic or evaluation path.

The modular structure is designed so that bespoke research modes can be introduced
without rewriting the training core.
