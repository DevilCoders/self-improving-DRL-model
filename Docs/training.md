# Training Strategies

This document expands upon the trainer orchestration options and shows how the new
agent implementations plug into offline, online, curriculum, and evaluation loops.

## Scheduler Overview

The `Trainer` coordinates multiple execution phases:

1. **Rollout collection** – via `collect_rollout` with modes `online`, `offline`,
   `parallel`, and `distributed`. The async runner fans out to multiple environments
   while capturing diagnostics like hierarchy traces and meta-value predictions.
2. **Replay management** – experiences are written to the unified replay buffer and the
   episodic memory store. Offline chunk files under `data/generated/<version>/chunks/`
   are streamed on boot to seed DQN/DDPG/TD3 updates.
3. **Update step** – `update_agent` now routes metrics for PPO-style actors, twin-critic
   losses, behaviour priors, dynamics consistency, and human feedback aggregation.

## Agent-specific Notes

- **PPO / A3C**: continue to rely on Generalised Advantage Estimation. Extra heads for
  dynamics, meta-values, and behaviour priors tighten convergence when mixing offline
  replays with live samples.
- **SAC**: unchanged interface but benefits from the richer diagnostics to adapt the
  entropy temperature via RLHF signals.
- **DQN**: uses double Q-learning by default with twin-head critics. Soft updates are
  configurable through `agent.soft_update_tau`, while quantile diagnostics feed into
  safety dashboards.
- **Quantile-DQN**: minimises quantile-Huber loss using the same backbone, exposing
  risk-aware metrics for safety-critical automation.
- **REINFORCE**: provides a lightweight Monte-Carlo policy gradient that reuses the
  actor-critic baseline when `agent.reinforce_baseline` is enabled.
- **DDPG**: inherits DQN's critic training and adds deterministic policy refinement
  (argmax over logits). The noise schedule decays automatically and can be overridden
  via `agent.ddpg_noise`.
- **TD3**: delays actor updates to every `policy_delay` steps and reports critic gaps to
  surface divergence between twin heads. Pair with `training.parallel_workers > 1` to
  keep replay diversity high.

## Curriculum & Evaluation

Define curriculum stages via `training.curriculum_stages`. At each stage you can tweak
the agent type—e.g., start with PPO for exploration, switch to DQN for discrete safety,
and finish with TD3 for deterministic fine-tuning. Evaluation runs should set
`training.modes=['evaluation']`, disable replay writes, and track meta-value alignment.

## Human Feedback Integration

`RLHFConfig` governs reward aggregation. The trainer feeds the aggregated reward back
into recent transitions, which is especially powerful for the deterministic DDPG/TD3
agents because it nudges the deterministic head without destabilising the critic.

## Exporting Artifacts

- Snapshots: call `torch.save(agent.model.state_dict(), path)` at `save_interval`.
- Dataset manifests: `SyntheticDatasetBuilder.export_manifest()` produces JSON metadata
  for each dataset folder, useful for audit trails.
- Diagnostics: the trainer returns a metrics dictionary—persist it alongside checkpoints
  to capture `meta_value_alignment`, `behaviour_prior_alignment`, and the new
  twin-critic statistics.
