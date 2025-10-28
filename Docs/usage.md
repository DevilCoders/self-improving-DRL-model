# Usage Playbook

This guide provides concrete recipes for operating the self-improving DRL stack across
offline, online, distributed, and hybrid workflows. Each scenario highlights the new
agent portfolio—PPO, A3C, SAC, DQN, DDPG, TD3, IMPALA, and TRPO—and the extended
feature heads that ship with the upgraded actor-critic backbone.

## Quick-start Matrix

| Objective | Recommended Agent | Key Flags | Notes |
| --- | --- | --- | --- |
| Fast policy iteration | `ppo` | `training.modes=['parallel']` | Uses transformer-enhanced PPO loops. |
| Exploration-heavy search | `sac` | `agent.temperature=1.2` | Soft actor-critic with adaptive entropy. |
| Discrete safety automation | `dqn` | `agent.double_q=True` | Relies on twin-Q diagnostics for conservative control. |
| Deterministic robotics | `ddpg` | `agent.ddpg_noise=0.15` | Twin-head critic plus deterministic policy updates. |
| Delayed low-noise tuning | `td3` | `agent.policy_delay=3` | Uses twin critics and delayed actor updates. |
| Low-resource async runs | `a3c` | `training.num_envs=16` | Multi-worker A3C with global weight synchronisation. |
| Massive distributed rollouts | `impala` | `agent.impala_clip_rho=1.0` | V-trace importance weighting stabilises replay-lagged data. |
| Safety-first evaluation sweeps | `trpo` | `agent.trpo_kl_target=0.01` | Trust-region penalties restrict abrupt behavioural shifts. |

## Running Experiments

1. **Configure the system** – edit `configs/base.yaml` (or `SystemConfig` in code) to
   select the agent type, dataset suite, and training mode. New agent knobs include
   `double_q`, `ddpg_noise`, `td3_noise`, `policy_delay`, `impala_clip_rho`, and
   `trpo_kl_target` for fine-grained control.
2. **Materialise datasets** – invoke `SyntheticDatasetBuilder.build_all()` to generate
   multimodal datasets in their dedicated folders (`terminal_commands/`,
   `stable_diffusion/`, `audio_language/`, `technical_pdfs/`, `code_corpus/`, etc.).
3. **Launch training** – use `Trainer(SystemConfig(), env_factory)` and call
   `collect_rollout` / `update_agent` loops. The trainer automatically surfaces
   diagnostics for meta-value alignment, behaviour priors, and the new Q-heads.
4. **Monitor metrics** – metrics returned from `update_agent` include PPO losses plus
   `q_loss`, `critic_gap`, `twin_consensus`, `meta_value_alignment`,
   `behaviour_prior_alignment`, IMPALA importance weight means, TRPO KL divergence,
   and latent drift scores. Feed these into your preferred logging backend.
5. **Deploy** – leverage `SystemManager` for cross-platform automation (Linux/Windows,
   admin or standard contexts). The updated docs in `Docs/system_operations.md` describe
   privilege escalation safeguards and dry-run support.

## Offline vs Online

- **Offline**: Mount pre-generated chunked datasets under `data/generated/v1/chunks/` to
  bootstrap the replay buffer. The trainer will mix offline experiences with live
  rollouts to stabilise updates for all agents, especially the distributional DQN and
  twin-critic TD3 variants.
- **Online**: Set `training.modes` to include `online` and optionally `distributed`
  to activate the async runner. Actor diagnostics expose `dynamics` predictions and
  `behaviour_prior` drifts so that the safe-action filter can be tuned on the fly.

## Hardware Notes

- **ROS / Robotics**: Pair DDPG/TD3 agents with the ROS bridge (`cpp/ros_bridge.cpp`) to
  stream sensor vectors into the shared actor-critic. The deterministic heads and
  behaviour priors stabilise real-world actuation.
- **Edge Devices**: For Raspberry Pi and Arduino deployments, favour DQN or PPO with
  reduced hidden sizes (`agent.hidden_sizes=[128,128]`). Chunked datasets and
  on-device replay buffers lower memory pressure.

Refer back to `Docs/models.md` for architectural diagrams and to
`Docs/training.md` for schedule orchestration tips.
