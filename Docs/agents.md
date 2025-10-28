# Agent Portfolio Deep Dive

This guide documents the built-in agents available through
`drl_system.agents.factory.create_agent` and how to extend the registry with new
strategies. All agents conform to the `AgentProtocol`, allowing the `Trainer`
class to orchestrate rollouts, batching, and updates without branching logic.

## Configuring Agents

Set `SystemConfig.agent.type` to select an agent:

| Type | Description | Suggested Use |
| --- | --- | --- |
| `ppo` | Baseline proximal policy optimisation with adaptive auxiliary heads. | Stable policy-gradient experiments, curriculum learning. |
| `a3c` | Asynchronous Advantage Actor-Critic with configurable n-step returns and global weight synchronisation. | Multi-threaded rollouts, low-latency robotics control. |
| `sac` | Discrete-friendly Soft Actor-Critic variant with twin critics and temperature scaling. | High-entropy exploration, hybrid continuous/discrete tasks. |
| `dqn` | Distributional DQN with twin critics, double-Q selection, and soft target updates. | Discrete automation, safety-critical command execution. |
| `ddpg` | Deterministic policy gradients layered on the shared backbone with decaying exploration noise. | Robotic manipulation, ROS-integrated control loops. |
| `td3` | Twin-delayed deterministic policy gradients with critic-gap monitoring. | High-precision control requiring conservative updates. |
| `reinforce` | Classic REINFORCE policy gradients with entropy bonuses and behaviour alignment. | Educational baselines, debugging new environments. |
| `impala` | Importance-weighted actor-critic with V-trace corrections for off-policy distributed rollouts. | Fleet-scale simulation, asynchronous data collection. |
| `rainbow` | Categorical distributional DQN with twin support heads and adaptive projection. | Safety-critical discrete control demanding calibrated uncertainty. |

Key knobs live inside `SystemConfig.agent`:

- `hidden_sizes`: Feed-forward backbone width.
- `hierarchy_levels` / `transformer_layers`: Depth of the hierarchical
  representation stack baked into the shared actor-critic.
- `n_step`: Number of steps used by A3C style advantage calculations.
- `temperature`: Softmax temperature applied to all policies.
- `sac_alpha` / `soft_update_tau`: Temperature and target blend for SAC and off-policy agents.
- `double_q`: Toggle double Q-learning for DQN/DDPG/TD3/Rainbow.
- `ddpg_noise` / `td3_noise`: Exploration noise magnitudes for deterministic agents.
- `policy_delay`: Update cadence for TD3 actors.
- `reinforce_baseline_momentum`: Moving-average smoothing for the REINFORCE value baseline.
- `vtrace_rho_bar` / `vtrace_c_bar`: Importance-ratio clipping thresholds for IMPALA.
- `rainbow_atoms` / `rainbow_v_min` / `rainbow_v_max`: Distributional support configuration for Rainbow.

```python
import numpy as np
from drl_system import SystemConfig, Trainer

config = SystemConfig()
config.agent.type = "a3c"
config.agent.n_step = 8
config.agent.sync_factor = 0.95
trainer = Trainer(config, env_factory=lambda: np.zeros(8, dtype="float32"))
```

## Extending the Factory

Register custom agents by wrapping `create_agent` or adding new modules within
`drl_system/agents/`. Each agent should expose:

1. `compute_advantages(transitions, gamma, lam)` – returns tensors.
2. `prepare_batch(transitions, advantages, returns)` – produces a `PPOBatch` or
   custom batch.
3. `update(batch, **kwargs)` – executes gradient steps and returns metrics.

The shared `ActorCritic` backbone exposes diagnostics such as predictive codes,
hierarchy traces, and reflection vectors. Custom agents may reuse these signals
for reward shaping, safety checks, or debugging dashboards.
