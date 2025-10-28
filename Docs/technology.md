# Technology Reference

This document summarises the architectural primitives that power the self-improving
reinforcement learning stack.

## Core Components

- **Actor-Critic Backbone** – transformer-enhanced shared encoder with mixture-of-experts
  blocks, episodic memory, predictive coding, dynamics, twin/tensorized Q, quantile,
  risk, and behaviour prior heads. Diagnostics expose `dynamics`, `meta_value`,
  `behaviour_prior`, `twin_q_values`, `quantiles`, and `risk` tensors.
- **Adaptive Optimiser** – wraps `torch.optim.Adam` and performs lightweight
  meta-learning by decaying the learning rate based on update counts.
- **Replay Memory** – standard and prioritized buffers with episodic long-term storage.
- **System Manager** – cross-platform automation harness with Linux/Windows admin and
  standard-role execution, dry-run safety, and audit logging.
- **Integration Layer** – ROS bridge (`cpp/ros_bridge.cpp`), hardware adapters, and
  dataset manifests enabling robotics and edge deployment.

## Agent Portfolio

- **PPO**: baseline policy-gradient agent enriched with dynamics/meta-value losses.
- **A3C**: asynchronous updates with global weight synchronisation and N-step targets.
- **SAC**: entropy regularised continuous control with safe-action filtering.
- **DQN**: distributional twin-Q updates with optional double Q-learning.
- **Quantile-DQN**: quantile-regression variant that outputs risk-aware metrics.
- **REINFORCE**: Monte-Carlo policy gradient leveraging the shared baseline heads.
- **DDPG**: deterministic actor updates and decaying exploration noise for robotics.
- **TD3**: twin critics with delayed policy refresh to mitigate overestimation.

## Diagnostics Stream

During training, `Trainer.update_agent` emits metrics that blend policy-gradient,
critic, and self-improvement signals:

- `actor_loss`, `critic_loss`, `entropy`
- `q_loss`, `critic_gap`, `twin_consensus`, `quantile_loss`
- `meta_value_alignment`, `behaviour_prior_alignment`, `dynamics_consistency`
- `uncertainty`, `noise_level`, `exploration_noise`, `risk_regulariser`

## Safety Layer

- **SafeActionsFilter** clamps actions to configured ranges and rejects forbidden zones.
- **RLHF Integration** merges human feedback buffers using `aggregation` strategies.
- **Self-improvement Loop** analyses metrics and adjusts optimiser parameters via the
  adaptive optimiser hook.

## Tooling Highlights

- **Tokeniser Stack**: custom encoder/decoder/tokeniser pipeline with vocabulary export.
- **Preprocessing**: modular pipelines for text, audio, image, and document ingestion.
- **Testing**: pytest suite covering agent factory instantiation, dataset building,
  safe-action filtering, and tokenisation.

See `Docs/usage.md` and `Docs/training.md` for operational workflows.
