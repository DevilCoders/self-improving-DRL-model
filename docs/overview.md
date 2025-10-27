# System Overview

This document provides a high-level description of the self-improving DRL framework, the
core architectural principles, and how the Python and C++ components cooperate.

## Architecture Highlights

- **Adaptive Agent Stack** – A switchable PPO/A3C/SAC/DQN/DDPG/TD3 portfolio enhanced with
  auxiliary advantage, uncertainty, twin-Q, and predictive-coding heads allows the
  policy to quantify its confidence and learn richer representations for
  meta-learning loops.
- **Self-Improvement Loop** – The `SelfImprovementLoop` records rolling metrics, evaluation
  scores, and checkpoints to bootstrap continual learning without manual supervision.
- **Memory Systems** – Replay buffers and episodic stores are designed to mix online,
  offline, and human-feedback experiences with curriculum support.
- **Training Orchestration** – The trainer coordinates curriculum stages, multi-environment
  execution, RLHF integration, and safety filtering across online, offline, parallel,
  distributed, and evaluation modes.
- **Extensible Tokenization Pipeline** – Custom tokenizer, encoder, and decoder modules
  support embedding textual feedback or symbolic observations alongside raw tensors.
- **Hardware & ROS Hooks** – Python interfaces and the provided C++ ROS bridge offer clean
  integration points for robotics or embedded deployments.

## Module Map

| Module | Responsibility |
| --- | --- |
| `drl_system.agents` | Actor-critic networks, PPO/A3C/SAC/DQN/DDPG/TD3 agents, and the agent factory. |
| `drl_system.training` | Multi-mode trainer, RLHF feedback buffers, and evaluation loops. |
| `drl_system.memory` | Replay buffer, episodic memory, and transition schemas. |
| `drl_system.data` | Synthetic dataset generation, chunking, and iterators. |
| `drl_system.tokenization` | Tokenizer, encoder, and decoder scaffolding. |
| `drl_system.integration` | ROS, hardware adapters, and asynchronous runners. |
| `drl_system.system` | Cross-platform automation helpers for Linux and Windows. |
| `cpp/` | ROS bridge example demonstrating mixed Python/C++ integration. |

## Workflow at a Glance

1. **Configure** – Assemble a `SystemConfig`, enabling the desired training modes,
   system management policies, and dataset parameters.
2. **Prepare Data** – Generate or ingest datasets with `SyntheticDatasetBuilder`; chunks
   are automatically produced for scalable offline ingestion.
3. **Train** – Instantiate `Trainer(config, env_factory)` and call `train()`; the trainer
   will iterate through configured modes (offline, online, parallel, distributed,
   curriculum, evaluation) applying curriculum adjustments and safety filtering.
4. **Improve** – RLHF feedback and self-improvement checkpoints refine the optimizer and
   record evaluation metrics for future restarts.
5. **Deploy** – Use `SystemManager` to automate environment setup tasks and the ROS bridge
   to interface with hardware targets.

Refer to the additional documents in this directory for details on specific subsystems.
