# Self-Improving DRL Model

This repository provides a modular deep reinforcement learning (DRL) research sandbox
that can self-improve over time. The project combines Python and C++ components,
policy gradient algorithms, RLHF utilities, safe action filtering, parallel/distributed
training modes, and hooks for robotics integrations as well as cross-platform system
automation.

## Features

- **Self-Improvement Loop** – Adaptive optimizer wrapper and metric tracker that
  periodically checkpoints learning state.
- **Policy Gradient Agents** – PPO-style actor-critic implementation with
  auxiliary advantage/uncertainty heads, generalized advantage estimation, and
  configurable hyper-parameters.
- **Replay and Episodic Memory** – Experience replay plus long-term episode storage
  for meta-learning workflows.
- **Human Feedback (RLHF)** – Queue-based feedback buffer with configurable reward
  aggregation strategies.
- **Safe Action Filtering** – Configurable clamps and forbidden zones to prevent
  unsafe actions.
- **Tokenization Stack** – Custom tokenizer, encoder, and decoder for handling
  textual observations or instructions.
- **Synthetic Dataset Builder** – Offline dataset generation utilities with automatic
  chunking/overlap for CPU and GPU workflows.
- **Training Modes** – Offline, online, parallel, distributed, curriculum, and
  evaluation loops configurable via `SystemConfig.training.modes`.
- **Multi-Environment Runner** – Async environment manager for distributed or
  multi-threaded rollouts.
- **System Automation** – Linux/Windows automation helpers for admin/non-admin
  workflows, service management, and scheduling.
- **ROS & Hardware Hooks** – ROS 2 adapter and Raspberry Pi / Arduino controller
  abstractions for real-world deployment.
- **C++ Integration** – Minimal ROS bridge example (see `cpp/` directory).

## Project Layout

```
drl_system/
  agents/               # PPO agent and policy networks
  config.py             # Centralized dataclass configuration
  data/                 # Synthetic dataset builder
  environments/         # Multi-environment manager
  integration/          # ROS / hardware adapters
  memory/               # Replay buffer and episodic memory
  optimization/         # Self-adaptive optimizer
  preprocess/           # Observation preprocessing pipeline
  runners/              # Async runner utilities
  safety/               # Safe action filtering
  self_improvement/     # Continual learning utilities
  tokenization/         # Tokenizer, encoder, decoder modules
  system/               # Cross-platform system management helpers
  training/             # RLHF integration and training orchestrator
```

## Python Usage

```python
import numpy as np

from drl_system import SystemConfig, Trainer

config = SystemConfig()
def dummy_env_factory():
    return np.zeros(8, dtype="float32")

config.training.modes = ["offline", "parallel", "evaluation"]
config.training.parallel_workers = 4

trainer = Trainer(config, dummy_env_factory)
trainer.train(steps=2048)
```

## Building the C++ Example

The `cpp/` directory contains a ROS bridge skeleton that can be extended for
hardware integration.

```
mkdir build && cd build
cmake ..
make
```

## Tests

Install dependencies and run the test suite:

```
pip install -r requirements.txt
pytest
```

## Documentation

Detailed guides are available in the `docs/` directory:

- `docs/overview.md` – Architectural overview.
- `docs/training_and_modes.md` – Training modes, curriculum logic, and evaluation flows.
- `docs/data_management.md` – Dataset generation, chunking, and offline ingestion.
- `docs/system_management.md` – Linux/Windows automation helpers.
- `docs/development.md` – Contribution and extension guidelines.
