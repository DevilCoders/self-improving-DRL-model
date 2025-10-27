# Self-Improving DRL Model

This repository provides a modular deep reinforcement learning (DRL) research sandbox
that can self-improve over time. The project combines Python and C++ components,
policy gradient algorithms, RLHF utilities, safe action filtering, and hooks for
robotics integrations.

## Features

- **Self-Improvement Loop** – Adaptive optimizer wrapper and metric tracker that
  periodically checkpoints learning state.
- **Policy Gradient Agents** – PPO-style actor-critic implementation with
  generalized advantage estimation and configurable hyper-parameters.
- **Replay and Episodic Memory** – Experience replay plus long-term episode storage
  for meta-learning workflows.
- **Human Feedback (RLHF)** – Queue-based feedback buffer with configurable reward
  aggregation strategies.
- **Safe Action Filtering** – Configurable clamps and forbidden zones to prevent
  unsafe actions.
- **Tokenization Stack** – Custom tokenizer, encoder, and decoder for handling
  textual observations or instructions.
- **Synthetic Dataset Builder** – Offline dataset generation utilities that run on
  CPU or GPU.
- **Multi-Environment Runner** – Async environment manager for distributed or
  multi-threaded rollouts.
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
  training/             # RLHF integration and training orchestrator
```

## Python Usage

```python
import numpy as np

from drl_system import SystemConfig, Trainer

config = SystemConfig()
def dummy_env_factory():
    return np.zeros(8, dtype="float32")

trainer = Trainer(config, dummy_env_factory)
trainer.train(steps=1024)
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
