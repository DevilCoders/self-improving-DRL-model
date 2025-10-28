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
  mixture-of-experts feature routing, cross-step memory, auxiliary
  advantage/uncertainty heads, generalized advantage estimation, and configurable
  hyper-parameters. The backbone now emits dynamics, meta-value, and behaviour
  prior diagnostics alongside twin-Q distributions.
- **Agent Portfolio** – Switch between PPO, A3C, SAC, DQN, DDPG, TD3, REINFORCE,
  and Quantile-DQN variants (or extend the factory) using `SystemConfig.agent`,
  enabling ensemble experimentation
  without rewriting training loops.
- **Replay and Episodic Memory** – Experience replay plus long-term episode storage
  for meta-learning workflows.
- **Human Feedback (RLHF)** – Queue-based feedback buffer with configurable reward
  aggregation strategies.
- **Safe Action Filtering** – Configurable clamps and forbidden zones to prevent
  unsafe actions.
- **Tokenization Stack** – Custom tokenizer, encoder, and decoder for handling
  textual observations or instructions.
- **Hierarchical World Model** – Transformer-enhanced actor-critic with
  predictive coding, hierarchical latent reasoning, reflection heads, and
  risk/quantile diagnostics that feed richer signals into any supported agent.
- **Multimodal Dataset Factory** – Offline dataset generation utilities with
  automatic chunking/overlap and curated corpora spanning terminal operations,
  ethical hacking playbooks, diffusion prompts, audio transcripts, PDF briefs,
  robotics control sketches, and multilingual codebases in CSV/TSV/TXT/JSON/JSONL,
  WAV, PNG, and PDF formats.
- **Evolutionary Skill Heads** – Hierarchical skill, world-model, and evolution
  estimators that regularise policy updates for continual self-improvement.
- **Risk-Aware Diagnostics** – Quantile distributions and risk logits enable
  distributional updates (Quantile-DQN) and policy-stability analysis across the
  entire agent suite.
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
  agents/               # PPO/A3C/SAC/DQN/DDPG/TD3/REINFORCE/Quantile agents and policy networks
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

from drl_system import SystemConfig, Trainer, SyntheticDatasetBuilder

config = SystemConfig()
def dummy_env_factory():
    return np.zeros(8, dtype="float32")

# Materialise multimodal datasets alongside standard offline rollouts
builder = SyntheticDatasetBuilder(config.dataset)
builder.generate(num_samples=256, obs_dim=8)

config.training.modes = ["offline", "parallel", "evaluation"]
config.training.parallel_workers = 4
config.agent.type = "sac"
config.agent.temperature = 0.7

trainer = Trainer(config, dummy_env_factory)
trainer.train(steps=2048)
```

## Multimodal Dataset Layout

Running `SyntheticDatasetBuilder.generate()` now produces structured datasets
under `data/<version>/multimodal/`:

| Dataset | Modality | Key Files |
| --- | --- | --- |
| `terminal_commands` | Text (Linux/Windows automation) | `dataset.csv`, `dataset.tsv`, `dataset.jsonl` |
| `ethical_hacking_commands` | Text (safe penetration testing) | `dataset.csv`, `dataset.txt` |
| `stable_diffusion_prompts` | Image prompts | `images/sample_*.png`, `prompts.json` |
| `audio_language_corpus` | Audio + transcripts | `audio/utterance_*.wav`, `transcripts.jsonl` |
| `technical_pdfs` | PDF briefs | `pdfs/briefing_*.pdf`, `summaries.json` |
| `code_corpus` | Multilingual code snippets | `python/analysis_agent.py`, `cpp/control_loop.cpp`, `javascript/dashboard.js` |
| `robotics_controls` | Control traces | `trajectories.csv`, `ros_topics.json` |

Each dataset ships with a `metadata.json` manifest capturing provenance, format,
and descriptive tags to streamline downstream loading.

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
- `Docs/usage.md` – Scenario-driven usage playbook.
- `Docs/training.md` – Detailed scheduling, RLHF, and curriculum notes.
- `Docs/datasets.md` – Dataset catalogue with chunking workflow.
- `Docs/technology.md` – Architecture and diagnostics reference.
- `Docs/` – Uppercase folder with printable quick-starts, agent deep dives, and
  hardware operations runbooks for teams that prefer capitalised documentation
  roots.
