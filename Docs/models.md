# Hierarchical Model Internals

The unified `ActorCritic` network now layers contextual adapters, hierarchical
transformer blocks, predictive coding heads, and reflection vectors on top of the
original mixture-of-experts pathway. This document walks through the forward
pass, highlights diagnostic tensors, and explains how agents exploit the richer
representations.

## Architectural Highlights

1. **Context Adapter** – Projects raw observations into the shared latent space
   before any residual blocks, acting as a learnable skip connection.
2. **Hierarchical Stack** – A configurable number of GELU + LayerNorm modules
   build progressively abstract representations.
3. **Transformer Encoder** – Multi-head attention fuses the hierarchy into a
   context summary vector that supplements the residual backbone.
4. **Predictive Coding Head** – Produces latent deltas used for world-model error
   estimation and curiosity metrics.
5. **Reflection Head** – Generates self-assessment vectors for meta-learning and
   safe exploration heuristics.

These components feed the policy, value, advantage, and evolution heads, so all
agents—PPO, A3C, SAC, or custom—benefit without additional wiring.

## Diagnostics

Call `ActorCritic.forward` (or `PPOAgent.model`) to receive the following extra
signals in the diagnostics map:

- `predictive_code`: Latent prediction errors.
- `hierarchy_context`: Aggregated transformer output.
- `hierarchy_trace`: Full tensor of per-level representations (levels × dim).
- `reflection`: Self-assessment vector used by self-improvement routines.

```python
import torch
from drl_system.agents.policy_network import ActorCritic

model = ActorCritic(obs_dim=8, action_dim=4, hierarchy_levels=3, transformer_layers=2)
obs = torch.zeros(8)
logits, value, advantages, uncertainty, diagnostics = model(obs)
print(diagnostics.keys())
```

## Checkpointing Tips

- Include `model._hierarchy_trace` snapshots when building dashboards; it is
  detached and ready for logging.
- When exporting to ONNX, freeze the transformer layers by scripting the module
  before tracing.
- For embedded deployments (Raspberry Pi / Arduino gateways), reduce
  `hierarchy_levels` to 1 and `transformer_layers` to 0 to maintain latency
  budgets.
