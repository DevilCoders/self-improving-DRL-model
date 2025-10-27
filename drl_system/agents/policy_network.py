"""Policy and value networks for PPO style agents."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        blocks = []
        last_dim = obs_dim
        for hidden in hidden_sizes:
            layer = nn.Linear(last_dim, hidden)
            components = [layer]
            if use_layer_norm:
                components.append(nn.LayerNorm(hidden))
            components.append(nn.GELU())
            if dropout > 0:
                components.append(nn.Dropout(dropout))
            block = nn.Sequential(*components)
            blocks.append(block)
            last_dim = hidden
        self.shared = nn.ModuleList(blocks)
        self.policy_head = nn.Linear(last_dim, action_dim)
        self.value_head = nn.Linear(last_dim, 1)
        self.advantage_head = nn.Linear(last_dim, action_dim)
        self.uncertainty_head = nn.Sequential(nn.Linear(last_dim, 1), nn.Softplus())

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = obs
        for block in self.shared:
            residual = features
            features = block(features)
            if residual.shape[-1] == features.shape[-1]:
                features = 0.5 * (features + residual)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        uncertainty = self.uncertainty_head(features)
        return policy_logits, value, advantage, uncertainty

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, value, advantage, uncertainty = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            value,
            {
                "advantage_logits": advantage.detach(),
                "uncertainty": uncertainty.detach(),
            },
        )


__all__ = ["ActorCritic"]
