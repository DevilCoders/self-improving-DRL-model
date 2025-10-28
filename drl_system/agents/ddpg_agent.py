"""DDPG-style agent that augments the shared backbone with deterministic policy updates."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn

from ..config import AgentConfig, MemoryConfig, PPOConfig
from ..memory.replay_buffer import Transition
from .dqn_agent import DQNAgent
from .ppo_agent import PPOBatch


class DDPGAgent(DQNAgent):
    """Extends :class:`DQNAgent` with deterministic actor optimisation and target tracking."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        memory_config: MemoryConfig,
        ppo_config: PPOConfig,
        device: torch.device,
        agent_config: AgentConfig,
    ) -> None:
        super().__init__(obs_dim, action_dim, memory_config, ppo_config, device, agent_config)
        self.noise_scale = float(getattr(agent_config, "ddpg_noise", 0.2))
        self.tau = float(agent_config.soft_update_tau)
        self._update_counter = 0

    def update(self, batch: PPOBatch, **kwargs: Dict) -> Dict[str, float]:
        stats = super().update(batch, **kwargs)
        transitions: Sequence[Transition] | None = kwargs.get("transitions")
        if not transitions:
            return stats

        states = torch.from_numpy(np.stack([t.state for t in transitions])).float().to(self.device)
        logits, _, _, _, diagnostics = self.model(states)
        probs = torch.softmax(logits, dim=-1)
        deterministic_actions = torch.argmax(probs, dim=-1, keepdim=True)
        q_values = diagnostics["q_values"].gather(1, deterministic_actions)

        policy_loss = -q_values.mean()
        self.optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo_config.max_grad_norm)
        self.optimizer.step()

        self._update_counter += 1
        noise_level = self.noise_scale * (0.99 ** self._update_counter)

        stats.update(
            {
                "agent": "ddpg",
                "policy_loss": float(policy_loss.item()),
                "noise_level": float(noise_level),
            }
        )
        return stats


__all__ = ["DDPGAgent"]
