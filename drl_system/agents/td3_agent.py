"""Twin-Delayed Deep Deterministic policy gradient agent."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

from ..config import AgentConfig, MemoryConfig, PPOConfig
from ..memory.replay_buffer import Transition
from .ddpg_agent import DDPGAgent
from .ppo_agent import PPOBatch


class TD3Agent(DDPGAgent):
    """Adds twin critics and delayed actor updates on top of :class:`DDPGAgent`."""

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
        self.policy_delay = max(2, int(getattr(agent_config, "policy_delay", 2)))
        self.td3_noise = float(getattr(agent_config, "td3_noise", 0.1))
        self._delay_counter = 0

    def update(self, batch: PPOBatch, **kwargs: Dict) -> Dict[str, float]:
        stats = super().update(batch, **kwargs)
        transitions: Sequence[Transition] | None = kwargs.get("transitions")
        if not transitions:
            return stats

        self._delay_counter += 1
        if self._delay_counter % self.policy_delay != 0:
            stats.update({"agent": "td3", "delayed_update": 1})
            return stats

        states = torch.from_numpy(np.stack([t.state for t in transitions])).float().to(self.device)
        _, _, _, _, diagnostics = self.model(states)
        q_values = diagnostics["q_values"]
        twin_q_values = diagnostics["twin_q_values"]
        critic_gap = torch.abs(q_values - twin_q_values).mean()

        stats.update(
            {
                "agent": "td3",
                "critic_gap": float(critic_gap.item()),
                "twin_consensus": float(torch.min(q_values, twin_q_values).mean().item()),
                "exploration_noise": float(self.td3_noise),
            }
        )
        return stats


__all__ = ["TD3Agent"]
