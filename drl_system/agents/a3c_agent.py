"""Asynchronous Advantage Actor-Critic agent built on the PPO backbone."""
from __future__ import annotations

from typing import Dict, Iterable

import torch

from ..config import AgentConfig, MemoryConfig, PPOConfig
from ..memory.replay_buffer import Transition
from .ppo_agent import PPOAgent, PPOBatch


class A3CAgent(PPOAgent):
    """Lightweight A3C variant that periodically synchronises with a global model."""

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
        self.sync_factor = float(agent_config.sync_factor)
        self.n_step = max(1, int(agent_config.n_step))
        self.global_model = type(self.model)(
            obs_dim,
            action_dim,
            hidden_sizes=tuple(agent_config.hidden_sizes),
            hierarchy_levels=agent_config.hierarchy_levels,
            transformer_layers=agent_config.transformer_layers,
        ).to(device)
        self.global_model.load_state_dict(self.model.state_dict())
        self._update_steps = 0
        self.gamma = memory_config.gamma

    def compute_advantages(
        self,
        transitions: Iterable[Transition],
        gamma: float,
        lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transitions = list(transitions)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)
        dones = torch.tensor([float(t.done) for t in transitions], dtype=torch.float32, device=self.device)
        values = []
        with torch.no_grad():
            for transition in transitions:
                obs = torch.from_numpy(transition.state).float().to(self.device)
                _, value, _, _, _ = self.model(obs)
                values.append(value.view(-1)[0])
        values.append(torch.zeros(1, device=self.device))
        advantages = torch.zeros(len(transitions), dtype=torch.float32, device=self.device)
        returns = torch.zeros(len(transitions), dtype=torch.float32, device=self.device)
        for idx in range(len(transitions)):
            n_step_return = torch.tensor(0.0, device=self.device)
            discount = 1.0
            for step in range(self.n_step):
                j = idx + step
                if j >= len(transitions):
                    break
                n_step_return += discount * rewards[j]
                discount *= gamma
                if dones[j] > 0:
                    break
            bootstrap_index = min(idx + self.n_step, len(values) - 1)
            n_step_return += discount * values[bootstrap_index]
            returns[idx] = n_step_return
            advantages[idx] = n_step_return - values[idx]
        return advantages.detach(), returns.detach()

    def update(self, batch: PPOBatch, **kwargs: Dict) -> Dict[str, float]:
        stats = super().update(batch, **kwargs)
        with torch.no_grad():
            for global_param, local_param in zip(self.global_model.parameters(), self.model.parameters()):
                global_param.data.mul_(self.sync_factor).add_((1 - self.sync_factor) * local_param.data)
                local_param.data.copy_(global_param.data)
        self._update_steps += 1
        stats.update(
            {
                "agent": "a3c",
                "sync_updates": self._update_steps,
            }
        )
        return stats


__all__ = ["A3CAgent"]
