"""Discrete-friendly Soft Actor-Critic variant built on the PPO base."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

from ..config import AgentConfig, MemoryConfig, PPOConfig
from ..memory.replay_buffer import Transition
from .ppo_agent import PPOAgent, PPOBatch


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        x = torch.cat([obs, actions.float()], dim=-1)
        return self.net(x)


class SACAgent(PPOAgent):
    """Soft Actor-Critic agent that reuses the ActorCritic policy for a unified API."""

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
        hidden_dim = self.agent_config.hidden_sizes[-1]
        critic_input = obs_dim + 1
        self.critic1 = Critic(critic_input, hidden_dim).to(device)
        self.critic2 = Critic(critic_input, hidden_dim).to(device)
        self.target_critic1 = Critic(critic_input, hidden_dim).to(device)
        self.target_critic2 = Critic(critic_input, hidden_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=3e-4
        )
        self.alpha = float(agent_config.sac_alpha)
        self.tau = float(agent_config.soft_update_tau)
        self.gamma = memory_config.gamma

    def compute_advantages(
        self,
        transitions: Iterable[Transition],
        gamma: float,
        lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Use the default PPO advantages for compatibility; SAC update ignores them directly.
        return super().compute_advantages(transitions, gamma, lam)

    def update(
        self,
        batch: PPOBatch,
        transitions: Iterable[Transition] | None = None,
        **_: Dict,
    ) -> Dict[str, float]:
        if transitions is None:
            raise ValueError("SACAgent requires raw transitions for updates")
        observations = batch.observations.to(self.device)
        actions = batch.actions.to(self.device).float()
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs = torch.from_numpy(np.stack([t.next_state for t in transitions])).float().to(self.device)
        dones = torch.tensor([float(t.done) for t in transitions], dtype=torch.float32, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            next_logits, _, _, _, _ = self.model(next_obs)
            next_dist = torch.distributions.Categorical(logits=next_logits / self.temperature)
            next_actions = next_dist.sample().float().unsqueeze(-1)
            next_log_probs = next_dist.log_prob(next_actions.squeeze(-1)).unsqueeze(-1)
            target_q1 = self.target_critic1(next_obs, next_actions)
            target_q2 = self.target_critic2(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_value = rewards + (1 - dones) * self.gamma * target_q

        current_q1 = self.critic1(observations, actions)
        current_q2 = self.critic2(observations, actions)
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        logits, _, _, _, _ = self.model(observations)
        dist = torch.distributions.Categorical(logits=logits / self.temperature)
        sampled_actions = dist.sample().float().unsqueeze(-1)
        log_probs = dist.log_prob(sampled_actions.squeeze(-1)).unsqueeze(-1)
        q_new_actions = torch.min(self.critic1(observations, sampled_actions), self.critic2(observations, sampled_actions))
        actor_loss = (self.alpha * log_probs - q_new_actions).mean()
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for target, source in (
                (self.target_critic1, self.critic1),
                (self.target_critic2, self.critic2),
            ):
                for target_param, param in zip(target.parameters(), source.parameters()):
                    target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)

        entropy = -log_probs.mean()
        return {
            "agent": "sac",
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "policy_entropy": float(entropy.item()),
        }


__all__ = ["SACAgent"]
