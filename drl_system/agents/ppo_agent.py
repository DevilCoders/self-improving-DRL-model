"""Implementation of a PPO style agent with meta-learning hooks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
from torch import nn, optim

from ..config import MemoryConfig, PPOConfig
from ..memory.replay_buffer import Transition
from ..optimization.adaptive_optimizer import AdaptiveOptimizer
from .policy_network import ActorCritic


@dataclass
class PPOBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        memory_config: MemoryConfig,
        ppo_config: PPOConfig,
        device: torch.device,
    ) -> None:
        self.device = device
        self.model = ActorCritic(obs_dim, action_dim).to(device)
        base_optimizer = optim.Adam(self.model.parameters(), lr=3e-4, eps=1e-5)
        self.optimizer = AdaptiveOptimizer(base_optimizer)
        self.ppo_config = ppo_config

    def compute_advantages(
        self,
        transitions: Iterable[Transition],
        gamma: float,
        lam: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = []
        values = []
        dones = []
        for transition in transitions:
            obs = torch.from_numpy(transition.state).float().to(self.device)
            _, value, _, _ = self.model(obs)
            rewards.append(transition.reward)
            values.append(value.item())
            dones.append(float(transition.done))
        values.append(0.0)

        advantages = []
        gae = 0.0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def update(self, batch: PPOBatch) -> Dict[str, float]:
        for _ in range(self.ppo_config.epochs):
            logits, values, advantage_logits, uncertainty = self.model(batch.observations.to(self.device))
            dist = torch.distributions.Categorical(logits=logits)
            action_tensor = batch.actions.to(self.device).squeeze(-1)
            if action_tensor.dtype != torch.long:
                action_tensor = action_tensor.long()
            log_probs = dist.log_prob(action_tensor)
            entropy = dist.entropy().mean()

            ratios = torch.exp(log_probs - batch.old_log_probs.to(self.device))
            advantages = batch.advantages.to(self.device)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.ppo_config.clip_range, 1.0 + self.ppo_config.clip_range) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            returns = batch.returns.to(self.device)
            critic_loss = nn.functional.mse_loss(values, returns.unsqueeze(-1))

            aux_target = returns.unsqueeze(-1).expand_as(advantage_logits)
            auxiliary_loss = nn.functional.mse_loss(advantage_logits, aux_target)
            uncertainty_penalty = uncertainty.mean()

            loss = (
                actor_loss
                + self.ppo_config.value_loss_coef * critic_loss
                - self.ppo_config.entropy_coef * entropy
                + 0.1 * auxiliary_loss
                + 0.01 * uncertainty_penalty
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo_config.max_grad_norm)
            self.optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "auxiliary_loss": auxiliary_loss.item(),
            "uncertainty": uncertainty_penalty.item(),
        }


__all__ = ["PPOAgent", "PPOBatch"]
