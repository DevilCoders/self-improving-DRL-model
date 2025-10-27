"""Implementation of a PPO style agent with meta-learning hooks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn, optim

from ..config import AgentConfig, MemoryConfig, PPOConfig
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
        agent_config: AgentConfig | None = None,
    ) -> None:
        self.device = device
        self.agent_config = agent_config or AgentConfig()
        hidden_sizes = tuple(self.agent_config.hidden_sizes)
        self.model = ActorCritic(
            obs_dim,
            action_dim,
            hidden_sizes=hidden_sizes,
            hierarchy_levels=self.agent_config.hierarchy_levels,
            transformer_layers=self.agent_config.transformer_layers,
        ).to(device)
        base_optimizer = optim.Adam(self.model.parameters(), lr=3e-4, eps=1e-5)
        self.optimizer = AdaptiveOptimizer(base_optimizer)
        self.ppo_config = ppo_config
        self.temperature = max(1e-3, float(self.agent_config.temperature))

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
            _, value, _, _, _ = self.model(obs)
            rewards.append(transition.reward)
            values.append(float(value.view(-1)[0].item()))
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

    def prepare_batch(
        self,
        transitions: Iterable[Transition],
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> PPOBatch:
        observations = torch.from_numpy(np.stack([t.state for t in transitions])).float()
        actions = torch.from_numpy(np.stack([t.action for t in transitions])).float()
        old_log_probs = torch.tensor([t.info.get("log_prob", 0.0) for t in transitions]).float()
        return PPOBatch(
            observations=observations,
            actions=actions,
            old_log_probs=old_log_probs,
            returns=returns.detach(),
            advantages=advantages.detach(),
        )

    def update(self, batch: PPOBatch, **_: Dict) -> Dict[str, float]:
        for _ in range(self.ppo_config.epochs):
            (
                logits,
                values,
                advantage_logits,
                uncertainty,
                diagnostics,
            ) = self.model(batch.observations.to(self.device))
            dist = torch.distributions.Categorical(logits=logits / self.temperature)
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
            skill_alignment_loss = nn.functional.mse_loss(diagnostics["skills"], advantage_logits.detach())
            world_consistency_loss = nn.functional.mse_loss(
                diagnostics["world_prediction"], batch.observations.to(self.device)
            )
            evolution_regulariser = nn.functional.mse_loss(
                diagnostics["evolution"], diagnostics["skills"].detach()
            )

            loss = (
                actor_loss
                + self.ppo_config.value_loss_coef * critic_loss
                - self.ppo_config.entropy_coef * entropy
                + 0.1 * auxiliary_loss
                + 0.01 * uncertainty_penalty
                + 0.05 * (skill_alignment_loss + world_consistency_loss)
                + 0.02 * evolution_regulariser
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
            "skill_alignment_loss": skill_alignment_loss.item(),
            "world_consistency_loss": world_consistency_loss.item(),
            "evolution_regulariser": evolution_regulariser.item(),
        }


__all__ = ["PPOAgent", "PPOBatch"]
