"""TRPO-style agent with adaptive KL regularisation atop the shared actor-critic."""
from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn

from ..config import AgentConfig, MemoryConfig, PPOConfig
from ..memory.replay_buffer import Transition
from .ppo_agent import PPOAgent, PPOBatch


class TRPOAgent(PPOAgent):
    """Simplified TRPO agent using KL penalties for trust-region enforcement."""

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
        self.kl_target = float(getattr(agent_config, "trpo_kl_target", 0.01))
        self.kl_penalty = float(getattr(agent_config, "trpo_kl_penalty", 1.0))
        self.kl_adjust_rate = float(getattr(agent_config, "trpo_kl_adjust_rate", 1.5))

    def update(self, batch: PPOBatch, **_: Dict) -> Dict[str, float]:
        logits, values, advantage_logits, uncertainty, diagnostics = self.model(
            batch.observations.to(self.device)
        )
        dist = torch.distributions.Categorical(logits=logits / self.temperature)
        action_tensor = batch.actions.to(self.device).squeeze(-1)
        if action_tensor.dtype != torch.long:
            action_tensor = action_tensor.long()
        log_probs = dist.log_prob(action_tensor)
        old_log_probs = batch.old_log_probs.to(self.device)
        entropy = dist.entropy().mean()

        advantages = batch.advantages.to(self.device)
        actor_loss = -(log_probs * advantages).mean()

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
        policy_embedding_norm = diagnostics["policy_embedding"].norm(dim=-1).mean()
        trajectory_value = diagnostics["trajectory_value"].mean()
        latent_drift_consistency = diagnostics["latent_drift"].mean()

        sample_kl = (old_log_probs - log_probs).mean()
        kl_loss = self.kl_penalty * (sample_kl - self.kl_target) ** 2

        loss = (
            actor_loss
            + self.ppo_config.value_loss_coef * critic_loss
            - self.ppo_config.entropy_coef * entropy
            + 0.1 * auxiliary_loss
            + 0.01 * uncertainty_penalty
            + 0.05 * (skill_alignment_loss + world_consistency_loss)
            + 0.02 * evolution_regulariser
            + kl_loss
            + 0.05 * nn.functional.mse_loss(diagnostics["dynamics"], batch.observations.to(self.device))
            + 0.02 * nn.functional.mse_loss(diagnostics["meta_value"], returns.unsqueeze(-1))
            + 0.01 * nn.functional.mse_loss(diagnostics["behaviour_prior"], advantage_logits.detach())
            + 0.01 * (policy_embedding_norm + latent_drift_consistency.abs())
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo_config.max_grad_norm)
        self.optimizer.step()

        # Adapt KL penalty for future iterations.
        kl_value = float(sample_kl.item())
        if kl_value > self.kl_target * 2:
            self.kl_penalty *= self.kl_adjust_rate
        elif kl_value < self.kl_target / 2:
            self.kl_penalty /= self.kl_adjust_rate

        return {
            "agent": "trpo",
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy.item()),
            "auxiliary_loss": float(auxiliary_loss.item()),
            "uncertainty": float(uncertainty_penalty.item()),
            "skill_alignment_loss": float(skill_alignment_loss.item()),
            "world_consistency_loss": float(world_consistency_loss.item()),
            "evolution_regulariser": float(evolution_regulariser.item()),
            "kl_divergence": kl_value,
            "kl_penalty": float(self.kl_penalty),
            "dynamics_consistency": float(
                nn.functional.mse_loss(diagnostics["dynamics"], batch.observations.to(self.device)).item()
            ),
            "meta_value_alignment": float(
                nn.functional.mse_loss(diagnostics["meta_value"], returns.unsqueeze(-1)).item()
            ),
            "behaviour_prior_alignment": float(
                nn.functional.mse_loss(diagnostics["behaviour_prior"], advantage_logits.detach()).item()
            ),
            "policy_embedding_norm": float(policy_embedding_norm.item()),
            "trajectory_value_mean": float(trajectory_value.item()),
            "latent_drift_mean": float(latent_drift_consistency.item()),
        }

    def compute_advantages(
        self,
        transitions: Iterable[Transition],
        gamma: float,
        lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TRPO relies on GAE, so reuse the parent implementation for stability.
        return super().compute_advantages(transitions, gamma, lam)


__all__ = ["TRPOAgent"]
