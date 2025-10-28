from __future__ import annotations

"""Synchronous Advantage Actor-Critic agent built on the PPO backbone."""

from typing import Dict

import torch
from torch import nn

from ..config import AgentConfig, MemoryConfig, PPOConfig
from .ppo_agent import PPOAgent, PPOBatch


class A2CAgent(PPOAgent):
    """Classic A2C agent that reuses the PPO model without importance sampling."""

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
        self.num_steps = max(1, int(agent_config.n_step))

    def update(self, batch: PPOBatch, **kwargs: Dict) -> Dict[str, float]:
        observations = batch.observations.to(self.device)
        returns = batch.returns.to(self.device)
        advantages = batch.advantages.to(self.device)
        logits, values, advantage_logits, uncertainty, diagnostics = self.model(observations)
        dist = torch.distributions.Categorical(logits=logits / self.temperature)
        action_tensor = batch.actions.to(self.device).squeeze(-1)
        if action_tensor.dtype != torch.long:
            action_tensor = action_tensor.long()
        log_probs = dist.log_prob(action_tensor)
        entropy = dist.entropy().mean()

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = nn.functional.mse_loss(values, returns.unsqueeze(-1))
        auxiliary_loss = nn.functional.mse_loss(advantage_logits, returns.unsqueeze(-1).expand_as(advantage_logits))
        uncertainty_penalty = uncertainty.mean()
        skill_alignment_loss = nn.functional.mse_loss(diagnostics["skills"], advantage_logits.detach())
        world_consistency_loss = nn.functional.mse_loss(diagnostics["world_prediction"], observations)
        evolution_regulariser = nn.functional.mse_loss(diagnostics["evolution"], diagnostics["skills"].detach())
        dynamics_consistency = nn.functional.mse_loss(diagnostics["dynamics"], observations)
        meta_alignment = nn.functional.mse_loss(diagnostics["meta_value"], returns.unsqueeze(-1))
        behaviour_alignment = nn.functional.mse_loss(diagnostics["behaviour_prior"], advantage_logits.detach())
        intrinsic_alignment = nn.functional.mse_loss(diagnostics["intrinsic_reward"], returns.unsqueeze(-1))
        consensus_alignment = nn.functional.mse_loss(diagnostics["consensus"], diagnostics["behaviour_prior"].detach())
        mode_consistency = nn.functional.mse_loss(diagnostics["mode_logits"], diagnostics["hierarchy_context"].detach())
        option_entropy = torch.distributions.Categorical(logits=diagnostics["options"]).entropy().mean()

        loss = (
            actor_loss
            + self.ppo_config.value_loss_coef * critic_loss
            - self.ppo_config.entropy_coef * entropy
            + 0.1 * auxiliary_loss
            + 0.01 * uncertainty_penalty
            + 0.05 * (skill_alignment_loss + world_consistency_loss)
            + 0.02 * evolution_regulariser
            + 0.05 * dynamics_consistency
            + 0.02 * meta_alignment
            + 0.01 * behaviour_alignment
            + 0.01 * intrinsic_alignment
            + 0.02 * consensus_alignment
            + 0.02 * mode_consistency
            - 0.005 * option_entropy
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo_config.max_grad_norm)
        self.optimizer.step()

        return {
            "agent": "a2c",
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "auxiliary_loss": auxiliary_loss.item(),
            "uncertainty": uncertainty_penalty.item(),
            "skill_alignment_loss": skill_alignment_loss.item(),
            "world_consistency_loss": world_consistency_loss.item(),
            "evolution_regulariser": evolution_regulariser.item(),
            "dynamics_consistency": dynamics_consistency.item(),
            "meta_value_alignment": meta_alignment.item(),
            "behaviour_prior_alignment": behaviour_alignment.item(),
            "intrinsic_alignment": intrinsic_alignment.item(),
            "option_entropy": option_entropy.item(),
            "consensus_alignment": consensus_alignment.item(),
            "mode_consistency": mode_consistency.item(),
        }


__all__ = ["A2CAgent"]
