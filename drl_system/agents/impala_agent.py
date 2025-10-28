"""IMPALA/V-trace style agent built on the shared actor-critic."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from ..config import AgentConfig, MemoryConfig, PPOConfig
from .ppo_agent import PPOAgent, PPOBatch


class IMPALAgent(PPOAgent):
    """Applies V-trace corrections for off-policy trajectories."""

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
        self.vtrace_rho_bar = float(getattr(agent_config, "vtrace_rho_bar", 1.0))
        self.vtrace_c_bar = float(getattr(agent_config, "vtrace_c_bar", 1.0))

    def update(self, batch: PPOBatch, **_: Dict) -> Dict[str, float]:
        observations = batch.observations.to(self.device)
        (
            policy_logits,
            values,
            _adv_logits,
            uncertainty,
            diagnostics,
        ) = self.model(observations)
        values = values.squeeze(-1)
        dist = torch.distributions.Categorical(logits=policy_logits / self.temperature)
        actions = batch.actions.to(self.device).squeeze(-1)
        if actions.dtype != torch.long:
            actions = actions.long()
        log_probs = dist.log_prob(actions)
        behaviour_log_probs = batch.old_log_probs.to(self.device)
        ratios = torch.exp(log_probs - behaviour_log_probs)
        rho = torch.clamp(ratios, max=self.vtrace_rho_bar)
        c = torch.clamp(ratios, max=self.vtrace_c_bar)

        returns = batch.returns.to(self.device)
        deltas = rho * (returns - values)
        critic_target = values + deltas.detach()

        policy_loss = -(rho.detach() * (returns - values.detach()) * log_probs).mean()
        critic_loss = nn.functional.mse_loss(values, critic_target)
        entropy = dist.entropy().mean()
        trace_penalty = (1.0 - c).abs().mean()
        latent_consistency = nn.functional.mse_loss(
            diagnostics["latent_features"], diagnostics["reflection"].detach()
        )

        loss = (
            policy_loss
            + self.ppo_config.value_loss_coef * critic_loss
            - self.ppo_config.entropy_coef * entropy
            + 0.05 * trace_penalty
            + 0.05 * latent_consistency
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo_config.max_grad_norm)
        self.optimizer.step()

        return {
            "agent": "impala",
            "policy_loss": float(policy_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy.item()),
            "uncertainty": float(uncertainty.mean().item()),
            "trace_penalty": float(trace_penalty.item()),
            "latent_consistency": float(latent_consistency.item()),
        }


__all__ = ["IMPALAgent"]
