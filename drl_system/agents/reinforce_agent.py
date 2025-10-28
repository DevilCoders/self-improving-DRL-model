"""REINFORCE-style agent leveraging the shared actor-critic backbone."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from ..config import AgentConfig, MemoryConfig, PPOConfig
from .ppo_agent import PPOAgent, PPOBatch


class REINFORCEAgent(PPOAgent):
    """Implements a vanilla policy-gradient update with entropy regularisation."""

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
        self.baseline_momentum = float(getattr(agent_config, "reinforce_baseline_momentum", 0.9))
        self._running_baseline: float = 0.0

    def update(self, batch: PPOBatch, **_: Dict) -> Dict[str, float]:
        observations = batch.observations.to(self.device)
        (
            policy_logits,
            values,
            _adv_logits,
            uncertainty,
            diagnostics,
        ) = self.model(observations)
        dist = torch.distributions.Categorical(logits=policy_logits / self.temperature)
        actions = batch.actions.to(self.device).squeeze(-1)
        if actions.dtype != torch.long:
            actions = actions.long()
        log_probs = dist.log_prob(actions)
        returns = batch.returns.to(self.device)
        baseline = values.detach().squeeze(-1)

        with torch.no_grad():
            batch_mean = baseline.mean().item()
            self._running_baseline = (
                self.baseline_momentum * self._running_baseline
                + (1.0 - self.baseline_momentum) * batch_mean
            )
        centred_returns = returns - float(self._running_baseline)

        policy_loss = -(log_probs * centred_returns.detach()).mean()
        value_loss = nn.functional.mse_loss(values.squeeze(-1), returns)
        entropy = dist.entropy().mean()
        behaviour_alignment = nn.functional.mse_loss(
            diagnostics["behaviour_prior"], diagnostics["latent_features"].detach()
        )

        loss = policy_loss + 0.5 * value_loss - self.ppo_config.entropy_coef * entropy + 0.05 * behaviour_alignment
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo_config.max_grad_norm)
        self.optimizer.step()

        return {
            "agent": "reinforce",
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "uncertainty": float(uncertainty.mean().item()),
            "behaviour_alignment": float(behaviour_alignment.item()),
        }


__all__ = ["REINFORCEAgent"]
