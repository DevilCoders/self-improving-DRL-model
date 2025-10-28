"""Quantile regression DQN agent built on the shared actor-critic."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn

from ..config import AgentConfig, MemoryConfig, PPOConfig
from ..memory.replay_buffer import Transition
from .dqn_agent import DQNAgent
from .ppo_agent import PPOBatch


class QuantileDQNAgent(DQNAgent):
    """Distributional extension of :class:`DQNAgent` using quantile regression."""

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
        self.quantile_atoms = max(8, int(getattr(agent_config, "quantile_atoms", 32)))
        tau = (2 * torch.arange(self.quantile_atoms, dtype=torch.float32) + 1) / (2.0 * self.quantile_atoms)
        self.registered_tau = tau.unsqueeze(0)  # lazily moved to device during updates

    def update(self, batch: PPOBatch, **kwargs: Dict) -> Dict[str, float]:
        transitions: Sequence[Transition] | None = kwargs.get("transitions")
        if not transitions:
            return super().update(batch, **kwargs)

        states = torch.from_numpy(np.stack([t.state for t in transitions])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([t.next_state for t in transitions])).float().to(self.device)
        rewards = torch.from_numpy(np.asarray([t.reward for t in transitions], dtype=np.float32)).unsqueeze(-1).to(
            self.device
        )
        dones = (
            torch.from_numpy(np.asarray([float(t.done) for t in transitions], dtype=np.float32))
            .unsqueeze(-1)
            .to(self.device)
        )
        actions = torch.from_numpy(
            np.asarray(
                [int(np.array(t.action).reshape(-1)[0]) if np.size(t.action) else 0 for t in transitions],
                dtype=np.int64,
            )
        ).unsqueeze(-1).to(self.device)

        logits, _, _, _, diagnostics = self.model(states)
        quantiles = diagnostics["quantiles"]
        current_quantiles = quantiles.gather(
            1, actions.unsqueeze(-1).expand(-1, -1, self.quantile_atoms)
        ).squeeze(1)

        tau = self.registered_tau.to(self.device)

        with torch.no_grad():
            _, _, _, _, target_diag = self.target_model(next_states)
            target_quantiles_all = target_diag["quantiles"]
            if self.double_q:
                online_next = self.model(next_states)[-1]["quantiles"].mean(dim=-1)
                next_actions = torch.argmax(online_next, dim=-1, keepdim=True)
            else:
                next_actions = torch.argmax(target_quantiles_all.mean(dim=-1), dim=-1, keepdim=True)
            next_quantiles = target_quantiles_all.gather(
                1, next_actions.unsqueeze(-1).expand(-1, -1, self.quantile_atoms)
            ).squeeze(1)
            target_quantiles = rewards + self.gamma * (1 - dones) * next_quantiles

        td_errors = target_quantiles.unsqueeze(-1) - current_quantiles.unsqueeze(-2)
        huber_loss = self._quantile_huber(td_errors)
        quantile_loss = (
            torch.abs(tau.unsqueeze(-1) - (td_errors.detach() < 0).float()) * huber_loss
        ).mean()

        self.optimizer.zero_grad()
        quantile_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo_config.max_grad_norm)
        self.optimizer.step()
        self._soft_update_target()

        return {
            "agent": "quantile_dqn",
            "quantile_loss": float(quantile_loss.item()),
            "mean_td": float(td_errors.abs().mean().item()),
            "expected_q": float(current_quantiles.mean().item()),
        }

    @staticmethod
    def _quantile_huber(errors: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
        abs_error = torch.abs(errors)
        quadratic = torch.clamp(abs_error, max=kappa)
        linear = abs_error - quadratic
        return 0.5 * quadratic ** 2 + kappa * linear


__all__ = ["QuantileDQNAgent"]
