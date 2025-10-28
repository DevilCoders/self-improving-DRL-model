"""Deep Q-Network style agent leveraging the shared actor-critic backbone."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn

from ..config import AgentConfig, MemoryConfig, PPOConfig
from ..memory.replay_buffer import Transition
from .ppo_agent import PPOAgent, PPOBatch


class DQNAgent(PPOAgent):
    """Implements a lightweight distributional DQN update on the actor-critic."""

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
        self.target_model = type(self.model)(
            obs_dim,
            action_dim,
            hidden_sizes=tuple(agent_config.hidden_sizes),
            hierarchy_levels=agent_config.hierarchy_levels,
            transformer_layers=agent_config.transformer_layers,
        ).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.double_q = bool(getattr(agent_config, "double_q", True))
        self.soft_update_tau = float(agent_config.soft_update_tau)

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

        logits, _, _, uncertainty, diagnostics = self.model(states)
        q_values = diagnostics["q_values"]
        twin_q_values = diagnostics["twin_q_values"]
        current_q = 0.5 * (
            q_values.gather(1, actions)
            + twin_q_values.gather(1, actions)
        )

        with torch.no_grad():
            target_logits, _, _, _, target_diag = self.target_model(next_states)
            next_q = torch.min(target_diag["q_values"], target_diag["twin_q_values"])
            if self.double_q:
                selector = torch.argmax(logits.detach(), dim=-1, keepdim=True)
                next_q = next_q.gather(1, selector)
            else:
                next_q = next_q.max(dim=1, keepdim=True)[0]
            targets = rewards + self.gamma * (1 - dones) * next_q

        loss = nn.functional.mse_loss(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo_config.max_grad_norm)
        self.optimizer.step()

        self._soft_update_target()

        return {
            "agent": "dqn",
            "q_loss": loss.item(),
            "mean_q": float(current_q.mean().item()),
            "target_gap": float(torch.abs(targets - current_q).mean().item()),
            "uncertainty": float(uncertainty.mean().item()),
        }

    def _soft_update_target(self) -> None:
        if self.soft_update_tau <= 0:
            return
        with torch.no_grad():
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.mul_(1.0 - self.soft_update_tau).add_(self.soft_update_tau * param.data)


__all__ = ["DQNAgent"]
