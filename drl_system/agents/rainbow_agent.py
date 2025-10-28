"""Distributional Rainbow-style agent leveraging the shared backbone."""
from __future__ import annotations

import copy
from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn, optim

from ..config import AgentConfig, MemoryConfig, PPOConfig
from ..memory.replay_buffer import Transition
from ..optimization.adaptive_optimizer import AdaptiveOptimizer
from .dqn_agent import DQNAgent
from .ppo_agent import PPOBatch


class RainbowAgent(DQNAgent):
    """Implements a lightweight categorical DQN variant with multi-head diagnostics."""

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
        self.num_atoms = int(getattr(agent_config, "rainbow_atoms", 51))
        self.v_min = float(getattr(agent_config, "rainbow_v_min", -10.0))
        self.v_max = float(getattr(agent_config, "rainbow_v_max", 10.0))
        feature_dim = self.model.policy_head[0].in_features
        self.distribution_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_dim * self.num_atoms),
        ).to(device)
        self.target_distribution_head = copy.deepcopy(self.distribution_head).to(device)
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        support = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=device)
        self.support = support
        base_optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.distribution_head.parameters()),
            lr=3e-4,
            eps=1e-5,
        )
        self.optimizer = AdaptiveOptimizer(base_optimizer)
        self.target_distribution_head.load_state_dict(self.distribution_head.state_dict())

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
        latent = diagnostics["latent_features"]
        dist_logits = self.distribution_head(latent)
        dist_logits = dist_logits.view(-1, self.action_dim, self.num_atoms)
        dist_probs = torch.softmax(dist_logits, dim=-1)
        chosen_dist = torch.gather(
            dist_probs,
            1,
            actions.unsqueeze(-1).expand(-1, 1, self.num_atoms),
        ).squeeze(1)

        with torch.no_grad():
            _, _, _, _, target_diag = self.target_model(next_states)
            target_latent = target_diag["latent_features"]
            target_dist = torch.softmax(
                self.target_distribution_head(target_latent).view(-1, self.action_dim, self.num_atoms),
                dim=-1,
            )
            _, _, _, _, online_diag = self.model(next_states)
            online_next_latent = online_diag["latent_features"]
            online_dist = torch.softmax(
                self.distribution_head(online_next_latent).view(-1, self.action_dim, self.num_atoms),
                dim=-1,
            )
            online_q = torch.sum(online_dist * self.support.view(1, 1, -1), dim=-1)
            if self.double_q:
                next_actions = torch.argmax(online_q, dim=-1)
            else:
                next_actions = torch.argmax(target_dist.sum(dim=-1), dim=-1)
            batch_indices = torch.arange(target_dist.shape[0], device=self.device)
            next_dist = target_dist[batch_indices, next_actions]
            projected = self._project_distribution(next_dist, rewards, dones)

        loss = -(projected * torch.log(chosen_dist + 1e-8)).sum(dim=-1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.distribution_head.parameters()), self.ppo_config.max_grad_norm)
        self.optimizer.step()

        self._soft_update_target()

        return {
            "agent": "rainbow",
            "distributional_loss": float(loss.item()),
            "mean_reward": float(rewards.mean().item()),
            "uncertainty": float(uncertainty.mean().item()),
            "support_min": float(projected.min().item()),
            "support_max": float(projected.max().item()),
        }

    def _project_distribution(self, next_dist: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        batch_size = next_dist.shape[0]
        support = self.support.view(1, -1)
        tz = rewards + (1 - dones) * self.gamma * support
        tz = tz.clamp(min=self.v_min, max=self.v_max)
        b = (tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        m = torch.zeros(batch_size, self.num_atoms, device=self.device)
        for i in range(self.num_atoms):
            lower_mask = (l == i).float()
            upper_mask = (u == i).float()
            m[:, i] += torch.sum(next_dist * (upper_mask * (b - l.float())), dim=1)
            m[:, i] += torch.sum(next_dist * (lower_mask * (u.float() - b)), dim=1)
        m = m / m.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return m

    def _soft_update_target(self) -> None:
        super()._soft_update_target()
        if self.soft_update_tau <= 0:
            return
        with torch.no_grad():
            for target_param, param in zip(
                self.target_distribution_head.parameters(), self.distribution_head.parameters()
            ):
                target_param.data.mul_(1.0 - self.soft_update_tau).add_(self.soft_update_tau * param.data)


__all__ = ["RainbowAgent"]
