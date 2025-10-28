"""IMPALA-style agent leveraging V-trace corrections on the shared actor-critic."""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from ..config import AgentConfig, MemoryConfig, PPOConfig
from ..memory.replay_buffer import Transition
from .ppo_agent import PPOAgent, PPOBatch


class IMPALAAgent(PPOAgent):
    """Implements an IMPALA style learner with lightweight V-trace weighting."""

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
        self.clip_rho = float(getattr(agent_config, "impala_clip_rho", 1.0))
        self.clip_c = float(getattr(agent_config, "impala_clip_c", 1.0))

    def compute_advantages(
        self,
        transitions: Iterable[Transition],
        gamma: float,
        lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transitions = list(transitions)
        if not transitions:
            return super().compute_advantages([], gamma, lam)

        rewards: list[float] = []
        dones: list[float] = []
        values: list[float] = []
        next_values: list[float] = []
        behaviour_log_probs: list[float] = []
        policy_log_probs: list[float] = []

        for transition in transitions:
            obs = torch.from_numpy(np.asarray(transition.state)).float().to(self.device)
            next_obs = torch.from_numpy(np.asarray(transition.next_state)).float().to(self.device)
            logits, value, _, _, _ = self.model(obs)
            with torch.no_grad():
                _, next_value, _, _, _ = self.model(next_obs)
            dist = torch.distributions.Categorical(logits=logits / self.temperature)
            action = torch.as_tensor(np.array(transition.action)).long().to(self.device)
            if action.dim() > 0:
                action = action.view(-1)[0]
            behaviour_lp = float(
                transition.info.get("behaviour_log_prob", transition.info.get("log_prob", 0.0))
            )
            policy_lp = float(dist.log_prob(action).item())

            rewards.append(float(transition.reward))
            dones.append(float(transition.done))
            values.append(float(value.view(-1)[0].item()))
            next_values.append(float(next_value.view(-1)[0].item()))
            behaviour_log_probs.append(behaviour_lp)
            policy_log_probs.append(policy_lp)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)
        values_tensor = torch.tensor(values, dtype=torch.float32)
        next_values_tensor = torch.tensor(next_values, dtype=torch.float32)
        behaviour_tensor = torch.tensor(behaviour_log_probs, dtype=torch.float32)
        policy_tensor = torch.tensor(policy_log_probs, dtype=torch.float32)

        rhos = torch.exp(policy_tensor - behaviour_tensor)
        clipped_rhos = torch.clamp(rhos, max=self.clip_rho)
        clipped_cs = torch.clamp(rhos, max=self.clip_c)

        advantages: list[float] = []
        vs: list[float] = []
        vs_next = 0.0
        for step in reversed(range(len(rewards))):
            bootstrap = (1.0 - dones_tensor[step]) * next_values_tensor[step]
            delta = clipped_rhos[step] * (
                rewards_tensor[step] + gamma * bootstrap - values_tensor[step]
            )
            vs_step = values_tensor[step] + delta + gamma * clipped_cs[step] * (
                vs_next - bootstrap
            )
            advantage = vs_step - values_tensor[step]
            vs.insert(0, float(vs_step.item()))
            advantages.insert(0, float(advantage.item()))
            vs_next = vs_step.item()

        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(vs, dtype=torch.float32)

    def update(self, batch: PPOBatch, **kwargs: Dict) -> Dict[str, float]:
        importance_weights = None
        if "transitions" in kwargs:
            transitions: Sequence[Transition] = kwargs["transitions"]
            if transitions:
                behaviour = torch.tensor(
                    [
                        t.info.get("behaviour_log_prob", t.info.get("log_prob", 0.0))
                        for t in transitions
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                observations = torch.from_numpy(
                    np.stack([np.asarray(t.state) for t in transitions])
                ).float().to(self.device)
                actions = torch.from_numpy(
                    np.stack([
                        np.asarray(t.action).reshape(-1)[0] if np.size(t.action) else 0
                        for t in transitions
                    ])
                ).long().to(self.device)
                logits, _, _, _, _ = self.model(observations)
                dist = torch.distributions.Categorical(logits=logits / self.temperature)
                policy = dist.log_prob(actions)
                ratios = torch.exp(policy - behaviour)
                importance_weights = torch.clamp(ratios.detach(), max=self.clip_rho)

        for _ in range(self.ppo_config.epochs):
            logits, values, advantage_logits, uncertainty, diagnostics = self.model(
                batch.observations.to(self.device)
            )
            dist = torch.distributions.Categorical(logits=logits / self.temperature)
            action_tensor = batch.actions.to(self.device).squeeze(-1)
            if action_tensor.dtype != torch.long:
                action_tensor = action_tensor.long()
            log_probs = dist.log_prob(action_tensor)
            entropy = dist.entropy().mean()

            ratios = torch.exp(log_probs - batch.old_log_probs.to(self.device))
            advantages = batch.advantages.to(self.device)
            if importance_weights is not None:
                clipped_weights = importance_weights[: advantages.shape[0]]
                weighted_advantages = clipped_weights * advantages
            else:
                weighted_advantages = advantages
            actor_loss = -(ratios * weighted_advantages).mean()

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

            loss = (
                actor_loss
                + self.ppo_config.value_loss_coef * critic_loss
                - self.ppo_config.entropy_coef * entropy
                + 0.1 * auxiliary_loss
                + 0.01 * uncertainty_penalty
                + 0.05 * (skill_alignment_loss + world_consistency_loss)
                + 0.02 * evolution_regulariser
                + 0.05 * nn.functional.mse_loss(diagnostics["dynamics"], batch.observations.to(self.device))
                + 0.02 * nn.functional.mse_loss(diagnostics["meta_value"], returns.unsqueeze(-1))
                + 0.01 * nn.functional.mse_loss(
                    diagnostics["behaviour_prior"], advantage_logits.detach()
                )
                + 0.01 * (policy_embedding_norm + latent_drift_consistency.abs())
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo_config.max_grad_norm)
            self.optimizer.step()

        metrics = {
            "agent": "impala",
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy.item()),
            "auxiliary_loss": float(auxiliary_loss.item()),
            "uncertainty": float(uncertainty_penalty.item()),
            "skill_alignment_loss": float(skill_alignment_loss.item()),
            "world_consistency_loss": float(world_consistency_loss.item()),
            "evolution_regulariser": float(evolution_regulariser.item()),
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
            "importance_weight_mean": float(importance_weights.mean().item())
            if importance_weights is not None
            else 1.0,
        }
        return metrics


__all__ = ["IMPALAAgent"]
