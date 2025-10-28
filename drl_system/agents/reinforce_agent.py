"""Implementation of a REINFORCE style policy-gradient agent."""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from ..config import AgentConfig, MemoryConfig, PPOConfig
from ..memory.replay_buffer import Transition
from .ppo_agent import PPOAgent, PPOBatch


class ReinforceAgent(PPOAgent):
    """Classic Monte Carlo policy gradient with an adaptive baseline."""

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
        self.use_baseline = bool(getattr(agent_config, "reinforce_baseline", True))

    def compute_advantages(
        self,
        transitions: Iterable[Transition],
        gamma: float,
        lam: float,  # noqa: ARG002 - interface compatibility
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Monte-Carlo returns with an optional learned baseline."""

        collected: Sequence[Transition] = list(transitions)
        returns: list[float] = []
        g_return = 0.0
        for transition in reversed(collected):
            g_return = transition.reward + gamma * g_return * (1.0 - float(transition.done))
            returns.insert(0, g_return)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        if not collected:
            return torch.zeros(0), torch.zeros(0)

        values: list[float] = []
        if self.use_baseline:
            for transition in collected:
                obs = torch.from_numpy(transition.state).float().to(self.device)
                _, value, _, _, _ = self.model(obs)
                values.append(float(value.view(-1)[0].item()))
        else:
            values = [returns_tensor.mean().item()] * len(collected)
        values_tensor = torch.tensor(values, dtype=torch.float32)
        advantages = returns_tensor - values_tensor
        return advantages, returns_tensor

    def update(self, batch: PPOBatch, **_: Dict) -> Dict[str, float]:
        observations = batch.observations.to(self.device)
        logits, values, advantage_logits, uncertainty, diagnostics = self.model(observations)
        distribution = Categorical(logits=logits / self.temperature)
        actions = batch.actions.to(self.device).long().squeeze(-1)
        log_probs = distribution.log_prob(actions)

        returns = batch.returns.to(self.device)
        advantages = batch.advantages.to(self.device)
        baseline = values.squeeze(-1)
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = nn.functional.mse_loss(baseline, returns)
        entropy = distribution.entropy().mean()
        auxiliary_loss = nn.functional.mse_loss(advantage_logits, returns.unsqueeze(-1))
        evolution_loss = nn.functional.mse_loss(diagnostics["evolution"], diagnostics["skills"].detach())
        risk_regulariser = diagnostics["risk"].abs().mean()

        loss = (
            policy_loss
            + 0.5 * value_loss
            + 0.05 * auxiliary_loss
            + 0.02 * evolution_loss
            + 0.01 * risk_regulariser
            - 0.01 * entropy
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo_config.max_grad_norm)
        self.optimizer.step()

        return {
            "agent": "reinforce",
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "auxiliary_loss": float(auxiliary_loss.item()),
            "evolution_loss": float(evolution_loss.item()),
            "risk_regulariser": float(risk_regulariser.item()),
        }


__all__ = ["ReinforceAgent"]
