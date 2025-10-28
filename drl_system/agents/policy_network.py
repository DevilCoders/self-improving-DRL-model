"""Policy and value networks for PPO style agents."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        hierarchy_levels: int = 2,
        transformer_layers: int = 1,
    ) -> None:
        super().__init__()
        blocks = []
        last_dim = obs_dim
        for hidden in hidden_sizes:
            layer = nn.Linear(last_dim, hidden)
            components = [layer]
            if use_layer_norm:
                components.append(nn.LayerNorm(hidden))
            components.append(nn.GELU())
            if dropout > 0:
                components.append(nn.Dropout(dropout))
            block = nn.Sequential(*components)
            blocks.append(block)
            last_dim = hidden
        self.shared = nn.ModuleList(blocks)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(last_dim, last_dim), nn.GELU()) for _ in range(3)]
        )
        self.gating_head = nn.Linear(last_dim, len(self.experts))
        self.memory_projector = nn.Linear(last_dim * 2, last_dim)
        self.attention = nn.MultiheadAttention(last_dim, num_heads=4, batch_first=True)
        self.context_adapter = nn.Linear(obs_dim, last_dim)
        self.hierarchy_levels = max(1, hierarchy_levels)
        self.hierarchical_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(last_dim, last_dim),
                    nn.GELU(),
                    nn.LayerNorm(last_dim),
                )
                for _ in range(self.hierarchy_levels)
            ]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=last_dim,
            nhead=4,
            dim_feedforward=last_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, transformer_layers))
        self.latent_adapter = nn.Linear(last_dim, last_dim)
        self.policy_head = nn.Sequential(nn.Linear(last_dim, last_dim), nn.GELU(), nn.Linear(last_dim, action_dim))
        self.value_head = nn.Sequential(nn.Linear(last_dim, last_dim), nn.GELU(), nn.Linear(last_dim, 1))
        self.advantage_head = nn.Sequential(nn.Linear(last_dim, last_dim), nn.GELU(), nn.Linear(last_dim, action_dim))
        self.uncertainty_head = nn.Sequential(nn.Linear(last_dim, last_dim), nn.ReLU(), nn.Linear(last_dim, 1), nn.Softplus())
        self.skill_head = nn.Sequential(nn.Linear(last_dim, last_dim), nn.GELU(), nn.Linear(last_dim, action_dim))
        self.world_model_head = nn.Sequential(nn.Linear(last_dim, last_dim), nn.GELU(), nn.Linear(last_dim, obs_dim))
        self.evolution_head = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.GELU(),
            nn.Linear(last_dim, action_dim),
        )
        self.predictive_coding_head = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.Tanh(),
        )
        self.reflection_head = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.GELU(),
            nn.Linear(last_dim, last_dim),
        )
        self.dynamics_head = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.GELU(),
            nn.Linear(last_dim, obs_dim),
        )
        self.q_head = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, action_dim),
        )
        self.twin_q_head = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, action_dim),
        )
        self.meta_value_head = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.GELU(),
            nn.Linear(last_dim, 1),
        )
        self.behaviour_head = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.GELU(),
            nn.Linear(last_dim, action_dim),
        )
        self._memory_state: torch.Tensor | None = None
        self._hierarchy_trace: torch.Tensor | None = None

    def reset_memory(self) -> None:
        self._memory_state = None

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        squeezed = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeezed = True
        features = obs
        for block in self.shared:
            residual = features
            features = block(features)
            if residual.shape[-1] == features.shape[-1]:
                features = 0.5 * (features + residual)
        contextual = torch.tanh(self.context_adapter(obs))
        features = features + 0.5 * contextual
        hierarchy_states = []
        hierarchy_input = features
        for layer in self.hierarchical_layers:
            hierarchy_input = layer(hierarchy_input)
            hierarchy_states.append(hierarchy_input)
        hierarchy_tensor = torch.stack(hierarchy_states, dim=1)
        transformer_output = self.transformer(hierarchy_tensor)
        hierarchy_context = transformer_output.mean(dim=1)
        latent_context = torch.tanh(self.latent_adapter(hierarchy_context))
        predictive_code = self.predictive_coding_head(hierarchy_context)
        reflection = self.reflection_head(latent_context)
        features = features + 0.5 * latent_context + 0.1 * reflection
        expert_outputs = torch.stack([expert(features) for expert in self.experts], dim=1)
        gating = torch.softmax(self.gating_head(features), dim=-1).unsqueeze(-1)
        mixture = (expert_outputs * gating).sum(dim=1)
        features = features + mixture
        memory = self._initialise_memory(features)
        combined = torch.cat([features, memory], dim=-1)
        memory_update = torch.tanh(self.memory_projector(combined))
        attn_input = torch.stack([features, memory, memory_update], dim=1)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attended = attn_output[:, 0, :]
        features = features + attended
        self._memory_state = 0.85 * memory + 0.15 * memory_update.detach()
        self._hierarchy_trace = hierarchy_tensor.detach()

        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        uncertainty = self.uncertainty_head(features)
        skills = self.skill_head(features)
        world_prediction = self.world_model_head(features)
        evolution_logits = self.evolution_head(features)
        dynamics = self.dynamics_head(features)
        q_values = self.q_head(features)
        twin_q_values = self.twin_q_head(features)
        meta_value = self.meta_value_head(features)
        behaviour_prior = self.behaviour_head(features)
        latent_features = features

        if squeezed:
            policy_logits = policy_logits.squeeze(0)
            value = value.squeeze(0)
            advantage = advantage.squeeze(0)
            uncertainty = uncertainty.squeeze(0)
            skills = skills.squeeze(0)
            world_prediction = world_prediction.squeeze(0)
            evolution_logits = evolution_logits.squeeze(0)
            dynamics = dynamics.squeeze(0)
            q_values = q_values.squeeze(0)
            twin_q_values = twin_q_values.squeeze(0)
            meta_value = meta_value.squeeze(0)
            behaviour_prior = behaviour_prior.squeeze(0)
            latent_features = latent_features.squeeze(0)
            self._memory_state = self._memory_state.squeeze(0) if self._memory_state is not None else None
            hierarchy_context = hierarchy_context.squeeze(0)
            predictive_code = predictive_code.squeeze(0)
            reflection = reflection.squeeze(0)
            hierarchy_tensor = hierarchy_tensor.squeeze(0)

        diagnostics = {
            "skills": skills,
            "world_prediction": world_prediction,
            "evolution": evolution_logits,
            "predictive_code": predictive_code,
            "hierarchy_context": hierarchy_context,
            "hierarchy_trace": hierarchy_tensor,
            "reflection": reflection,
            "dynamics": dynamics,
            "q_values": q_values,
            "twin_q_values": twin_q_values,
            "meta_value": meta_value,
            "behaviour_prior": behaviour_prior,
            "latent_features": latent_features,
        }
        return policy_logits, value, advantage, uncertainty, diagnostics

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, value, advantage, uncertainty, diagnostics = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            value,
            {
                "advantage_logits": advantage.detach(),
                "uncertainty": uncertainty.detach(),
                "skills": diagnostics["skills"].detach(),
                "world_prediction": diagnostics["world_prediction"].detach(),
                "evolution": diagnostics["evolution"].detach(),
                "predictive_code": diagnostics["predictive_code"].detach(),
                "hierarchy_context": diagnostics["hierarchy_context"].detach(),
                "hierarchy_trace": diagnostics["hierarchy_trace"].detach(),
                "reflection": diagnostics["reflection"].detach(),
                "dynamics": diagnostics["dynamics"].detach(),
                "q_values": diagnostics["q_values"].detach(),
                "twin_q_values": diagnostics["twin_q_values"].detach(),
                "meta_value": diagnostics["meta_value"].detach(),
                "behaviour_prior": diagnostics["behaviour_prior"].detach(),
                "latent_features": diagnostics["latent_features"].detach(),
            },
        )

    def _initialise_memory(self, features: torch.Tensor) -> torch.Tensor:
        if self._memory_state is None:
            self._memory_state = torch.zeros_like(features)
        elif self._memory_state.shape != features.shape:
            self._memory_state = torch.zeros_like(features)
        return self._memory_state.to(features.device)


__all__ = ["ActorCritic"]
