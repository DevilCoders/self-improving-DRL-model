"""Factory helpers for instantiating configured agents."""
from __future__ import annotations

from typing import Callable

import torch

from ..config import AgentConfig, SystemConfig
from .a3c_agent import A3CAgent
from .ddpg_agent import DDPGAgent
from .dqn_agent import DQNAgent
from .ppo_agent import PPOAgent
from .sac_agent import SACAgent
from .td3_agent import TD3Agent

AgentBuilder = Callable[[int, int, SystemConfig, torch.device], PPOAgent]


def create_agent(
    agent_config: AgentConfig,
    config: SystemConfig,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
):
    """Return an agent instance matching ``agent_config.type``."""

    agent_type = agent_config.type.lower()
    if agent_type == "ppo":
        return PPOAgent(obs_dim, action_dim, config.memory, config.ppo, device, agent_config)
    if agent_type == "a3c":
        return A3CAgent(obs_dim, action_dim, config.memory, config.ppo, device, agent_config)
    if agent_type == "sac":
        return SACAgent(obs_dim, action_dim, config.memory, config.ppo, device, agent_config)
    if agent_type == "dqn":
        return DQNAgent(obs_dim, action_dim, config.memory, config.ppo, device, agent_config)
    if agent_type == "ddpg":
        return DDPGAgent(obs_dim, action_dim, config.memory, config.ppo, device, agent_config)
    if agent_type == "td3":
        return TD3Agent(obs_dim, action_dim, config.memory, config.ppo, device, agent_config)
    raise ValueError(f"Unsupported agent type: {agent_config.type}")


__all__ = ["create_agent", "AgentBuilder"]
