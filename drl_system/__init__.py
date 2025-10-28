"""Self-improving DRL system package with lazy imports."""

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:  # pragma: no cover - thin wrapper
    mapping = {
        "PPOAgent": "drl_system.agents.ppo_agent",
        "PPOBatch": "drl_system.agents.ppo_agent",
        "A3CAgent": "drl_system.agents.a3c_agent",
        "SACAgent": "drl_system.agents.sac_agent",
        "DQNAgent": "drl_system.agents.dqn_agent",
        "DDPGAgent": "drl_system.agents.ddpg_agent",
        "TD3Agent": "drl_system.agents.td3_agent",
        "IMPALAAgent": "drl_system.agents.impala_agent",
        "TRPOAgent": "drl_system.agents.trpo_agent",
        "create_agent": "drl_system.agents.factory",
        "SystemConfig": "drl_system.config",
        "SyntheticDatasetBuilder": "drl_system.data.dataset_builder",
        "EpisodicMemory": "drl_system.memory.replay_buffer",
        "ReplayBuffer": "drl_system.memory.replay_buffer",
        "Transition": "drl_system.memory.replay_buffer",
        "PreprocessingPipeline": "drl_system.preprocess.preprocessor",
        "DynamicTokenizer": "drl_system.tokenization.tokenizer",
        "SequenceEncoder": "drl_system.tokenization.encoder",
        "SequenceDecoder": "drl_system.tokenization.decoder",
        "Trainer": "drl_system.training.trainer",
        "SystemManager": "drl_system.system.system_manager",
        "CommandResult": "drl_system.system.system_manager",
    }
    if name not in mapping:
        raise AttributeError(name)
    module = import_module(mapping[name])
    return getattr(module, name)


__all__ = [
    "PPOAgent",
    "PPOBatch",
    "A3CAgent",
    "SACAgent",
    "DQNAgent",
    "DDPGAgent",
    "TD3Agent",
    "IMPALAAgent",
    "TRPOAgent",
    "create_agent",
    "SystemConfig",
    "SyntheticDatasetBuilder",
    "EpisodicMemory",
    "ReplayBuffer",
    "Transition",
    "PreprocessingPipeline",
    "DynamicTokenizer",
    "SequenceEncoder",
    "SequenceDecoder",
    "Trainer",
    "SystemManager",
    "CommandResult",
]
