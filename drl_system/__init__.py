"""Self-improving DRL system package with lazy imports."""

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:  # pragma: no cover - thin wrapper
    mapping = {
        "PPOAgent": "drl_system.agents.ppo_agent",
        "PPOBatch": "drl_system.agents.ppo_agent",
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
        "DeliberationEngine": "drl_system.self_improvement.thinking",
        "DeliberationTrace": "drl_system.self_improvement.thinking",
        "DeliberationStep": "drl_system.self_improvement.thinking",
        "WebScraper": "drl_system.integration.web_tools",
        "WebSessionManager": "drl_system.integration.web_tools",
        "MultiTaskScheduler": "drl_system.runners.multitask_runner",
        "ScheduledResult": "drl_system.runners.multitask_runner",
    }
    if name not in mapping:
        raise AttributeError(name)
    module = import_module(mapping[name])
    return getattr(module, name)


__all__ = [
    "PPOAgent",
    "PPOBatch",
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
    "DeliberationEngine",
    "DeliberationTrace",
    "DeliberationStep",
    "WebScraper",
    "WebSessionManager",
    "MultiTaskScheduler",
    "ScheduledResult",
]
