"""Configuration dataclasses for the DRL system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


def default_network_hidden_sizes() -> List[int]:
    return [256, 256]


def default_optimizer_config() -> Dict[str, float]:
    return {"lr": 3e-4, "betas": (0.9, 0.999), "eps": 1e-8}


@dataclass
class MemoryConfig:
    capacity: int = 100_000
    batch_size: int = 512
    prioritized: bool = False
    gamma: float = 0.99


@dataclass
class PPOConfig:
    clip_range: float = 0.2
    epochs: int = 4
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class TrainingConfig:
    total_steps: int = 1_000_000
    rollout_length: int = 2048
    update_interval: int = 2048
    save_interval: int = 10_000
    eval_interval: int = 50_000
    device: str = "auto"
    num_envs: int = 8
    distributed: bool = True


@dataclass
class TokenizerConfig:
    vocab_size: int = 4096
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"


@dataclass
class MetaLearningConfig:
    adaptation_steps: int = 5
    meta_lr: float = 1e-4
    base_optimizer: Dict[str, float] = field(default_factory=default_optimizer_config)


@dataclass
class SafeFilterConfig:
    min_action: float = -1.0
    max_action: float = 1.0
    forbidden_zones: Optional[List[tuple]] = None


@dataclass
class RLHFConfig:
    reward_scale: float = 1.0
    human_buffer_size: int = 10_000
    aggregation: str = "mean"


@dataclass
class DatasetConfig:
    root: str = "data/generated"
    version: str = "v1"
    features: Optional[List[str]] = None


@dataclass
class ThinkingConfig:
    enable: bool = True
    default_mode: str = "extensive"
    modes: List[str] = field(
        default_factory=lambda: ["thinking", "extensive", "deep", "extended"]
    )
    max_steps: int = 128
    auto_summarize: bool = True


@dataclass
class WebToolsConfig:
    enable_browsing: bool = True
    enable_scraping: bool = True
    allowed_domains: Optional[List[str]] = None
    user_agent: str = "SelfImprovingDRL/1.0"
    max_concurrent_requests: int = 4


@dataclass
class ConcurrencyConfig:
    max_workers: int = 4
    gather_timeout: float = 0.0
    track_metrics: bool = True


@dataclass
class SystemConfig:
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    meta_learning: MetaLearningConfig = field(default_factory=MetaLearningConfig)
    safe_filter: SafeFilterConfig = field(default_factory=SafeFilterConfig)
    rlhf: RLHFConfig = field(default_factory=RLHFConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)
    web_tools: WebToolsConfig = field(default_factory=WebToolsConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)


__all__ = [
    "MemoryConfig",
    "PPOConfig",
    "TrainingConfig",
    "TokenizerConfig",
    "MetaLearningConfig",
    "SafeFilterConfig",
    "RLHFConfig",
    "DatasetConfig",
    "ThinkingConfig",
    "WebToolsConfig",
    "ConcurrencyConfig",
    "SystemConfig",
]
