"""Configuration dataclasses for the DRL system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DatasetSpec:
    """Describes a concrete dataset artefact to materialise."""

    name: str
    category: str
    modality: str
    formats: List[str]
    samples: int = 128
    description: str = ""
    tags: Optional[List[str]] = None


def default_dataset_specs() -> List["DatasetSpec"]:
    return [
        DatasetSpec(
            name="terminal_commands",
            category="terminal_commands",
            modality="text",
            formats=["csv", "tsv", "txt", "json", "jsonl"],
            samples=160,
            description="Cross-platform terminal automation snippets covering admin and standard user flows.",
            tags=["linux", "windows", "automation"],
        ),
        DatasetSpec(
            name="ethical_hacking_commands",
            category="security_commands",
            modality="text",
            formats=["csv", "tsv", "txt", "json", "jsonl"],
            samples=96,
            description="Safety-filtered ethical hacking and penetration testing command references.",
            tags=["cybersecurity", "penetration-testing", "safe"],
        ),
        DatasetSpec(
            name="stable_diffusion_prompts",
            category="stable_diffusion",
            modality="image",
            formats=["png", "json"],
            samples=48,
            description="Prompt and latent pairings suitable for seeding diffusion fine-tuning runs.",
            tags=["computer-vision", "generative"],
        ),
        DatasetSpec(
            name="audio_language_corpus",
            category="audio_language",
            modality="audio",
            formats=["wav", "jsonl"],
            samples=40,
            description="Multi-lingual NLP/NLU/NLG audio snippets with paired transcripts.",
            tags=["audio", "speech", "nlp"],
        ),
        DatasetSpec(
            name="technical_pdfs",
            category="pdf_knowledge",
            modality="document",
            formats=["pdf", "json"],
            samples=24,
            description="Compact PDF primers and metadata for offline policy conditioning.",
            tags=["documentation", "reference"],
        ),
        DatasetSpec(
            name="code_corpus",
            category="code_corpus",
            modality="code",
            formats=["py", "cpp", "js", "json"],
            samples=30,
            description="Polyglot programming snippets for grounded reasoning and tool use.",
            tags=["python", "c++", "javascript", "programming"],
        ),
        DatasetSpec(
            name="robotics_controls",
            category="robotics_controls",
            modality="control",
            formats=["json", "csv"],
            samples=36,
            description="Synthetic robotics trajectories with ROS topic metadata and actuator bounds.",
            tags=["ros", "robotics", "safety"],
        ),
    ]


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
class AgentConfig:
    """Configuration for selecting and tuning the control agent."""

    type: str = "ppo"
    hidden_sizes: List[int] = field(default_factory=default_network_hidden_sizes)
    transformer_layers: int = 1
    hierarchy_levels: int = 2
    ensemble: List[str] = field(
        default_factory=lambda: [
            "ppo",
            "a3c",
            "sac",
            "dqn",
            "ddpg",
            "td3",
            "reinforce",
            "quantile_dqn",
        ]
    )
    sync_factor: float = 0.9
    n_step: int = 5
    temperature: float = 1.0
    sac_alpha: float = 0.2
    soft_update_tau: float = 0.01
    double_q: bool = True
    ddpg_noise: float = 0.2
    td3_noise: float = 0.1
    policy_delay: int = 2
    quantile_atoms: int = 32
    reinforce_baseline: bool = True


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
    modes: List[str] = field(
        default_factory=lambda: [
            "offline",
            "online",
            "parallel",
            "distributed",
            "curriculum",
            "evaluation",
        ]
    )
    parallel_workers: int = 4
    curriculum_stages: int = 3


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
    chunk_size: int = 1024
    chunk_overlap: int = 128
    datasets: List[DatasetSpec] = field(default_factory=default_dataset_specs)


@dataclass
class SystemManagementConfig:
    enable_linux_admin: bool = True
    enable_linux_standard: bool = True
    enable_windows_admin: bool = True
    enable_windows_standard: bool = True
    audit_commands: bool = True
    dry_run: bool = False


@dataclass
class SystemConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    meta_learning: MetaLearningConfig = field(default_factory=MetaLearningConfig)
    safe_filter: SafeFilterConfig = field(default_factory=SafeFilterConfig)
    rlhf: RLHFConfig = field(default_factory=RLHFConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    system_management: SystemManagementConfig = field(default_factory=SystemManagementConfig)


__all__ = [
    "AgentConfig",
    "MemoryConfig",
    "PPOConfig",
    "TrainingConfig",
    "TokenizerConfig",
    "MetaLearningConfig",
    "SafeFilterConfig",
    "RLHFConfig",
    "DatasetSpec",
    "DatasetConfig",
    "SystemManagementConfig",
    "SystemConfig",
]
