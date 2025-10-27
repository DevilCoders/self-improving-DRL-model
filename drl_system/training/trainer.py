"""High level training orchestrator supporting online/offline modes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch

from ..agents.ppo_agent import PPOAgent, PPOBatch
from ..config import SystemConfig
from ..data.dataset_builder import SyntheticDatasetBuilder
from ..memory.replay_buffer import ReplayBuffer, Transition
from ..safety.safe_filter import SafeActionsFilter
from ..self_improvement.self_evaluator import SelfImprovementLoop
from ..training.rlhf import FeedbackBuffer, HumanFeedback, aggregate_feedback
from ..utils.device import DeviceManager


@dataclass
class TrainingArtifacts:
    trainer: "Trainer"
    agent: PPOAgent
    replay_buffer: ReplayBuffer
    feedback_buffer: FeedbackBuffer
    safe_filter: SafeActionsFilter


class Trainer:
    def __init__(self, config: SystemConfig, env_factory: Callable[[], np.ndarray]) -> None:
        self.config = config
        self.device_manager = DeviceManager(config.training.device)
        device = self.device_manager.select()
        self.replay_buffer = ReplayBuffer(config.memory.capacity, config.memory.prioritized)
        self.agent = PPOAgent(
            obs_dim=env_factory().shape[0],
            action_dim=env_factory().shape[0],
            memory_config=config.memory,
            ppo_config=config.ppo,
            device=device,
        )
        self.safe_filter = SafeActionsFilter(config.safe_filter)
        self.feedback_buffer = FeedbackBuffer(config.rlhf)
        self.self_improvement = SelfImprovementLoop(config.meta_learning, self.agent.optimizer)
        self.env_factory = env_factory
        self.curriculum_stage = 0
        self._maybe_seed_offline_data()

    def _maybe_seed_offline_data(self) -> None:
        dataset_root = Path(self.config.dataset.root) / self.config.dataset.version / "chunks"
        if not dataset_root.exists():
            return
        builder = SyntheticDatasetBuilder(self.config.dataset)
        for obs_chunk, action_chunk, reward_chunk in builder.iter_chunks():
            for obs, action, reward in zip(obs_chunk, action_chunk, reward_chunk):
                action_array = np.atleast_1d(action.astype(np.float32))
                transition = Transition(
                    state=obs,
                    action=action_array,
                    reward=float(reward),
                    next_state=obs,
                    done=False,
                    info={
                        "log_prob": 0.0,
                        "entropy": 0.0,
                        "value": 0.0,
                        "source": "offline",
                    },
                )
                self.replay_buffer.push(transition)

    def collect_rollout(self, rollout_length: int, mode: str = "online") -> list[Transition]:
        if mode == "offline":
            return self._collect_rollout_offline(rollout_length)
        if mode == "parallel":
            return self._collect_rollout_parallel(rollout_length)
        if mode == "distributed":
            return self._collect_rollout_parallel(rollout_length, distributed=True)
        return self._collect_rollout_single(rollout_length)

    def _collect_rollout_offline(self, rollout_length: int) -> list[Transition]:
        if len(self.replay_buffer.buffer) >= rollout_length:
            indices = np.random.choice(len(self.replay_buffer.buffer), rollout_length, replace=False)
            return [self.replay_buffer.buffer[i] for i in indices]
        return self._collect_rollout_single(rollout_length)

    def _collect_rollout_single(self, rollout_length: int) -> list[Transition]:
        transitions = []
        obs = self.env_factory()
        for _ in range(rollout_length):
            obs_tensor = torch.from_numpy(obs).float().to(self.agent.device)
            action, log_prob, entropy, value, extras = self.agent.model.act(obs_tensor)
            action_np = np.atleast_1d(action.detach().cpu().numpy()).astype(np.float32)
            filtered_action = self.safe_filter.filter(action_np)
            next_obs = self.env_factory()  # placeholder for real environment interaction
            reward = float(np.random.randn())
            done = bool(np.random.rand() < 0.05)
            transitions.append(
                Transition(
                    state=obs,
                    action=filtered_action,
                    reward=reward,
                    next_state=next_obs,
                    done=done,
                    info={
                        "log_prob": float(log_prob.item()),
                        "entropy": float(entropy.mean().item()),
                        "value": float(value.mean().item()),
                        "uncertainty": float(extras["uncertainty"].mean().item()),
                        "skill_activation": float(extras["skills"].mean().item()),
                        "world_prediction_error": float(
                            np.abs(extras["world_prediction"].detach().cpu().numpy() - obs).mean()
                        ),
                        "evolution": float(extras["evolution"].mean().item()),
                    },
                )
            )
            obs = next_obs
            if done:
                obs = self.env_factory()
        return transitions

    def _collect_rollout_parallel(self, rollout_length: int, distributed: bool = False) -> list[Transition]:
        transitions: list[Transition] = []
        workers = max(1, self.config.training.parallel_workers)
        observations = [self.env_factory() for _ in range(workers)]
        for _ in range(rollout_length):
            obs_tensor = torch.from_numpy(np.stack(observations)).float().to(self.agent.device)
            (
                logits,
                values,
                advantage_logits,
                uncertainty,
                diagnostics,
            ) = self.agent.model(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()
            for idx in range(workers):
                action_np = np.atleast_1d(actions[idx].detach().cpu().numpy()).astype(np.float32)
                filtered_action = self.safe_filter.filter(action_np)
                next_obs = self.env_factory()
                reward = float(np.random.randn())
                done = bool(np.random.rand() < 0.05)
                transitions.append(
                    Transition(
                        state=observations[idx],
                        action=filtered_action,
                        reward=reward,
                        next_state=next_obs,
                        done=done,
                        info={
                            "log_prob": float(log_probs[idx].item()),
                        "entropy": float(entropies[idx].item()),
                        "value": float(values[idx].mean().item()),
                        "uncertainty": float(uncertainty[idx].mean().item()),
                        "skill_activation": float(diagnostics["skills"][idx].mean().item()),
                        "world_prediction_error": float(
                            torch.abs(
                                diagnostics["world_prediction"][idx].detach().cpu()
                                - torch.from_numpy(observations[idx]).float()
                            )
                            .mean()
                            .item()
                        ),
                        "evolution": float(diagnostics["evolution"][idx].mean().item()),
                        "mode": "distributed" if distributed else "parallel",
                    },
                )
                )
                observations[idx] = next_obs if not done else self.env_factory()
        return transitions

    def integrate_feedback(self, feedback: Iterable[HumanFeedback]) -> float:
        aggregated = aggregate_feedback(feedback, self.config.rlhf)
        for transition in list(self.replay_buffer.buffer)[-len(feedback) :]:
            transition.reward += aggregated
        return aggregated

    def update_agent(self, transitions: list[Transition]) -> dict:
        advantages, returns = self.agent.compute_advantages(
            transitions, self.config.memory.gamma, self.config.ppo.gae_lambda
        )
        observations = torch.from_numpy(np.stack([t.state for t in transitions])).float()
        actions = torch.from_numpy(np.stack([t.action for t in transitions])).float()
        old_log_probs = torch.tensor([t.info["log_prob"] for t in transitions]).float()
        batch = PPOBatch(
            observations=observations,
            actions=actions,
            old_log_probs=old_log_probs,
            returns=returns,
            advantages=advantages,
        )
        stats = self.agent.update(batch)
        self.self_improvement.step(stats)
        return stats

    def train(self, steps: Optional[int] = None, mode: Optional[str] = None) -> None:
        modes = [mode] if mode else list(self.config.training.modes)
        for active_mode in modes:
            if active_mode == "evaluation":
                self.evaluate()
                continue
            total_steps = steps or self.config.training.total_steps
            for step in range(0, total_steps, self.config.training.update_interval):
                if active_mode == "curriculum" and step > 0 and step % (
                    self.config.training.update_interval * max(1, self.config.training.curriculum_stages)
                ) == 0:
                    self.curriculum_stage = min(
                        self.curriculum_stage + 1, self.config.training.curriculum_stages - 1
                    )
                transitions = self.collect_rollout(
                    self.config.training.rollout_length + self.curriculum_stage * 128,
                    mode=active_mode,
                )
                for transition in transitions:
                    self.replay_buffer.push(transition)
                stats = self.update_agent(transitions)
                human_feedback = self.feedback_buffer.collect(limit=10)
                if human_feedback:
                    self.integrate_feedback(human_feedback)
                if step % self.config.training.save_interval == 0:
                    self.self_improvement.checkpoint(step, stats)

    def evaluate(self) -> dict:
        transitions = self.collect_rollout(self.config.training.rollout_length // 2, mode="parallel")
        rewards = [transition.reward for transition in transitions]
        evaluation = {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "num_transitions": len(transitions),
        }
        self.self_improvement.record_evaluation(evaluation)
        return evaluation


__all__ = ["Trainer", "TrainingArtifacts"]
