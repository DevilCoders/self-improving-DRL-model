"""High level training orchestrator supporting online/offline modes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import torch

from ..agents.ppo_agent import PPOAgent, PPOBatch
from ..config import SystemConfig
from ..memory.replay_buffer import ReplayBuffer, Transition
from ..safety.safe_filter import SafeActionsFilter
from ..runners.multitask_runner import MultiTaskScheduler
from ..self_improvement.self_evaluator import SelfImprovementLoop
from ..self_improvement.thinking import DeliberationEngine
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
        self.deliberation_engine = DeliberationEngine(config.thinking)
        self.scheduler = MultiTaskScheduler(max_workers=config.concurrency.max_workers)
        self.env_factory = env_factory

    def collect_rollout(self, rollout_length: int) -> list[Transition]:
        transitions = []
        trace = self.deliberation_engine.start_trace("rollout", mode="extensive")
        obs = self.env_factory()
        for _ in range(rollout_length):
            obs_tensor = torch.from_numpy(obs).float()
            action, log_prob, entropy, value = self.agent.model.act(obs_tensor)
            action_np = action.detach().cpu().numpy()
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
                    info={"log_prob": log_prob.item(), "entropy": entropy.mean().item(), "value": value.mean().item()},
                )
            )
            self.deliberation_engine.add_step(
                trace.trace_id,
                f"step reward={reward:.4f} entropy={entropy.mean().item():.4f}",
                done=done,
            )
            obs = next_obs
            if done:
                obs = self.env_factory()
        if self.config.thinking.auto_summarize:
            self.scheduler.schedule(
                self.deliberation_engine.summarize,
                trace.trace_id,
                description="summarize_rollout",
            )
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
        if self.config.concurrency.track_metrics:
            self.scheduler.schedule(
                self.deliberation_engine.record_metrics,
                stats,
                description="record_update_metrics",
            )
        return stats

    def train(self, steps: Optional[int] = None) -> None:
        total_steps = steps or self.config.training.total_steps
        for step in range(0, total_steps, self.config.training.update_interval):
            transitions = self.collect_rollout(self.config.training.rollout_length)
            for transition in transitions:
                self.replay_buffer.push(transition)
            stats = self.update_agent(transitions)
            human_feedback = self.feedback_buffer.collect(limit=10)
            if human_feedback:
                self.integrate_feedback(human_feedback)
            if step % self.config.training.save_interval == 0:
                self.self_improvement.checkpoint(step, stats)
            self.scheduler.gather(timeout=self.config.concurrency.gather_timeout)

    def shutdown(self) -> None:
        self.scheduler.shutdown()


__all__ = ["Trainer", "TrainingArtifacts"]
