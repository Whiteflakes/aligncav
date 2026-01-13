"""
Reinforcement Learning Training Pipeline.

This module provides training utilities for the DQN-based
cavity alignment agent.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from ..models.dqn import DQNAgent
from ..simulation.cavity import CavityEnvironment

logger = logging.getLogger(__name__)


@dataclass
class RLTrainingConfig:
    """Configuration for RL training."""

    # Training parameters
    num_episodes: int = 1000
    max_steps_per_episode: int = 100
    batch_size: int = 64

    # DQN parameters
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10000
    target_update: int = 1000

    # Replay buffer
    buffer_size: int = 100000
    min_buffer_size: int = 1000

    # Evaluation
    eval_frequency: int = 50
    eval_episodes: int = 10

    # Logging and saving
    log_frequency: int = 10
    save_frequency: int = 100
    checkpoint_dir: str = "checkpoints"

    # Success criterion
    target_quality: float = 0.95


class RLTrainer:
    """
    Trainer for DQN-based cavity alignment agent.

    Handles the complete RL training pipeline including:
    - Episode rollouts
    - Experience collection
    - Network updates
    - Evaluation
    - Logging
    """

    def __init__(
        self,
        agent: Optional[DQNAgent] = None,
        env: Optional[CavityEnvironment] = None,
        config: Optional[RLTrainingConfig] = None,
    ):
        """
        Initialize RL trainer.

        Args:
            agent: DQN agent (created if not provided)
            env: Cavity environment (created if not provided)
            config: Training configuration
        """
        self.config = config or RLTrainingConfig()

        # Environment
        self.env = env or CavityEnvironment(
            max_steps=self.config.max_steps_per_episode,
            target_quality=self.config.target_quality,
        )

        # Agent
        self.agent = agent or DQNAgent(
            num_actions=self.env.num_actions,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            epsilon_start=self.config.epsilon_start,
            epsilon_end=self.config.epsilon_end,
            epsilon_decay=self.config.epsilon_decay,
            target_update=self.config.target_update,
        )

        # Training statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_successes: List[bool] = []
        self.eval_rewards: List[float] = []
        self.losses: List[float] = []

    def train_episode(self) -> Dict[str, Any]:
        """
        Train for one episode.

        Returns:
            Dictionary with episode statistics
        """
        state = self.env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        num_updates = 0

        for step in range(self.config.max_steps_per_episode):
            # Select action
            action = self.agent.select_action(state, training=True)

            # Environment step
            next_state, reward, done, info = self.env.step(action)

            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)

            # Update statistics
            episode_reward += reward
            state = next_state

            # Training step
            if len(self.agent.memory) >= self.config.min_buffer_size:
                loss = self.agent.train_step(self.config.batch_size)
                if loss is not None:
                    episode_loss += loss
                    num_updates += 1

            if done:
                break

        avg_loss = episode_loss / max(num_updates, 1)

        return {
            "reward": episode_reward,
            "length": step + 1,
            "success": info.get("aligned", False),
            "final_quality": info.get("alignment_quality", 0.0),
            "loss": avg_loss,
            "epsilon": self.agent.epsilon,
        }

    @torch.no_grad()
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate the agent.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Dictionary with evaluation statistics
        """
        rewards = []
        lengths = []
        successes = []
        final_qualities = []

        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0.0

            for step in range(self.config.max_steps_per_episode):
                action = self.agent.select_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                state = next_state

                if done:
                    break

            rewards.append(episode_reward)
            lengths.append(step + 1)
            successes.append(info.get("aligned", False))
            final_qualities.append(info.get("alignment_quality", 0.0))

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths),
            "success_rate": np.mean(successes),
            "mean_quality": np.mean(final_qualities),
        }

    def train(
        self,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, List]:
        """
        Train the agent.

        Args:
            callbacks: List of callback functions

        Returns:
            Training history
        """
        logger.info("Starting RL training")
        logger.info(f"Environment: {self.env.num_actions} actions")
        logger.info(f"Agent device: {self.agent.device}")

        # Create checkpoint directory
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_eval_reward = float("-inf")

        for episode in tqdm(range(self.config.num_episodes), desc="Training"):
            # Training episode
            stats = self.train_episode()

            self.episode_rewards.append(stats["reward"])
            self.episode_lengths.append(stats["length"])
            self.episode_successes.append(stats["success"])
            self.losses.append(stats["loss"])

            # Logging
            if (episode + 1) % self.config.log_frequency == 0:
                recent_rewards = self.episode_rewards[-self.config.log_frequency :]
                recent_successes = self.episode_successes[-self.config.log_frequency :]
                logger.info(
                    f"Episode {episode + 1}/{self.config.num_episodes} - "
                    f"Reward: {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}, "
                    f"Success: {100 * np.mean(recent_successes):.1f}%, "
                    f"Epsilon: {stats['epsilon']:.3f}"
                )

            # Evaluation
            if (episode + 1) % self.config.eval_frequency == 0:
                eval_stats = self.evaluate(self.config.eval_episodes)
                self.eval_rewards.append(eval_stats["mean_reward"])

                logger.info(
                    f"Evaluation - "
                    f"Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}, "
                    f"Success: {100 * eval_stats['success_rate']:.1f}%, "
                    f"Quality: {eval_stats['mean_quality']:.3f}"
                )

                # Save best model
                if eval_stats["mean_reward"] > best_eval_reward:
                    best_eval_reward = eval_stats["mean_reward"]
                    self.agent.save(str(checkpoint_dir / "best_agent.pt"))
                    logger.info(f"Saved best agent with reward: {best_eval_reward:.2f}")

            # Regular checkpointing
            if (episode + 1) % self.config.save_frequency == 0:
                self.agent.save(str(checkpoint_dir / f"agent_episode_{episode + 1}.pt"))

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, episode, stats)

        # Save final model
        self.agent.save(str(checkpoint_dir / "final_agent.pt"))

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_successes": self.episode_successes,
            "eval_rewards": self.eval_rewards,
            "losses": self.losses,
        }

    def load_checkpoint(self, path: str | Path) -> None:
        """Load agent from checkpoint."""
        self.agent.load(str(path))
        logger.info(f"Loaded agent from {path}")


def train_rl_agent(
    num_episodes: int = 1000,
    target_quality: float = 0.95,
    checkpoint_dir: str = "checkpoints",
) -> DQNAgent:
    """
    Convenience function to train an RL agent.

    Args:
        num_episodes: Number of training episodes
        target_quality: Target alignment quality
        checkpoint_dir: Directory for saving checkpoints

    Returns:
        Trained DQN agent
    """
    config = RLTrainingConfig(
        num_episodes=num_episodes,
        target_quality=target_quality,
        checkpoint_dir=checkpoint_dir,
    )

    trainer = RLTrainer(config=config)
    trainer.train()

    return trainer.agent
