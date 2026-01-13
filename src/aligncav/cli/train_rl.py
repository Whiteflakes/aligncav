"""
CLI command for training the RL alignment agent.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL agent for cavity alignment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training arguments
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1000,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training",
    )

    # DQN arguments
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Initial exploration rate",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.01,
        help="Final exploration rate",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=int,
        default=10000,
        help="Steps for epsilon decay",
    )
    parser.add_argument(
        "--target-update",
        type=int,
        default=1000,
        help="Steps between target network updates",
    )

    # Environment arguments
    parser.add_argument(
        "--target-quality",
        type=float,
        default=0.95,
        help="Target alignment quality",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Input image size",
    )

    # Replay buffer
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=100000,
        help="Replay buffer size",
    )
    parser.add_argument(
        "--min-buffer-size",
        type=int,
        default=1000,
        help="Minimum buffer size before training",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="rl_agent",
        help="Experiment name",
    )

    # Evaluation
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=50,
        help="Episodes between evaluations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )

    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Starting RL training")

    try:
        from ..models import DQNAgent
        from ..simulation import CavityConfig, CavityEnvironment
        from ..training import RLTrainer, RLTrainingConfig

        # Create environment
        cavity_config = CavityConfig(image_size=args.image_size)
        env = CavityEnvironment(
            config=cavity_config,
            max_steps=args.max_steps,
            target_quality=args.target_quality,
        )

        logger.info(f"Environment: {env.num_actions} actions")

        # Create agent
        agent = DQNAgent(
            input_size=args.image_size,
            num_actions=env.num_actions,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            target_update=args.target_update,
            device=args.device,
        )

        logger.info(f"Agent device: {agent.device}")

        # Create trainer
        config = RLTrainingConfig(
            num_episodes=args.num_episodes,
            max_steps_per_episode=args.max_steps,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            min_buffer_size=args.min_buffer_size,
            eval_frequency=args.eval_frequency,
            eval_episodes=args.eval_episodes,
            checkpoint_dir=args.output_dir,
            target_quality=args.target_quality,
        )

        trainer = RLTrainer(agent=agent, env=env, config=config)

        # Train
        history = trainer.train()

        # Log final results
        logger.info("Training complete!")
        logger.info(f"Final success rate: {100 * trainer.episode_successes[-100:].count(True) / 100:.1f}%")

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
