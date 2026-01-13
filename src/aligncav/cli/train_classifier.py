"""
CLI command for training the mode classifier.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional


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
        description="Train mode classifier for beam alignment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with training data (uses simulated if not provided)",
    )
    parser.add_argument(
        "--simulated-size",
        type=int,
        default=10000,
        help="Number of simulated samples to generate",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Input image size",
    )
    parser.add_argument(
        "--max-mode",
        type=int,
        default=10,
        help="Maximum mode index",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="standard",
        choices=["standard", "deep"],
        help="Model architecture",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability",
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
        default="classifier",
        help="Experiment name for logging",
    )

    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (auto-detect if not specified)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
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
    logger.info("Starting classifier training")

    try:
        import torch

        from ..data import create_mode_dataloaders
        from ..models import DeepModeClassifier, ModeClassifier
        from ..training import ClassifierTrainer, TrainingConfig

        # Calculate number of classes
        num_classes = (args.max_mode + 1) ** 2
        logger.info(f"Number of classes: {num_classes}")

        # Create model
        if args.model == "standard":
            model = ModeClassifier(
                num_classes=num_classes,
                input_size=args.image_size,
                dropout=args.dropout,
            )
        else:
            model = DeepModeClassifier(
                num_classes=num_classes,
                input_size=args.image_size,
                dropout=args.dropout,
            )

        logger.info(f"Model: {args.model}")
        logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create dataloaders
        use_simulated = args.data_dir is None
        train_loader, val_loader = create_mode_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            max_mode=args.max_mode,
            num_workers=args.num_workers,
            use_simulated=use_simulated,
            simulated_size=args.simulated_size,
        )

        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")

        # Create trainer
        config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            checkpoint_dir=args.output_dir,
            device=args.device,
        )

        trainer = ClassifierTrainer(model, config)

        # Train
        history = trainer.fit(train_loader, val_loader)

        # Log final results
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Final train accuracy: {history['train_acc'][-1]:.4f}")
        logger.info(f"Final val accuracy: {history['val_acc'][-1]:.4f}")

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
