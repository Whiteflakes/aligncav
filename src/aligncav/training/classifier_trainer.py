"""
Classifier Training Pipeline.

This module provides training utilities for the mode classifier CNN.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for classifier training."""

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # "step", "cosine", "plateau"
    lr_step_size: int = 30
    lr_gamma: float = 0.1

    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4

    # Checkpointing
    save_best: bool = True
    checkpoint_dir: str = "checkpoints"

    # Device
    device: Optional[str] = None


class ClassifierTrainer:
    """
    Trainer for mode classifier networks.

    Handles the complete training pipeline including:
    - Training and validation loops
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Logging
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        criterion: Optional[nn.Module] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Neural network model to train
            config: Training configuration
            criterion: Loss function (default: CrossEntropyLoss)
        """
        self.config = config or TrainingConfig()
        self.model = model

        # Device setup
        if self.config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        self.model = self.model.to(self.device)

        # Loss function
        self.criterion = criterion or nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rate": [],
        }

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on config."""
        if self.config.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
            )
        elif self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
            )
        elif self.config.lr_scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.lr_gamma,
                patience=self.config.patience // 2,
            )
        return None

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            callbacks: List of callback functions

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Create checkpoint directory
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
            else:
                val_loss, val_acc = train_loss, train_acc

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Logging
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )

            # Checkpointing
            if self.config.save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(checkpoint_dir / "best_model.pt")
                logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.config.early_stopping:
                if self.epochs_without_improvement >= self.config.patience:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    break

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch, train_loss, val_loss)

        return self.history

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "epoch": self.current_epoch,
                "best_val_loss": self.best_val_loss,
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    device: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Convenience function to train a classifier.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to use

    Returns:
        Training history
    """
    config = TrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )
    trainer = ClassifierTrainer(model, config)
    return trainer.fit(train_loader, val_loader)
