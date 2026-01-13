"""
Loss Functions for Training.

This module provides custom loss functions for mode classification
and reinforcement learning tasks.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Reduces the relative loss for well-classified examples,
    focusing on hard examples.

    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Class weights tensor
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Model logits of shape (N, C)
            targets: Ground truth labels of shape (N,)

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Cross entropy loss with label smoothing.

    Helps prevent overconfident predictions and improves generalization.
    """

    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        """
        Initialize label smoothing loss.

        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor (0 = no smoothing)
            reduction: Reduction method
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction

        self.confidence = 1.0 - smoothing
        self.smooth_value = smoothing / (num_classes - 1)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.

        Args:
            inputs: Model logits of shape (N, C)
            targets: Ground truth labels of shape (N,)

        Returns:
            Loss value
        """
        log_probs = F.log_softmax(inputs, dim=-1)

        # Create smooth labels
        smooth_targets = torch.full_like(log_probs, self.smooth_value)
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)

        # Compute loss
        loss = -smooth_targets * log_probs
        loss = loss.sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1) for robust regression.

    Less sensitive to outliers than MSE.
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        """
        Initialize Huber loss.

        Args:
            delta: Threshold for switching between L1 and L2
            reduction: Reduction method
        """
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss."""
        return F.smooth_l1_loss(
            inputs, targets, beta=self.delta, reduction=self.reduction
        )


class ModalDecompositionLoss(nn.Module):
    """
    Loss function for modal decomposition training.

    Combines coefficient prediction loss with optional
    reconstruction loss.
    """

    def __init__(
        self,
        reconstruction_weight: float = 0.1,
    ):
        """
        Initialize modal decomposition loss.

        Args:
            reconstruction_weight: Weight for reconstruction loss
        """
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.mse = nn.MSELoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        predicted_coeffs: torch.Tensor,
        target_coeffs: torch.Tensor,
        reconstructed: Optional[torch.Tensor] = None,
        original: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss.

        Args:
            predicted_coeffs: Predicted mode coefficients
            target_coeffs: Target coefficients
            reconstructed: Reconstructed image (optional)
            original: Original image (optional)

        Returns:
            Total loss value
        """
        # KL divergence for coefficient distributions
        log_pred = torch.log(predicted_coeffs + 1e-8)
        coeff_loss = self.kl_div(log_pred, target_coeffs)

        total_loss = coeff_loss

        # Optional reconstruction loss
        if reconstructed is not None and original is not None:
            recon_loss = self.mse(reconstructed, original)
            total_loss = total_loss + self.reconstruction_weight * recon_loss

        return total_loss


class RewardShapingLoss(nn.Module):
    """
    Loss for reward prediction in model-based RL.

    Used for training reward models.
    """

    def __init__(self, scale: float = 1.0):
        """
        Initialize reward shaping loss.

        Args:
            scale: Scale factor for loss
        """
        super().__init__()
        self.scale = scale
        self.mse = nn.MSELoss()

    def forward(
        self,
        predicted_reward: torch.Tensor,
        actual_reward: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reward prediction loss."""
        return self.scale * self.mse(predicted_reward, actual_reward)


class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions with weights.

    Useful for multi-task learning scenarios.
    """

    def __init__(
        self,
        losses: list[tuple[nn.Module, float]],
    ):
        """
        Initialize combined loss.

        Args:
            losses: List of (loss_fn, weight) tuples
        """
        super().__init__()
        self.losses = nn.ModuleList([loss for loss, _ in losses])
        self.weights = [weight for _, weight in losses]

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Compute weighted sum of losses."""
        total = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total = total + weight * loss_fn(*args, **kwargs)
        return total
