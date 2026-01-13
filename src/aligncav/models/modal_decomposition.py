"""
Modal Decomposition Networks.

This module provides neural network architectures for decomposing
beam images into Hermite-Gaussian mode coefficients.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalDecompositionCNN(nn.Module):
    """
    CNN for modal decomposition of beam images.

    Outputs mode coefficients (amplitudes) for each HG mode,
    rather than discrete class labels.
    """

    def __init__(
        self,
        max_mode: int = 10,
        input_size: int = 256,
        in_channels: int = 1,
    ):
        """
        Initialize modal decomposition network.

        Args:
            max_mode: Maximum mode index (outputs (max_mode+1)^2 coefficients)
            input_size: Input image size
            in_channels: Number of input channels
        """
        super().__init__()

        self.max_mode = max_mode
        self.num_modes = (max_mode + 1) ** 2

        # Encoder
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Coefficient prediction
        self.fc = nn.Sequential(
            nn.Linear(256 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_modes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict mode coefficients.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Mode coefficients of shape (batch, num_modes)
        """
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        coefficients = self.fc(features)

        # Ensure non-negative coefficients (intensities)
        coefficients = F.softplus(coefficients)

        # Normalize to sum to 1
        coefficients = coefficients / (coefficients.sum(dim=1, keepdim=True) + 1e-8)

        return coefficients

    def get_dominant_mode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the dominant mode and its coefficient.

        Args:
            x: Input tensor

        Returns:
            Tuple of (mode_indices, coefficients)
        """
        coefficients = self.forward(x)
        max_coeffs, max_indices = coefficients.max(dim=1)
        return max_indices, max_coeffs

    def decode_mode_index(self, index: int) -> Tuple[int, int]:
        """
        Convert flat index to (m, n) mode indices.

        Args:
            index: Flat mode index

        Returns:
            Tuple of (m, n) indices
        """
        m = index // (self.max_mode + 1)
        n = index % (self.max_mode + 1)
        return m, n


class ModalDecompositionAttention(nn.Module):
    """
    Modal decomposition with attention mechanism.

    Uses attention to focus on relevant spatial features
    for mode coefficient prediction.
    """

    def __init__(
        self,
        max_mode: int = 10,
        input_size: int = 256,
        num_heads: int = 4,
    ):
        """
        Initialize attention-based modal decomposition.

        Args:
            max_mode: Maximum mode index
            input_size: Input image size
            num_heads: Number of attention heads
        """
        super().__init__()

        self.max_mode = max_mode
        self.num_modes = (max_mode + 1) ** 2

        # Spatial encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Self-attention
        self.attention = nn.MultiheadAttention(256, num_heads, batch_first=True)

        # Mode prediction
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_modes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict mode coefficients with attention."""
        # Encode
        features = self.encoder(x)  # (B, 256, H, W)

        # Reshape for attention
        B, C, H, W = features.shape
        features = features.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, 256)

        # Self-attention
        attended, _ = self.attention(features, features, features)

        # Global pooling
        pooled = attended.mean(dim=1)  # (B, 256)

        # Predict coefficients
        coefficients = self.fc(pooled)
        coefficients = F.softplus(coefficients)
        coefficients = coefficients / (coefficients.sum(dim=1, keepdim=True) + 1e-8)

        return coefficients


class HybridModalClassifier(nn.Module):
    """
    Hybrid model that outputs both classification and decomposition.

    Combines discrete mode classification with continuous
    mode coefficient estimation.
    """

    def __init__(
        self,
        max_mode: int = 10,
        input_size: int = 256,
    ):
        """
        Initialize hybrid classifier.

        Args:
            max_mode: Maximum mode index
            input_size: Input image size
        """
        super().__init__()

        self.max_mode = max_mode
        self.num_classes = (max_mode + 1) ** 2

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
        )

        # Decomposition head
        self.decomposer = nn.Sequential(
            nn.Linear(128 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both outputs.

        Args:
            x: Input tensor

        Returns:
            Tuple of (class_logits, mode_coefficients)
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        # Classification
        logits = self.classifier(features)

        # Decomposition
        coefficients = self.decomposer(features)
        coefficients = F.softplus(coefficients)
        coefficients = coefficients / (coefficients.sum(dim=1, keepdim=True) + 1e-8)

        return logits, coefficients

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions.

        Returns:
            Tuple of (predicted_classes, confidences, mode_coefficients)
        """
        logits, coefficients = self.forward(x)
        probs = F.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)
        return predictions, confidences, coefficients
