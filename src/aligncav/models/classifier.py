"""
CNN Mode Classifier for Hermite-Gaussian beam modes.

This module provides CNN architectures for classifying laser beam modes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModeClassifier(nn.Module):
    """
    CNN classifier for Hermite-Gaussian beam mode identification.

    Architecture:
        - 3 convolutional layers with batch normalization
        - Max pooling and dropout for regularization
        - Fully connected layers for classification

    Maps input beam images to (max_mode+1)^2 classes,
    where class = m * (max_mode+1) + n for mode indices (m, n).
    """

    def __init__(
        self,
        num_classes: int = 121,
        input_size: int = 256,
        in_channels: int = 1,
        dropout: float = 0.5,
    ):
        """
        Initialize mode classifier.

        Args:
            num_classes: Number of output classes (default: 121 for modes 0-10)
            input_size: Input image size (assumed square)
            in_channels: Number of input channels (1 for grayscale)
            dropout: Dropout probability
        """
        super().__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.in_channels = in_channels

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        # Calculate size after convolutions
        self._feature_size = self._calculate_feature_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self._feature_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _calculate_feature_size(self) -> int:
        """Calculate the size of features after conv layers."""
        # After 3 pooling operations, size is reduced by 8x
        size = self.input_size // 8
        return 128 * size * size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class and confidence.

        Args:
            x: Input tensor

        Returns:
            Tuple of (predicted_classes, confidences)
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)
        return predictions, confidences

    def predict_mode_indices(
        self, x: torch.Tensor, max_mode: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict mode indices (m, n) from input.

        Args:
            x: Input tensor
            max_mode: Maximum mode index

        Returns:
            Tuple of (m_indices, n_indices, confidences)
        """
        predictions, confidences = self.predict(x)

        # Decode class to (m, n) indices
        m_indices = predictions // (max_mode + 1)
        n_indices = predictions % (max_mode + 1)

        return m_indices, n_indices, confidences


class DeepModeClassifier(nn.Module):
    """
    Deeper CNN classifier with residual connections.

    Suitable for more challenging classification tasks or
    when higher accuracy is required.
    """

    def __init__(
        self,
        num_classes: int = 121,
        input_size: int = 256,
        in_channels: int = 1,
        base_channels: int = 64,
        dropout: float = 0.5,
    ):
        """
        Initialize deep mode classifier.

        Args:
            num_classes: Number of output classes
            input_size: Input image size
            in_channels: Number of input channels
            base_channels: Base number of channels
            dropout: Dropout probability
        """
        super().__init__()

        self.num_classes = num_classes
        self.input_size = input_size

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn_in = nn.BatchNorm2d(base_channels)

        # Residual blocks
        self.layer1 = self._make_layer(base_channels, base_channels, 2)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 1
    ) -> nn.Sequential:
        """Create a residual layer with multiple blocks."""
        layers = []

        # First block may downsample
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = F.max_pool2d(x, 3, stride=2, padding=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class ResidualBlock(nn.Module):
    """Basic residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
