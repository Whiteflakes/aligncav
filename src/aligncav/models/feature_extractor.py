"""
Feature Extraction Networks.

This module provides pre-trained and custom feature extractors
for beam image analysis.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ResNetFeatureExtractor(nn.Module):
    """
    Feature extractor based on ResNet architecture.

    Can use pre-trained ImageNet weights or train from scratch.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: int = 512,
    ):
        """
        Initialize ResNet feature extractor.

        Args:
            pretrained: Whether to use pre-trained ImageNet weights
            freeze_backbone: Whether to freeze backbone weights
            output_dim: Dimension of output features
        """
        super().__init__()

        self.output_dim = output_dim

        # Import here to avoid issues if torchvision not installed
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError:
            raise ImportError("torchvision is required for ResNetFeatureExtractor")

        # Load ResNet18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)

        # Modify first conv for single-channel input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Copy weights from RGB to grayscale (average)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(resnet.conv1.weight.mean(dim=1, keepdim=True))

        # Extract backbone layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Projection head
        self.projection = nn.Linear(512, output_dim)

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for name, param in self.named_parameters():
            if "projection" not in name:
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input.

        Args:
            x: Input tensor of shape (batch, 1, height, width)

        Returns:
            Feature tensor of shape (batch, output_dim)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)

        return x


class CustomFeatureExtractor(nn.Module):
    """
    Custom lightweight feature extractor.

    Designed specifically for beam images, without pre-training.
    """

    def __init__(
        self,
        input_size: int = 256,
        output_dim: int = 256,
    ):
        """
        Initialize custom feature extractor.

        Args:
            input_size: Input image size
            output_dim: Dimension of output features
        """
        super().__init__()

        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            # Block 1: 256 -> 128
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Block 2: 128 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Block 3: 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Block 4: 32 -> 16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Block 5: 16 -> 8
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features."""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return x


class AttentionFeatureExtractor(nn.Module):
    """
    Feature extractor with spatial attention.

    Focuses on relevant parts of the beam image.
    """

    def __init__(
        self,
        input_size: int = 256,
        output_dim: int = 256,
    ):
        """
        Initialize attention feature extractor.

        Args:
            input_size: Input image size
            output_dim: Output feature dimension
        """
        super().__init__()

        self.output_dim = output_dim

        # Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Spatial attention
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

        # Feature processing
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(128 * 16, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features with attention."""
        features = self.backbone(x)

        # Compute attention map
        attention = self.attention(features)

        # Apply attention
        attended = features * attention

        # Pool and project
        pooled = self.pool(attended)
        flat = pooled.view(pooled.size(0), -1)
        output = self.fc(flat)

        return output

    def get_attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention map for visualization.

        Args:
            x: Input tensor

        Returns:
            Attention map tensor
        """
        features = self.backbone(x)
        attention = self.attention(features)
        return attention
