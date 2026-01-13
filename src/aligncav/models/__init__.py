"""
Neural network models for beam mode classification and alignment.

This module provides CNN classifiers, DQN agents, and feature extractors
for the cavity alignment system.
"""

from .classifier import DeepModeClassifier, ModeClassifier
from .dqn import (
    DQN,
    DQNAgent,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    Transition,
    decode_action,
    encode_action,
)
from .feature_extractor import (
    AttentionFeatureExtractor,
    CustomFeatureExtractor,
    ResNetFeatureExtractor,
)
from .modal_decomposition import (
    HybridModalClassifier,
    ModalDecompositionAttention,
    ModalDecompositionCNN,
)

__all__ = [
    # Classifiers
    "ModeClassifier",
    "DeepModeClassifier",
    # DQN
    "DQN",
    "DQNAgent",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "Transition",
    "decode_action",
    "encode_action",
    # Feature extractors
    "ResNetFeatureExtractor",
    "CustomFeatureExtractor",
    "AttentionFeatureExtractor",
    # Modal decomposition
    "ModalDecompositionCNN",
    "ModalDecompositionAttention",
    "HybridModalClassifier",
]
