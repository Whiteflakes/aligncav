"""
Training module for AlignCav.

Provides training pipelines, loss functions, and metrics for
classifier and RL training.
"""

from .classifier_trainer import ClassifierTrainer, TrainingConfig, train_classifier
from .losses import (
    CombinedLoss,
    FocalLoss,
    HuberLoss,
    LabelSmoothingLoss,
    ModalDecompositionLoss,
    RewardShapingLoss,
)
from .metrics import (
    AccuracyMetric,
    ConfusionMatrix,
    MetricCollection,
    ModeDistanceMetric,
    RewardTracker,
    TopKAccuracyMetric,
)
from .rl_trainer import RLTrainer, RLTrainingConfig, train_rl_agent

__all__ = [
    # Classifier training
    "ClassifierTrainer",
    "TrainingConfig",
    "train_classifier",
    # RL training
    "RLTrainer",
    "RLTrainingConfig",
    "train_rl_agent",
    # Losses
    "FocalLoss",
    "LabelSmoothingLoss",
    "HuberLoss",
    "ModalDecompositionLoss",
    "RewardShapingLoss",
    "CombinedLoss",
    # Metrics
    "AccuracyMetric",
    "TopKAccuracyMetric",
    "ModeDistanceMetric",
    "ConfusionMatrix",
    "RewardTracker",
    "MetricCollection",
]
