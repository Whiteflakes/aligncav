"""
Metrics for Training Evaluation.

This module provides metrics for evaluating model performance
during training and testing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class Metric(ABC):
    """Base class for metrics."""

    @abstractmethod
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric with new predictions."""
        pass

    @abstractmethod
    def compute(self) -> float:
        """Compute the metric value."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the metric state."""
        pass


class AccuracyMetric(Metric):
    """Standard classification accuracy."""

    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update with batch predictions."""
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=1)
        self.correct += (predictions == targets).sum().item()
        self.total += targets.size(0)

    def compute(self) -> float:
        """Compute accuracy."""
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def reset(self) -> None:
        """Reset state."""
        self.correct = 0
        self.total = 0


class TopKAccuracyMetric(Metric):
    """Top-K classification accuracy."""

    def __init__(self, k: int = 5):
        self.k = k
        self.correct = 0
        self.total = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update with batch predictions."""
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)

        _, top_k_preds = predictions.topk(self.k, dim=1)
        targets = targets.view(-1, 1).expand_as(top_k_preds)
        self.correct += (top_k_preds == targets).any(dim=1).sum().item()
        self.total += targets.size(0)

    def compute(self) -> float:
        """Compute top-k accuracy."""
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def reset(self) -> None:
        """Reset state."""
        self.correct = 0
        self.total = 0


class ModeDistanceMetric(Metric):
    """
    Distance-based metric for mode classification.

    Considers the distance between predicted and true mode indices,
    not just exact matches.
    """

    def __init__(self, max_mode: int = 10):
        self.max_mode = max_mode
        self.total_distance = 0.0
        self.count = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update with batch predictions."""
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=1)

        # Decode to (m, n) indices
        pred_m = predictions // (self.max_mode + 1)
        pred_n = predictions % (self.max_mode + 1)
        true_m = targets // (self.max_mode + 1)
        true_n = targets % (self.max_mode + 1)

        # Manhattan distance in mode space
        distance = torch.abs(pred_m - true_m) + torch.abs(pred_n - true_n)
        self.total_distance += distance.sum().item()
        self.count += targets.size(0)

    def compute(self) -> float:
        """Compute average mode distance."""
        if self.count == 0:
            return 0.0
        return self.total_distance / self.count

    def reset(self) -> None:
        """Reset state."""
        self.total_distance = 0.0
        self.count = 0


class ConfusionMatrix:
    """
    Confusion matrix for classification.

    Tracks true positives, false positives, etc. for each class.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update confusion matrix."""
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=1)

        preds = predictions.cpu().numpy()
        targs = targets.cpu().numpy()

        for p, t in zip(preds, targs):
            if 0 <= p < self.num_classes and 0 <= t < self.num_classes:
                self.matrix[t, p] += 1

    def compute_precision(self, class_idx: Optional[int] = None) -> np.ndarray | float:
        """Compute precision per class or for specific class."""
        # Precision = TP / (TP + FP)
        tp = np.diag(self.matrix)
        fp = self.matrix.sum(axis=0) - tp

        precision = np.divide(tp, tp + fp, where=(tp + fp) > 0)

        if class_idx is not None:
            return float(precision[class_idx])
        return precision

    def compute_recall(self, class_idx: Optional[int] = None) -> np.ndarray | float:
        """Compute recall per class or for specific class."""
        # Recall = TP / (TP + FN)
        tp = np.diag(self.matrix)
        fn = self.matrix.sum(axis=1) - tp

        recall = np.divide(tp, tp + fn, where=(tp + fn) > 0)

        if class_idx is not None:
            return float(recall[class_idx])
        return recall

    def compute_f1(self, class_idx: Optional[int] = None) -> np.ndarray | float:
        """Compute F1 score per class or for specific class."""
        precision = self.compute_precision()
        recall = self.compute_recall()

        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            where=(precision + recall) > 0,
        )

        if class_idx is not None:
            return float(f1[class_idx])
        return f1

    def reset(self) -> None:
        """Reset confusion matrix."""
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)


class RewardTracker:
    """
    Tracks rewards during RL training.

    Computes running averages and statistics.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.successes: List[bool] = []

    def update(
        self,
        reward: float,
        episode_length: int,
        success: bool = False,
    ) -> None:
        """Update with episode results."""
        self.rewards.append(reward)
        self.episode_lengths.append(episode_length)
        self.successes.append(success)

    def get_mean_reward(self, window: Optional[int] = None) -> float:
        """Get mean reward over recent episodes."""
        window = window or self.window_size
        recent = self.rewards[-window:]
        return float(np.mean(recent)) if recent else 0.0

    def get_success_rate(self, window: Optional[int] = None) -> float:
        """Get success rate over recent episodes."""
        window = window or self.window_size
        recent = self.successes[-window:]
        return float(np.mean(recent)) if recent else 0.0

    def get_mean_length(self, window: Optional[int] = None) -> float:
        """Get mean episode length."""
        window = window or self.window_size
        recent = self.episode_lengths[-window:]
        return float(np.mean(recent)) if recent else 0.0

    def get_stats(self) -> Dict[str, float]:
        """Get all statistics."""
        return {
            "mean_reward": self.get_mean_reward(),
            "success_rate": self.get_success_rate(),
            "mean_length": self.get_mean_length(),
            "total_episodes": len(self.rewards),
        }

    def reset(self) -> None:
        """Reset tracker."""
        self.rewards.clear()
        self.episode_lengths.clear()
        self.successes.clear()


class MetricCollection:
    """
    Collection of metrics for convenient tracking.

    Allows updating and computing multiple metrics at once.
    """

    def __init__(self, metrics: Dict[str, Metric]):
        """
        Initialize metric collection.

        Args:
            metrics: Dictionary of metric name -> Metric instance
        """
        self.metrics = metrics

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update all metrics."""
        for metric in self.metrics.values():
            metric.update(predictions, targets)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()
