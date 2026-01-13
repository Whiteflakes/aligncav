"""
Tests for the training module.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from aligncav.training import (
    AccuracyMetric,
    ClassifierTrainer,
    ConfusionMatrix,
    FocalLoss,
    HuberLoss,
    LabelSmoothingLoss,
    MetricCollection,
    ModeDistanceMetric,
    RewardTracker,
    TopKAccuracyMetric,
    TrainingConfig,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 64, num_classes)
    
    def forward(self, x):
        return self.fc(self.flatten(x))


class TestFocalLoss:
    """Tests for FocalLoss."""

    def test_forward(self):
        """Test focal loss forward pass."""
        loss_fn = FocalLoss(gamma=2.0)
        inputs = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10,))
        
        loss = loss_fn(inputs, targets)
        
        assert loss.shape == ()
        assert loss >= 0

    def test_gamma_effect(self):
        """Test that gamma affects the loss."""
        inputs = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10,))
        
        loss_gamma_0 = FocalLoss(gamma=0.0)(inputs, targets)
        loss_gamma_2 = FocalLoss(gamma=2.0)(inputs, targets)
        
        # With gamma=0, should be similar to cross entropy
        # With gamma>0, should be smaller for easy examples
        assert loss_gamma_0 != loss_gamma_2


class TestLabelSmoothingLoss:
    """Tests for LabelSmoothingLoss."""

    def test_forward(self):
        """Test label smoothing loss forward pass."""
        loss_fn = LabelSmoothingLoss(num_classes=5, smoothing=0.1)
        inputs = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10,))
        
        loss = loss_fn(inputs, targets)
        
        assert loss.shape == ()
        assert loss >= 0

    def test_no_smoothing(self):
        """Test with smoothing=0 matches cross entropy."""
        loss_fn = LabelSmoothingLoss(num_classes=5, smoothing=0.0)
        ce_loss = nn.CrossEntropyLoss()
        
        inputs = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10,))
        
        loss1 = loss_fn(inputs, targets)
        loss2 = ce_loss(inputs, targets)
        
        assert torch.isclose(loss1, loss2, rtol=1e-4)


class TestHuberLoss:
    """Tests for HuberLoss."""

    def test_forward(self):
        """Test Huber loss forward pass."""
        loss_fn = HuberLoss(delta=1.0)
        inputs = torch.randn(10)
        targets = torch.randn(10)
        
        loss = loss_fn(inputs, targets)
        
        assert loss.shape == ()
        assert loss >= 0


class TestAccuracyMetric:
    """Tests for AccuracyMetric."""

    @pytest.fixture
    def metric(self):
        """Create metric fixture."""
        return AccuracyMetric()

    def test_perfect_accuracy(self, metric):
        """Test perfect accuracy."""
        predictions = torch.tensor([0, 1, 2, 3, 4])
        targets = torch.tensor([0, 1, 2, 3, 4])
        
        metric.update(predictions, targets)
        acc = metric.compute()
        
        assert acc == 1.0

    def test_zero_accuracy(self, metric):
        """Test zero accuracy."""
        predictions = torch.tensor([1, 2, 3, 4, 0])
        targets = torch.tensor([0, 1, 2, 3, 4])
        
        metric.update(predictions, targets)
        acc = metric.compute()
        
        assert acc == 0.0

    def test_reset(self, metric):
        """Test metric reset."""
        predictions = torch.tensor([0, 1, 2])
        targets = torch.tensor([0, 1, 2])
        
        metric.update(predictions, targets)
        metric.reset()
        
        assert metric.correct == 0
        assert metric.total == 0


class TestTopKAccuracyMetric:
    """Tests for TopKAccuracyMetric."""

    def test_top5_accuracy(self):
        """Test top-5 accuracy."""
        metric = TopKAccuracyMetric(k=5)
        
        # Predictions where correct class is in top 5
        predictions = torch.randn(10, 10)
        targets = torch.randint(0, 10, (10,))
        
        # Manually ensure correct class is in top 5
        for i, t in enumerate(targets):
            predictions[i, t] = 100.0  # Make correct class score highest
        
        metric.update(predictions, targets)
        acc = metric.compute()
        
        assert acc == 1.0


class TestModeDistanceMetric:
    """Tests for ModeDistanceMetric."""

    def test_zero_distance(self):
        """Test zero distance for perfect predictions."""
        metric = ModeDistanceMetric(max_mode=5)
        
        predictions = torch.tensor([0, 6, 12])  # (0,0), (1,0), (2,0)
        targets = torch.tensor([0, 6, 12])
        
        metric.update(predictions, targets)
        distance = metric.compute()
        
        assert distance == 0.0

    def test_nonzero_distance(self):
        """Test nonzero distance."""
        metric = ModeDistanceMetric(max_mode=5)
        
        # Predict (1,0) instead of (0,0) - distance of 1
        predictions = torch.tensor([6])  # (1,0)
        targets = torch.tensor([0])      # (0,0)
        
        metric.update(predictions, targets)
        distance = metric.compute()
        
        assert distance == 1.0


class TestConfusionMatrix:
    """Tests for ConfusionMatrix."""

    @pytest.fixture
    def cm(self):
        """Create confusion matrix fixture."""
        return ConfusionMatrix(num_classes=5)

    def test_update(self, cm):
        """Test confusion matrix update."""
        predictions = torch.tensor([0, 1, 2, 0, 1])
        targets = torch.tensor([0, 1, 1, 0, 2])
        
        cm.update(predictions, targets)
        
        assert cm.matrix[0, 0] == 2  # True positives for class 0
        assert cm.matrix[1, 1] == 1  # True positives for class 1
        assert cm.matrix[1, 2] == 1  # Class 1 predicted as 2

    def test_precision_recall(self, cm):
        """Test precision and recall computation."""
        predictions = torch.tensor([0, 0, 1, 1, 2])
        targets = torch.tensor([0, 1, 1, 1, 2])
        
        cm.update(predictions, targets)
        
        precision = cm.compute_precision()
        recall = cm.compute_recall()
        
        assert len(precision) == 5
        assert len(recall) == 5


class TestRewardTracker:
    """Tests for RewardTracker."""

    @pytest.fixture
    def tracker(self):
        """Create tracker fixture."""
        return RewardTracker(window_size=10)

    def test_update(self, tracker):
        """Test reward tracking."""
        for i in range(20):
            tracker.update(reward=float(i), episode_length=100, success=i > 10)
        
        mean_reward = tracker.get_mean_reward()
        success_rate = tracker.get_success_rate()
        
        assert mean_reward > 0
        assert 0 <= success_rate <= 1

    def test_get_stats(self, tracker):
        """Test stats retrieval."""
        tracker.update(reward=1.0, episode_length=50, success=True)
        tracker.update(reward=2.0, episode_length=60, success=False)
        
        stats = tracker.get_stats()
        
        assert "mean_reward" in stats
        assert "success_rate" in stats
        assert "mean_length" in stats
        assert "total_episodes" in stats
        assert stats["total_episodes"] == 2


class TestMetricCollection:
    """Tests for MetricCollection."""

    def test_collection(self):
        """Test metric collection."""
        metrics = MetricCollection({
            "accuracy": AccuracyMetric(),
            "top5": TopKAccuracyMetric(k=5),
        })
        
        predictions = torch.randn(10, 10)
        targets = torch.randint(0, 10, (10,))
        
        metrics.update(predictions, targets)
        results = metrics.compute()
        
        assert "accuracy" in results
        assert "top5" in results


class TestClassifierTrainer:
    """Tests for ClassifierTrainer."""

    @pytest.fixture
    def trainer(self):
        """Create trainer fixture."""
        model = SimpleModel(num_classes=10)
        config = TrainingConfig(epochs=2, batch_size=4)
        return ClassifierTrainer(model, config)

    @pytest.fixture
    def dataloaders(self):
        """Create dataloader fixtures."""
        # Create dummy data
        x_train = torch.randn(20, 1, 64, 64)
        y_train = torch.randint(0, 10, (20,))
        x_val = torch.randn(10, 1, 64, 64)
        y_val = torch.randint(0, 10, (10,))
        
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=4)
        val_loader = DataLoader(val_dataset, batch_size=4)
        
        return train_loader, val_loader

    def test_train_epoch(self, trainer, dataloaders):
        """Test single epoch training."""
        train_loader, _ = dataloaders
        
        loss, acc = trainer.train_epoch(train_loader)
        
        assert loss >= 0
        assert 0 <= acc <= 1

    def test_validate(self, trainer, dataloaders):
        """Test validation."""
        _, val_loader = dataloaders
        
        loss, acc = trainer.validate(val_loader)
        
        assert loss >= 0
        assert 0 <= acc <= 1

    def test_fit(self, trainer, dataloaders):
        """Test full training loop."""
        train_loader, val_loader = dataloaders
        
        history = trainer.fit(train_loader, val_loader)
        
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 2
