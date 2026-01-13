"""
Tests for the models module.
"""

import numpy as np
import pytest
import torch

from aligncav.models import (
    DQN,
    DQNAgent,
    DeepModeClassifier,
    ModeClassifier,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    Transition,
    decode_action,
    encode_action,
)


class TestModeClassifier:
    """Tests for ModeClassifier."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return ModeClassifier(num_classes=36, input_size=64, dropout=0.0)

    def test_forward_shape(self, model):
        """Test forward pass output shape."""
        x = torch.randn(4, 1, 64, 64)
        output = model(x)
        
        assert output.shape == (4, 36)

    def test_predict(self, model):
        """Test prediction method."""
        x = torch.randn(4, 1, 64, 64)
        predictions, confidences = model.predict(x)
        
        assert predictions.shape == (4,)
        assert confidences.shape == (4,)
        assert (confidences >= 0).all()
        assert (confidences <= 1).all()

    def test_predict_mode_indices(self, model):
        """Test mode index prediction."""
        x = torch.randn(4, 1, 64, 64)
        m_indices, n_indices, confidences = model.predict_mode_indices(x, max_mode=5)
        
        assert m_indices.shape == (4,)
        assert n_indices.shape == (4,)
        assert (m_indices >= 0).all()
        assert (m_indices <= 5).all()
        assert (n_indices >= 0).all()
        assert (n_indices <= 5).all()

    def test_gradient_flow(self, model):
        """Test that gradients flow correctly."""
        x = torch.randn(2, 1, 64, 64)
        y = torch.randint(0, 36, (2,))
        
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestDeepModeClassifier:
    """Tests for DeepModeClassifier."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return DeepModeClassifier(num_classes=36, input_size=64, dropout=0.0)

    def test_forward_shape(self, model):
        """Test forward pass output shape."""
        x = torch.randn(4, 1, 64, 64)
        output = model(x)
        
        assert output.shape == (4, 36)


class TestDQN:
    """Tests for DQN."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return DQN(input_size=64, num_actions=81)

    def test_forward_shape(self, model):
        """Test forward pass output shape."""
        x = torch.randn(4, 1, 64, 64)
        output = model(x)
        
        assert output.shape == (4, 81)

    def test_dueling_architecture(self, model):
        """Test that dueling architecture produces valid Q-values."""
        x = torch.randn(4, 1, 64, 64)
        q_values = model(x)
        
        # Q-values should be finite
        assert torch.isfinite(q_values).all()


class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    @pytest.fixture
    def buffer(self):
        """Create buffer fixture."""
        return ReplayBuffer(capacity=100)

    def test_push_and_sample(self, buffer):
        """Test pushing and sampling."""
        for i in range(50):
            buffer.push(
                state=np.random.randn(64, 64).astype(np.float32),
                action=i % 81,
                reward=float(i),
                next_state=np.random.randn(64, 64).astype(np.float32),
                done=False,
            )
        
        assert len(buffer) == 50
        
        samples = buffer.sample(10)
        assert len(samples) == 10
        assert all(isinstance(s, Transition) for s in samples)

    def test_capacity(self, buffer):
        """Test that capacity is respected."""
        for i in range(150):
            buffer.push(
                state=np.zeros((64, 64), dtype=np.float32),
                action=0,
                reward=0.0,
                next_state=np.zeros((64, 64), dtype=np.float32),
                done=False,
            )
        
        assert len(buffer) == 100


class TestPrioritizedReplayBuffer:
    """Tests for PrioritizedReplayBuffer."""

    @pytest.fixture
    def buffer(self):
        """Create buffer fixture."""
        return PrioritizedReplayBuffer(capacity=100)

    def test_push_and_sample(self, buffer):
        """Test pushing and sampling with priorities."""
        for i in range(50):
            buffer.push(
                state=np.random.randn(64, 64).astype(np.float32),
                action=i % 81,
                reward=float(i),
                next_state=np.random.randn(64, 64).astype(np.float32),
                done=False,
            )
        
        assert len(buffer) == 50
        
        transitions, indices, weights = buffer.sample(10)
        assert len(transitions) == 10
        assert len(indices) == 10
        assert len(weights) == 10

    def test_update_priorities(self, buffer):
        """Test priority updates."""
        for i in range(20):
            buffer.push(
                state=np.zeros((64, 64), dtype=np.float32),
                action=0,
                reward=0.0,
                next_state=np.zeros((64, 64), dtype=np.float32),
                done=False,
            )
        
        _, indices, _ = buffer.sample(5)
        new_priorities = np.ones(5) * 10.0
        buffer.update_priorities(indices, new_priorities)
        
        # No assertion needed, just checking it doesn't error


class TestDQNAgent:
    """Tests for DQNAgent."""

    @pytest.fixture
    def agent(self):
        """Create agent fixture."""
        return DQNAgent(
            input_size=64,
            num_actions=81,
            epsilon_start=0.5,
            epsilon_end=0.1,
        )

    def test_select_action(self, agent):
        """Test action selection."""
        state = np.random.randn(64, 64).astype(np.float32)
        action = agent.select_action(state, training=True)
        
        assert 0 <= action < 81

    def test_select_action_greedy(self, agent):
        """Test greedy action selection."""
        state = np.random.randn(64, 64).astype(np.float32)
        
        # With training=False, should be deterministic
        actions = [agent.select_action(state, training=False) for _ in range(5)]
        assert len(set(actions)) == 1  # All same action

    def test_store_transition(self, agent):
        """Test transition storage."""
        state = np.random.randn(64, 64).astype(np.float32)
        next_state = np.random.randn(64, 64).astype(np.float32)
        
        agent.store_transition(state, 40, 1.0, next_state, False)
        assert len(agent.memory) == 1

    def test_train_step(self, agent):
        """Test training step."""
        # Fill buffer
        for _ in range(100):
            state = np.random.randn(64, 64).astype(np.float32)
            next_state = np.random.randn(64, 64).astype(np.float32)
            agent.store_transition(state, np.random.randint(81), np.random.randn(), next_state, False)
        
        loss = agent.train_step(batch_size=32)
        assert loss is not None
        assert loss >= 0

    def test_epsilon_decay(self, agent):
        """Test epsilon decay."""
        initial_epsilon = agent.epsilon
        
        # Simulate some steps
        agent.steps_done = 1000
        
        assert agent.epsilon < initial_epsilon


class TestActionEncoding:
    """Tests for action encoding/decoding."""

    def test_decode_action(self):
        """Test action decoding."""
        # Action 0 should be all -1
        action = decode_action(0, num_motors=4)
        expected = np.array([-1, -1, -1, -1], dtype=np.float32)
        np.testing.assert_array_equal(action, expected)
        
        # Action 40 (middle) should be all 0
        action = decode_action(40, num_motors=4)
        expected = np.array([0, 0, 0, 0], dtype=np.float32)
        np.testing.assert_array_equal(action, expected)

    def test_encode_action(self):
        """Test action encoding."""
        # All zeros should give middle action
        action = encode_action(np.array([0, 0, 0, 0]))
        assert action == 40
        
        # All -1 should give action 0
        action = encode_action(np.array([-1, -1, -1, -1]))
        assert action == 0

    def test_encode_decode_roundtrip(self):
        """Test encode/decode roundtrip."""
        for action_idx in [0, 40, 80]:
            decoded = decode_action(action_idx, num_motors=4)
            encoded = encode_action(decoded)
            assert encoded == action_idx
