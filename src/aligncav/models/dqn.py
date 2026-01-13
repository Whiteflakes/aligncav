"""
Deep Q-Network (DQN) for Cavity Alignment.

This module implements DQN and related components for reinforcement
learning-based cavity alignment.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray


@dataclass
class Transition:
    """A single experience transition."""

    state: NDArray
    action: int
    reward: float
    next_state: NDArray
    done: bool


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.

    Stores transitions and samples random batches for training.
    """

    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: NDArray,
        action: int,
        reward: float,
        next_state: NDArray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a random batch of transitions."""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.

    Samples transitions based on TD-error priority.
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent
            beta_increment: How much to increase beta each sample
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.buffer: List[Optional[Transition]] = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(
        self,
        state: NDArray,
        action: int,
        reward: float,
        next_state: NDArray,
        done: bool,
    ) -> None:
        """Add transition with max priority."""
        max_priority = self.priorities[: self.size].max() if self.size > 0 else 1.0

        self.buffer[self.position] = Transition(state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[List[Transition], NDArray, NDArray]:
        """
        Sample batch with priorities.

        Returns:
            Tuple of (transitions, indices, importance weights)
        """
        if self.size == 0:
            return [], np.array([]), np.array([])

        # Calculate sampling probabilities
        priorities = self.priorities[: self.size] ** self.alpha
        probs = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(self.size, min(batch_size, self.size), p=probs, replace=False)

        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        # Increase beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        transitions = [self.buffer[i] for i in indices]
        return transitions, indices, weights.astype(np.float32)

    def update_priorities(self, indices: NDArray, priorities: NDArray) -> None:
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small constant to avoid zero priority

    def __len__(self) -> int:
        return self.size


class DQN(nn.Module):
    """
    Deep Q-Network for cavity alignment.

    Takes beam images as input and outputs Q-values for each
    possible motor action combination.

    Action space: 81 discrete actions (4 motors × 3 choices each)
    Each motor can move {-1, 0, +1} steps.
    """

    def __init__(
        self,
        input_size: int = 256,
        num_actions: int = 81,
        in_channels: int = 1,
    ):
        """
        Initialize DQN.

        Args:
            input_size: Input image size
            num_actions: Number of discrete actions (3^num_motors)
            in_channels: Number of input channels
        """
        super().__init__()

        self.num_actions = num_actions
        self.input_size = input_size

        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Calculate feature size
        self._feature_size = self._get_feature_size(input_size, in_channels)

        # Value stream (Dueling DQN)
        self.value_stream = nn.Sequential(
            nn.Linear(self._feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Advantage stream (Dueling DQN)
        self.advantage_stream = nn.Sequential(
            nn.Linear(self._feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def _get_feature_size(self, input_size: int, in_channels: int) -> int:
        """Calculate output size of feature extractor."""
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            features = self.features(dummy)
            return features.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Q-values for each action, shape (batch, num_actions)
        """
        features = self.features(x)
        features = features.view(features.size(0), -1)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Dueling architecture: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values


class DQNAgent:
    """
    DQN Agent for cavity alignment.

    Handles action selection, training, and target network updates.
    """

    def __init__(
        self,
        input_size: int = 256,
        num_actions: int = 81,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10000,
        target_update: int = 1000,
        device: Optional[str] = None,
    ):
        """
        Initialize DQN agent.

        Args:
            input_size: Input image size
            num_actions: Number of discrete actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Steps for epsilon decay
            target_update: Steps between target network updates
            device: Device to use (auto-detect if None)
        """
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.policy_net = DQN(input_size, num_actions).to(self.device)
        self.target_net = DQN(input_size, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.memory = ReplayBuffer()

        # Training state
        self.steps_done = 0

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self.steps_done / self.epsilon_decay
        )

    def select_action(self, state: NDArray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state (beam image)
            training: Whether in training mode

        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(1).item())

    def store_transition(
        self,
        state: NDArray,
        action: int,
        reward: float,
        next_state: NDArray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """
        Perform one training step.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Loss value or None if not enough samples
        """
        if len(self.memory) < batch_size:
            return None

        # Sample batch
        transitions = self.memory.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(
            np.array([t.state for t in transitions])
        ).unsqueeze(1).to(self.device)
        actions = torch.LongTensor([t.action for t in transitions]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in transitions]).to(self.device)
        next_states = torch.FloatTensor(
            np.array([t.next_state for t in transitions])
        ).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor([t.done for t in transitions]).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Next Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))

        # Loss
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self) -> None:
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]


def decode_action(action_idx: int, num_motors: int = 4) -> NDArray:
    """
    Decode discrete action index to motor commands.

    Maps action index to {-1, 0, +1} for each motor.

    Args:
        action_idx: Action index (0 to 3^num_motors - 1)
        num_motors: Number of motors

    Returns:
        Array of motor commands
    """
    commands = np.zeros(num_motors, dtype=np.float32)
    for i in range(num_motors):
        commands[i] = (action_idx % 3) - 1
        action_idx //= 3
    return commands


def encode_action(commands: NDArray) -> int:
    """
    Encode motor commands to discrete action index.

    Args:
        commands: Array of motor commands {-1, 0, +1}

    Returns:
        Action index
    """
    action_idx = 0
    for i, cmd in enumerate(commands):
        action_idx += int(cmd + 1) * (3**i)
    return action_idx
