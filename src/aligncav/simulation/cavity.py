"""
Fabry-Pérot Cavity Simulation.

This module simulates the physics of a Fabry-Pérot optical cavity,
including mode resonance, transmission, and alignment behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class CavityConfig:
    """Configuration parameters for Fabry-Pérot cavity."""

    # Mirror parameters
    mirror1_reflectivity: float = 0.99
    mirror2_reflectivity: float = 0.99
    mirror1_radius: float = 0.5  # meters (radius of curvature)
    mirror2_radius: float = 0.5  # meters

    # Cavity geometry
    length: float = 0.1  # cavity length in meters

    # Beam parameters
    wavelength: float = 1064e-9  # meters
    input_waist: float = 100e-6  # input beam waist in meters

    # Motor/alignment parameters
    num_motors: int = 4
    motor_step_size: float = 1e-6  # meters per step

    # Simulation parameters
    image_size: int = 256
    pixel_scale: float = 1e-6

    @property
    def finesse(self) -> float:
        """Cavity finesse."""
        r = np.sqrt(self.mirror1_reflectivity * self.mirror2_reflectivity)
        return np.pi * r / (1 - r**2)

    @property
    def fsr(self) -> float:
        """Free spectral range in Hz."""
        c = 299792458  # speed of light
        return c / (2 * self.length)

    @property
    def linewidth(self) -> float:
        """Cavity linewidth in Hz."""
        return self.fsr / self.finesse

    @property
    def g_parameters(self) -> Tuple[float, float]:
        """Cavity g-parameters (g1, g2)."""
        g1 = 1 - self.length / self.mirror1_radius
        g2 = 1 - self.length / self.mirror2_radius
        return g1, g2

    @property
    def is_stable(self) -> bool:
        """Check if cavity is geometrically stable."""
        g1, g2 = self.g_parameters
        return 0 <= g1 * g2 <= 1

    @property
    def cavity_waist(self) -> float:
        """Calculate the cavity mode waist."""
        g1, g2 = self.g_parameters
        if not self.is_stable:
            return self.input_waist

        L = self.length
        lam = self.wavelength

        w0_squared = (lam * L / np.pi) * np.sqrt(g1 * g2 * (1 - g1 * g2)) / (g1 + g2 - 2 * g1 * g2)
        return np.sqrt(abs(w0_squared)) if w0_squared > 0 else self.input_waist


@dataclass
class AlignmentState:
    """Current alignment state of the cavity."""

    # Mirror tilts (in radians)
    mirror1_tilt_x: float = 0.0
    mirror1_tilt_y: float = 0.0
    mirror2_tilt_x: float = 0.0
    mirror2_tilt_y: float = 0.0

    # Beam position offsets (in meters)
    beam_offset_x: float = 0.0
    beam_offset_y: float = 0.0

    def as_array(self) -> NDArray:
        """Convert to numpy array."""
        return np.array(
            [
                self.mirror1_tilt_x,
                self.mirror1_tilt_y,
                self.mirror2_tilt_x,
                self.mirror2_tilt_y,
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_array(cls, arr: NDArray) -> "AlignmentState":
        """Create from numpy array."""
        return cls(
            mirror1_tilt_x=float(arr[0]),
            mirror1_tilt_y=float(arr[1]),
            mirror2_tilt_x=float(arr[2]),
            mirror2_tilt_y=float(arr[3]),
        )


class CavitySimulator:
    """
    Simulator for Fabry-Pérot cavity alignment.

    This class simulates the optical behavior of a Fabry-Pérot cavity,
    including the effect of mirror alignment on transmitted beam patterns.
    """

    def __init__(
        self,
        config: Optional[CavityConfig] = None,
        initial_state: Optional[AlignmentState] = None,
    ):
        """
        Initialize cavity simulator.

        Args:
            config: Cavity configuration parameters
            initial_state: Initial alignment state
        """
        self.config = config or CavityConfig()
        self.state = initial_state or AlignmentState()
        self._setup_coordinates()

    def _setup_coordinates(self) -> None:
        """Setup coordinate grids."""
        size = self.config.image_size
        scale = self.config.pixel_scale

        coords = np.linspace(-size // 2 * scale, size // 2 * scale, size, dtype=np.float32)
        self.x, self.y = np.meshgrid(coords, coords)

    def reset(self, random_misalignment: bool = False, max_misalignment: float = 1e-4) -> None:
        """
        Reset cavity to initial or random state.

        Args:
            random_misalignment: Whether to add random misalignment
            max_misalignment: Maximum misalignment angle in radians
        """
        if random_misalignment:
            self.state = AlignmentState(
                mirror1_tilt_x=np.random.uniform(-max_misalignment, max_misalignment),
                mirror1_tilt_y=np.random.uniform(-max_misalignment, max_misalignment),
                mirror2_tilt_x=np.random.uniform(-max_misalignment, max_misalignment),
                mirror2_tilt_y=np.random.uniform(-max_misalignment, max_misalignment),
            )
        else:
            self.state = AlignmentState()

    def apply_action(self, action: NDArray) -> None:
        """
        Apply motor adjustment action.

        Args:
            action: Array of motor movements [-1, 0, 1] for each motor
        """
        step = self.config.motor_step_size * 1e-2  # Convert to radians approximately

        self.state.mirror1_tilt_x += action[0] * step
        self.state.mirror1_tilt_y += action[1] * step
        self.state.mirror2_tilt_x += action[2] * step
        self.state.mirror2_tilt_y += action[3] * step

    def get_transmitted_beam(self) -> NDArray:
        """
        Calculate transmitted beam pattern.

        Returns:
            2D array representing transmitted intensity pattern
        """
        w = self.config.cavity_waist

        # Base Gaussian
        r_squared = self.x**2 + self.y**2
        gaussian = np.exp(-2 * r_squared / w**2)

        # Compute effective beam displacement from misalignment
        # Tilts cause beam walk-off
        tilt_effect_x = (self.state.mirror1_tilt_x + self.state.mirror2_tilt_x) * self.config.length
        tilt_effect_y = (self.state.mirror1_tilt_y + self.state.mirror2_tilt_y) * self.config.length

        # Apply displacement
        x_shifted = self.x - tilt_effect_x
        y_shifted = self.y - tilt_effect_y

        # Calculate coupling efficiency (overlap with cavity mode)
        mode_mismatch = (x_shifted**2 + y_shifted**2) / w**2
        coupling = np.exp(-mode_mismatch)

        # Higher-order mode excitation from tilts
        tilt_magnitude = np.sqrt(
            self.state.mirror1_tilt_x**2
            + self.state.mirror1_tilt_y**2
            + self.state.mirror2_tilt_x**2
            + self.state.mirror2_tilt_y**2
        )

        # Generate higher order mode contributions
        from .mode_generator import HGModeGenerator, ModeParameters

        params = ModeParameters(
            wavelength=self.config.wavelength,
            waist=w,
            image_size=self.config.image_size,
            pixel_scale=self.config.pixel_scale,
        )
        generator = HGModeGenerator(params=params, add_noise=False)

        # Start with fundamental mode
        intensity = generator.generate_mode(0, 0)

        # Add higher order mode contributions based on misalignment
        if tilt_magnitude > 1e-6:
            # TEM01 and TEM10 modes excited by tilts
            mode_01 = generator.generate_mode(0, 1)
            mode_10 = generator.generate_mode(1, 0)

            tilt_weight = min(tilt_magnitude * 1e4, 0.5)  # Scale and cap contribution
            intensity = (1 - tilt_weight) * intensity + tilt_weight * 0.5 * (mode_01 + mode_10)

        # Apply overall coupling loss
        alignment_quality = self.get_alignment_quality()
        intensity = intensity * alignment_quality

        # Add small noise
        noise = np.random.normal(0, 0.02, intensity.shape)
        intensity = np.clip(intensity + noise, 0, 1)

        return intensity.astype(np.float32)

    def get_alignment_quality(self) -> float:
        """
        Calculate overall alignment quality metric.

        Returns:
            Float from 0 (completely misaligned) to 1 (perfectly aligned)
        """
        # Calculate total misalignment
        tilts = self.state.as_array()
        total_misalignment = np.sum(tilts**2)

        # Exponential decay of quality with misalignment
        characteristic_scale = 1e-8  # rad^2
        quality = np.exp(-total_misalignment / characteristic_scale)

        return float(quality)

    def get_transmitted_power(self) -> float:
        """
        Calculate total transmitted power (normalized).

        Returns:
            Normalized transmitted power (0 to 1)
        """
        beam = self.get_transmitted_beam()
        return float(beam.sum() / beam.size)

    def is_aligned(self, threshold: float = 0.95) -> bool:
        """
        Check if cavity is well-aligned.

        Args:
            threshold: Quality threshold for alignment

        Returns:
            True if alignment quality exceeds threshold
        """
        return self.get_alignment_quality() >= threshold


class CavityEnvironment:
    """
    Reinforcement learning environment for cavity alignment.

    Wraps CavitySimulator to provide a gym-like interface for RL training.
    """

    def __init__(
        self,
        config: Optional[CavityConfig] = None,
        max_steps: int = 100,
        target_quality: float = 0.95,
    ):
        """
        Initialize RL environment.

        Args:
            config: Cavity configuration
            max_steps: Maximum steps per episode
            target_quality: Target alignment quality
        """
        self.simulator = CavitySimulator(config)
        self.config = self.simulator.config
        self.max_steps = max_steps
        self.target_quality = target_quality
        self.current_step = 0

        # Action space: 3^4 = 81 discrete actions (4 motors x 3 choices each)
        self.num_actions = 3**self.config.num_motors

    def reset(self) -> NDArray:
        """
        Reset environment to initial state.

        Returns:
            Initial observation (beam image)
        """
        self.simulator.reset(random_misalignment=True)
        self.current_step = 0
        return self.simulator.get_transmitted_beam()

    def step(self, action: int) -> Tuple[NDArray, float, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: Discrete action index (0 to 80)

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Decode discrete action to motor commands
        motor_commands = self._decode_action(action)
        self.simulator.apply_action(motor_commands)
        self.current_step += 1

        # Get new observation
        observation = self.simulator.get_transmitted_beam()

        # Calculate reward
        quality = self.simulator.get_alignment_quality()
        reward = self._calculate_reward(quality)

        # Check termination
        done = quality >= self.target_quality or self.current_step >= self.max_steps

        info = {
            "alignment_quality": quality,
            "steps": self.current_step,
            "aligned": quality >= self.target_quality,
        }

        return observation, reward, done, info

    def _decode_action(self, action_idx: int) -> NDArray:
        """
        Decode discrete action index to motor commands.

        Maps action index to {-1, 0, +1} for each of 4 motors.

        Args:
            action_idx: Action index (0 to 80)

        Returns:
            Array of motor commands
        """
        commands = np.zeros(self.config.num_motors, dtype=np.float32)
        for i in range(self.config.num_motors):
            commands[i] = (action_idx % 3) - 1  # -1, 0, or 1
            action_idx //= 3
        return commands

    def _calculate_reward(self, quality: float) -> float:
        """
        Calculate reward from alignment quality.

        Uses reward formula: 0.01 + 0.99 * quality (simplified)

        Args:
            quality: Current alignment quality (0 to 1)

        Returns:
            Reward value
        """
        # Base reward with quality scaling
        reward = 0.01 + 0.99 * quality

        # Bonus for reaching target
        if quality >= self.target_quality:
            reward += 1.0

        return reward
