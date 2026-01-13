"""
Hermite-Gaussian Mode Generation for Fabry-Pérot Cavity Simulation.

This module provides utilities for generating Hermite-Gaussian (HG) beam modes,
which are fundamental solutions to the paraxial wave equation in optical cavities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import hermite


@dataclass
class ModeParameters:
    """Parameters for Hermite-Gaussian mode generation."""

    wavelength: float = 1064e-9  # meters
    waist: float = 100e-6  # beam waist in meters
    image_size: int = 256
    pixel_scale: float = 1e-6  # meters per pixel

    @property
    def k(self) -> float:
        """Wave number."""
        return 2 * np.pi / self.wavelength

    @property
    def rayleigh_range(self) -> float:
        """Rayleigh range (z_R)."""
        return np.pi * self.waist**2 / self.wavelength


class HGModeGenerator:
    """
    Generator for Hermite-Gaussian beam modes.

    HG modes are characterized by two indices (m, n) representing the
    number of nodes in the x and y directions respectively.

    Attributes:
        params: Mode generation parameters
        max_mode: Maximum mode index (modes from 0 to max_mode)
        add_noise: Whether to add realistic noise to generated modes
    """

    def __init__(
        self,
        params: Optional[ModeParameters] = None,
        max_mode: int = 10,
        add_noise: bool = True,
        noise_level: float = 0.05,
    ):
        """
        Initialize the HG mode generator.

        Args:
            params: Mode generation parameters
            max_mode: Maximum mode index (generates modes 0 to max_mode)
            add_noise: Whether to add noise to generated images
            noise_level: Standard deviation of Gaussian noise (relative)
        """
        self.params = params or ModeParameters()
        self.max_mode = max_mode
        self.add_noise = add_noise
        self.noise_level = noise_level
        self._setup_coordinates()
        self._cache_hermite_polys()

    def _setup_coordinates(self) -> None:
        """Setup coordinate grids for mode generation."""
        size = self.params.image_size
        scale = self.params.pixel_scale

        # Create centered coordinate grid
        coords = np.linspace(
            -size // 2 * scale, size // 2 * scale, size, dtype=np.float32
        )
        self.x, self.y = np.meshgrid(coords, coords)

    def _cache_hermite_polys(self) -> None:
        """Cache Hermite polynomial coefficients for efficiency."""
        self._hermite_coeffs = [hermite(n) for n in range(self.max_mode + 1)]

    def _hermite_poly(self, n: int, x: NDArray) -> NDArray:
        """
        Evaluate Hermite polynomial of order n.

        Args:
            n: Order of the Hermite polynomial
            x: Points at which to evaluate

        Returns:
            Polynomial values at given points
        """
        if n <= self.max_mode:
            return self._hermite_coeffs[n](x)
        return hermite(n)(x)

    def generate_mode(
        self,
        m: int,
        n: int,
        waist_scale: float = 1.0,
        rotation: float = 0.0,
        offset: Tuple[float, float] = (0.0, 0.0),
    ) -> NDArray:
        """
        Generate a single HG mode intensity pattern.

        Args:
            m: Mode index in x direction
            n: Mode index in y direction
            waist_scale: Scale factor for beam waist
            rotation: Rotation angle in radians
            offset: (x, y) offset in pixels

        Returns:
            2D numpy array with the mode intensity pattern
        """
        w = self.params.waist * waist_scale
        scale = self.params.pixel_scale

        # Apply rotation and offset
        x = self.x
        y = self.y

        if rotation != 0:
            cos_r, sin_r = np.cos(rotation), np.sin(rotation)
            x_rot = x * cos_r - y * sin_r
            y_rot = x * sin_r + y * cos_r
            x, y = x_rot, y_rot

        if offset != (0.0, 0.0):
            x = x - offset[0] * scale
            y = y - offset[1] * scale

        # Normalized coordinates
        x_norm = np.sqrt(2) * x / w
        y_norm = np.sqrt(2) * y / w

        # Hermite-Gaussian mode amplitude
        hm = self._hermite_poly(m, x_norm)
        hn = self._hermite_poly(n, y_norm)

        # Gaussian envelope
        r_squared = x**2 + y**2
        gaussian = np.exp(-r_squared / w**2)

        # Mode amplitude
        amplitude = hm * hn * gaussian

        # Normalization factor
        import math
        norm = 1.0 / (np.sqrt(2 ** (m + n) * math.factorial(m) * math.factorial(n)))
        amplitude = amplitude * norm

        # Intensity is |amplitude|^2
        intensity = np.abs(amplitude) ** 2

        # Normalize to [0, 1]
        intensity = intensity / intensity.max() if intensity.max() > 0 else intensity

        # Add noise if requested
        if self.add_noise:
            noise = np.random.normal(0, self.noise_level, intensity.shape)
            intensity = np.clip(intensity + noise * intensity.max(), 0, 1)

        return intensity.astype(np.float32)

    def generate_superposition(
        self,
        modes: list[Tuple[int, int, float]],
        normalize: bool = True,
    ) -> NDArray:
        """
        Generate a superposition of HG modes.

        Args:
            modes: List of (m, n, weight) tuples
            normalize: Whether to normalize output to [0, 1]

        Returns:
            2D numpy array with superposition intensity
        """
        result = np.zeros((self.params.image_size, self.params.image_size), dtype=np.float32)

        for m, n, weight in modes:
            mode_intensity = self.generate_mode(m, n)
            result += weight * mode_intensity

        if normalize and result.max() > 0:
            result = result / result.max()

        return result

    def generate_random_mode(
        self,
        include_variations: bool = True,
    ) -> Tuple[NDArray, int, int]:
        """
        Generate a random HG mode with optional variations.

        Args:
            include_variations: Add random waist scale, rotation, offset

        Returns:
            Tuple of (intensity array, m index, n index)
        """
        m = np.random.randint(0, self.max_mode + 1)
        n = np.random.randint(0, self.max_mode + 1)

        kwargs = {}
        if include_variations:
            kwargs["waist_scale"] = np.random.uniform(0.8, 1.2)
            kwargs["rotation"] = np.random.uniform(-0.1, 0.1)
            kwargs["offset"] = (np.random.uniform(-10, 10), np.random.uniform(-10, 10))

        intensity = self.generate_mode(m, n, **kwargs)
        return intensity, m, n

    def get_class_from_indices(self, m: int, n: int) -> int:
        """
        Convert mode indices to class label.

        Uses encoding: class = m * (max_mode + 1) + n

        Args:
            m: Mode index in x direction
            n: Mode index in y direction

        Returns:
            Integer class label
        """
        return m * (self.max_mode + 1) + n

    def get_indices_from_class(self, class_idx: int) -> Tuple[int, int]:
        """
        Convert class label back to mode indices.

        Args:
            class_idx: Integer class label

        Returns:
            Tuple of (m, n) mode indices
        """
        m = class_idx // (self.max_mode + 1)
        n = class_idx % (self.max_mode + 1)
        return m, n

    @property
    def num_classes(self) -> int:
        """Total number of mode classes."""
        return (self.max_mode + 1) ** 2

    def generate_dataset(
        self,
        samples_per_class: int = 100,
        include_variations: bool = True,
    ) -> Tuple[NDArray, NDArray]:
        """
        Generate a complete dataset of HG modes.

        Args:
            samples_per_class: Number of samples per (m, n) combination
            include_variations: Add random variations to each sample

        Returns:
            Tuple of (images array, labels array)
        """
        total_samples = self.num_classes * samples_per_class
        images = np.zeros(
            (total_samples, self.params.image_size, self.params.image_size),
            dtype=np.float32,
        )
        labels = np.zeros(total_samples, dtype=np.int64)

        idx = 0
        for m in range(self.max_mode + 1):
            for n in range(self.max_mode + 1):
                class_idx = self.get_class_from_indices(m, n)
                for _ in range(samples_per_class):
                    if include_variations:
                        images[idx] = self.generate_mode(
                            m,
                            n,
                            waist_scale=np.random.uniform(0.8, 1.2),
                            rotation=np.random.uniform(-0.1, 0.1),
                            offset=(
                                np.random.uniform(-10, 10),
                                np.random.uniform(-10, 10),
                            ),
                        )
                    else:
                        images[idx] = self.generate_mode(m, n)
                    labels[idx] = class_idx
                    idx += 1

        return images, labels
