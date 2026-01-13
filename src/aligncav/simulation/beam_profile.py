"""
Beam Profile Analysis and Evaluation.

This module provides tools for analyzing and evaluating beam profiles,
including quality metrics and comparison functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.optimize import curve_fit


@dataclass
class BeamMetrics:
    """Metrics describing a beam profile."""

    centroid_x: float
    centroid_y: float
    sigma_x: float
    sigma_y: float
    ellipticity: float
    peak_intensity: float
    total_power: float
    m_squared: float  # M² beam quality factor


class BeamProfileEvaluator:
    """
    Evaluator for beam profile quality and characteristics.

    Provides methods for analyzing beam images and computing
    quality metrics relevant to cavity alignment.
    """

    def __init__(
        self,
        pixel_scale: float = 1e-6,
        reference_waist: Optional[float] = None,
    ):
        """
        Initialize beam profile evaluator.

        Args:
            pixel_scale: Physical size of each pixel in meters
            reference_waist: Reference beam waist for comparison
        """
        self.pixel_scale = pixel_scale
        self.reference_waist = reference_waist

    def compute_centroid(self, image: NDArray) -> Tuple[float, float]:
        """
        Compute beam centroid (center of mass).

        Args:
            image: 2D beam intensity image

        Returns:
            Tuple of (x, y) centroid coordinates in pixels
        """
        image = np.asarray(image, dtype=np.float64)
        image = np.maximum(image, 0)  # Ensure non-negative

        total = image.sum()
        if total == 0:
            return image.shape[1] / 2, image.shape[0] / 2

        y_coords, x_coords = np.mgrid[: image.shape[0], : image.shape[1]]
        centroid_x = (x_coords * image).sum() / total
        centroid_y = (y_coords * image).sum() / total

        return float(centroid_x), float(centroid_y)

    def compute_beam_widths(
        self, image: NDArray, centroid: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, float]:
        """
        Compute beam widths (second moments) in x and y directions.

        Args:
            image: 2D beam intensity image
            centroid: Pre-computed centroid (optional)

        Returns:
            Tuple of (sigma_x, sigma_y) in pixels
        """
        image = np.asarray(image, dtype=np.float64)
        image = np.maximum(image, 0)

        if centroid is None:
            centroid = self.compute_centroid(image)

        cx, cy = centroid
        total = image.sum()

        if total == 0:
            return 1.0, 1.0

        y_coords, x_coords = np.mgrid[: image.shape[0], : image.shape[1]]

        sigma_x = np.sqrt(((x_coords - cx) ** 2 * image).sum() / total)
        sigma_y = np.sqrt(((y_coords - cy) ** 2 * image).sum() / total)

        return float(sigma_x), float(sigma_y)

    def compute_ellipticity(self, sigma_x: float, sigma_y: float) -> float:
        """
        Compute beam ellipticity.

        Args:
            sigma_x: Beam width in x direction
            sigma_y: Beam width in y direction

        Returns:
            Ellipticity (1.0 for circular beam)
        """
        if sigma_x == 0 or sigma_y == 0:
            return 1.0
        return min(sigma_x, sigma_y) / max(sigma_x, sigma_y)

    def analyze(self, image: NDArray) -> BeamMetrics:
        """
        Perform complete beam analysis.

        Args:
            image: 2D beam intensity image

        Returns:
            BeamMetrics object with all computed metrics
        """
        image = np.asarray(image, dtype=np.float64)

        centroid = self.compute_centroid(image)
        sigma_x, sigma_y = self.compute_beam_widths(image, centroid)
        ellipticity = self.compute_ellipticity(sigma_x, sigma_y)

        peak_intensity = float(image.max())
        total_power = float(image.sum())

        # Estimate M² (simplified)
        m_squared = self._estimate_m_squared(image, sigma_x, sigma_y)

        return BeamMetrics(
            centroid_x=centroid[0],
            centroid_y=centroid[1],
            sigma_x=sigma_x * self.pixel_scale,
            sigma_y=sigma_y * self.pixel_scale,
            ellipticity=ellipticity,
            peak_intensity=peak_intensity,
            total_power=total_power,
            m_squared=m_squared,
        )

    def _estimate_m_squared(self, image: NDArray, sigma_x: float, sigma_y: float) -> float:
        """
        Estimate M² beam quality factor.

        For a perfect Gaussian beam, M² = 1.
        Higher order modes have M² > 1.

        Args:
            image: 2D beam intensity image
            sigma_x: Measured beam width in x
            sigma_y: Measured beam width in y

        Returns:
            Estimated M² value
        """
        if self.reference_waist is None:
            return 1.0

        # Convert reference waist to pixels
        ref_sigma = self.reference_waist / (self.pixel_scale * 2)

        # M² estimated from width ratio
        avg_sigma = (sigma_x + sigma_y) / 2
        m_squared = (avg_sigma / ref_sigma) ** 2

        return float(max(1.0, m_squared))

    def compute_correlation(self, image1: NDArray, image2: NDArray) -> float:
        """
        Compute normalized cross-correlation between two beam images.

        Args:
            image1: First beam image
            image2: Second beam image (reference)

        Returns:
            Correlation coefficient (0 to 1)
        """
        img1 = np.asarray(image1, dtype=np.float64).flatten()
        img2 = np.asarray(image2, dtype=np.float64).flatten()

        # Normalize
        img1 = img1 - img1.mean()
        img2 = img2 - img2.mean()

        std1 = img1.std()
        std2 = img2.std()

        if std1 == 0 or std2 == 0:
            return 0.0

        correlation = np.dot(img1, img2) / (len(img1) * std1 * std2)
        return float(np.clip(correlation, 0, 1))

    def compute_variance_difference(self, image1: NDArray, image2: NDArray) -> float:
        """
        Compute normalized variance difference between images.

        Args:
            image1: First beam image
            image2: Second beam image (reference)

        Returns:
            Normalized variance difference (0 to 1)
        """
        var1 = np.var(image1)
        var2 = np.var(image2)

        if var2 == 0:
            return 1.0 if var1 > 0 else 0.0

        var_diff = abs(var1 - var2) / max(var1, var2)
        return float(var_diff)

    def compute_alignment_reward(
        self,
        current_image: NDArray,
        target_image: NDArray,
        correlation_weight: float = 0.99,
        base_reward: float = 0.01,
        var_penalty_scale: float = 50.0,
    ) -> float:
        """
        Compute alignment reward based on image comparison.

        Uses the formula: reward = base + weight * correlation * exp(-scale * var_diff)

        Args:
            current_image: Current beam image
            target_image: Target (aligned) beam image
            correlation_weight: Weight for correlation component
            base_reward: Minimum base reward
            var_penalty_scale: Scale for variance penalty

        Returns:
            Reward value
        """
        correlation = self.compute_correlation(current_image, target_image)
        var_diff = self.compute_variance_difference(current_image, target_image)

        reward = base_reward + correlation_weight * correlation * np.exp(-var_penalty_scale * var_diff)
        return float(reward)

    def fit_gaussian(self, image: NDArray) -> Optional[dict]:
        """
        Fit a 2D Gaussian to the beam image.

        Args:
            image: 2D beam intensity image

        Returns:
            Dictionary with fit parameters or None if fit fails
        """
        try:
            # Get initial estimates
            centroid = self.compute_centroid(image)
            sigma_x, sigma_y = self.compute_beam_widths(image, centroid)

            # Create coordinate grids
            y, x = np.mgrid[: image.shape[0], : image.shape[1]]

            def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, offset):
                x, y = coords
                return (
                    amplitude
                    * np.exp(
                        -((x - x0) ** 2 / (2 * sigma_x**2) + (y - y0) ** 2 / (2 * sigma_y**2))
                    )
                    + offset
                ).ravel()

            initial_guess = [
                image.max(),
                centroid[0],
                centroid[1],
                sigma_x,
                sigma_y,
                image.min(),
            ]

            bounds = (
                [0, 0, 0, 0.1, 0.1, 0],
                [np.inf, image.shape[1], image.shape[0], image.shape[1], image.shape[0], np.inf],
            )

            popt, _ = curve_fit(
                gaussian_2d,
                (x, y),
                image.ravel(),
                p0=initial_guess,
                bounds=bounds,
                maxfev=5000,
            )

            return {
                "amplitude": popt[0],
                "x0": popt[1],
                "y0": popt[2],
                "sigma_x": popt[3],
                "sigma_y": popt[4],
                "offset": popt[5],
            }

        except Exception:
            return None


def compute_beam_quality(image: NDArray, target: Optional[NDArray] = None) -> float:
    """
    Convenience function to compute beam quality metric.

    Args:
        image: Beam intensity image
        target: Optional target image for comparison

    Returns:
        Quality metric (0 to 1)
    """
    evaluator = BeamProfileEvaluator()
    metrics = evaluator.analyze(image)

    # Base quality from beam properties
    quality = metrics.ellipticity / metrics.m_squared

    # If target provided, include correlation
    if target is not None:
        correlation = evaluator.compute_correlation(image, target)
        quality = 0.5 * quality + 0.5 * correlation

    return float(np.clip(quality, 0, 1))
