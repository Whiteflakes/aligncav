"""
Fresnel Propagation for Beam Simulation.

This module implements Fresnel diffraction propagation for simulating
beam propagation through optical systems.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class FresnelPropagator:
    """
    Fresnel propagation for beam simulation.

    Implements the Angular Spectrum Method for Fresnel diffraction,
    which is useful for simulating beam propagation through free space
    and optical elements.
    """

    def __init__(
        self,
        wavelength: float = 1064e-9,
        pixel_size: float = 1e-6,
        grid_size: int = 256,
    ):
        """
        Initialize Fresnel propagator.

        Args:
            wavelength: Light wavelength in meters
            pixel_size: Physical size of each pixel in meters
            grid_size: Size of the computational grid (grid_size x grid_size)
        """
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.grid_size = grid_size
        self.k = 2 * np.pi / wavelength
        self._setup_frequency_grid()

    def _setup_frequency_grid(self) -> None:
        """Setup spatial frequency grid for propagation."""
        # Spatial frequencies
        freq = np.fft.fftfreq(self.grid_size, self.pixel_size)
        self.fx, self.fy = np.meshgrid(freq, freq)
        self.freq_squared = self.fx**2 + self.fy**2

    def propagate(
        self,
        field: NDArray,
        distance: float,
        return_intensity: bool = True,
    ) -> NDArray:
        """
        Propagate optical field using Fresnel diffraction.

        Uses the Angular Spectrum Method:
        U(x,y,z) = F^(-1) { F{U(x,y,0)} * H(fx,fy,z) }

        where H is the transfer function:
        H(fx,fy,z) = exp(i*k*z) * exp(-i*pi*lambda*z*(fx^2+fy^2))

        Args:
            field: Input complex field (2D array)
            distance: Propagation distance in meters
            return_intensity: If True, return intensity; else complex field

        Returns:
            Propagated field (intensity or complex amplitude)
        """
        # Convert to complex if necessary
        field = np.asarray(field, dtype=np.complex128)

        # Angular spectrum
        angular_spectrum = np.fft.fft2(field)

        # Fresnel transfer function
        phase = np.pi * self.wavelength * distance * self.freq_squared
        transfer_function = np.exp(1j * self.k * distance) * np.exp(-1j * phase)

        # Propagate
        propagated_spectrum = angular_spectrum * transfer_function

        # Inverse FFT
        propagated_field = np.fft.ifft2(propagated_spectrum)

        if return_intensity:
            return np.abs(propagated_field) ** 2

        return propagated_field

    def propagate_through_lens(
        self,
        field: NDArray,
        focal_length: float,
        propagation_distance: Optional[float] = None,
    ) -> NDArray:
        """
        Propagate field through a thin lens.

        Args:
            field: Input complex field
            focal_length: Lens focal length in meters
            propagation_distance: Distance to propagate after lens

        Returns:
            Propagated field intensity
        """
        # Create coordinate grid
        coords = (np.arange(self.grid_size) - self.grid_size // 2) * self.pixel_size
        x, y = np.meshgrid(coords, coords)
        r_squared = x**2 + y**2

        # Lens phase transformation
        lens_phase = np.exp(-1j * self.k * r_squared / (2 * focal_length))

        # Apply lens
        field_after_lens = field * lens_phase

        # Propagate if distance specified
        if propagation_distance is not None:
            return self.propagate(field_after_lens, propagation_distance)

        return np.abs(field_after_lens) ** 2

    def create_gaussian_beam(
        self,
        waist: float,
        center: Tuple[float, float] = (0.0, 0.0),
        amplitude: float = 1.0,
    ) -> NDArray:
        """
        Create a Gaussian beam field.

        Args:
            waist: Beam waist (1/e^2 radius) in meters
            center: Beam center coordinates in meters
            amplitude: Peak amplitude

        Returns:
            Complex Gaussian field
        """
        coords = (np.arange(self.grid_size) - self.grid_size // 2) * self.pixel_size
        x, y = np.meshgrid(coords, coords)

        x = x - center[0]
        y = y - center[1]

        r_squared = x**2 + y**2
        field = amplitude * np.exp(-r_squared / waist**2)

        return field.astype(np.complex128)

    def create_hermite_gaussian(
        self,
        waist: float,
        m: int,
        n: int,
        center: Tuple[float, float] = (0.0, 0.0),
    ) -> NDArray:
        """
        Create a Hermite-Gaussian mode field.

        Args:
            waist: Beam waist in meters
            m: Mode index in x direction
            n: Mode index in y direction
            center: Beam center coordinates

        Returns:
            Complex HG mode field
        """
        from scipy.special import hermite

        coords = (np.arange(self.grid_size) - self.grid_size // 2) * self.pixel_size
        x, y = np.meshgrid(coords, coords)

        x = x - center[0]
        y = y - center[1]

        # Normalized coordinates
        x_norm = np.sqrt(2) * x / waist
        y_norm = np.sqrt(2) * y / waist

        # Hermite polynomials
        hm = hermite(m)(x_norm)
        hn = hermite(n)(y_norm)

        # Gaussian envelope
        r_squared = x**2 + y**2
        gaussian = np.exp(-r_squared / waist**2)

        # Normalization
        import math
        norm = 1.0 / np.sqrt(2 ** (m + n) * math.factorial(m) * math.factorial(n))

        field = norm * hm * hn * gaussian

        return field.astype(np.complex128)

    def cavity_round_trip(
        self,
        field: NDArray,
        cavity_length: float,
        mirror_radii: Tuple[float, float],
        num_trips: int = 1,
    ) -> NDArray:
        """
        Simulate cavity round-trip propagation.

        Args:
            field: Input field
            cavity_length: Cavity length in meters
            mirror_radii: (R1, R2) mirror radii of curvature
            num_trips: Number of round trips to simulate

        Returns:
            Field after round trips (intensity)
        """
        r1, r2 = mirror_radii

        for _ in range(num_trips):
            # Propagate to second mirror
            field = self.propagate(field, cavity_length, return_intensity=False)

            # Reflect from second mirror (curved)
            field = self.propagate_through_lens(
                field, r2 / 2, propagation_distance=None
            )
            field = np.sqrt(field).astype(np.complex128)  # Convert back to field

            # Propagate back to first mirror
            field = self.propagate(field, cavity_length, return_intensity=False)

            # Reflect from first mirror (curved)
            field = self.propagate_through_lens(
                field, r1 / 2, propagation_distance=None
            )
            field = np.sqrt(field).astype(np.complex128)

        return np.abs(field) ** 2


def propagate_beam(
    field: NDArray,
    distance: float,
    wavelength: float = 1064e-9,
    pixel_size: float = 1e-6,
) -> NDArray:
    """
    Convenience function for beam propagation.

    Args:
        field: Input field (2D array)
        distance: Propagation distance in meters
        wavelength: Light wavelength
        pixel_size: Pixel size in meters

    Returns:
        Propagated intensity pattern
    """
    propagator = FresnelPropagator(
        wavelength=wavelength,
        pixel_size=pixel_size,
        grid_size=field.shape[0],
    )
    return propagator.propagate(field, distance)
