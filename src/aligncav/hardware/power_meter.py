"""
Power Meter Interface.

This module provides interfaces for reading power measurements
from optical power meters.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PowerMeterConfig:
    """Configuration for power meter."""

    wavelength: float = 1064e-9  # meters
    auto_range: bool = True
    averaging: int = 1
    timeout: float = 1.0


class PowerMeter(ABC):
    """
    Abstract base class for power meters.

    Defines interface for optical power measurement.
    """

    def __init__(self, config: Optional[PowerMeterConfig] = None):
        """
        Initialize power meter.

        Args:
            config: Power meter configuration
        """
        self.config = config or PowerMeterConfig()
        self._connected = False

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the power meter.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the power meter."""
        pass

    @abstractmethod
    def read_power(self) -> Optional[float]:
        """
        Read current power value.

        Returns:
            Power in watts, or None if read failed
        """
        pass

    def read_averaged(self, num_samples: int = 10) -> Optional[float]:
        """
        Read averaged power.

        Args:
            num_samples: Number of samples to average

        Returns:
            Averaged power in watts
        """
        readings = []
        for _ in range(num_samples):
            power = self.read_power()
            if power is not None:
                readings.append(power)

        if not readings:
            return None

        return float(np.mean(readings))

    @property
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected


class ThorlabsPM100A(PowerMeter):
    """
    Interface for Thorlabs PM100A power meter.

    Uses VISA for communication.
    """

    def __init__(
        self,
        resource_name: str = "USB0::0x1313::0x8078::P0000001::INSTR",
        config: Optional[PowerMeterConfig] = None,
    ):
        """
        Initialize Thorlabs power meter.

        Args:
            resource_name: VISA resource name
            config: Power meter configuration
        """
        super().__init__(config)
        self.resource_name = resource_name
        self._instrument = None

    def connect(self) -> bool:
        """Connect to power meter via VISA."""
        try:
            import pyvisa

            rm = pyvisa.ResourceManager()
            self._instrument = rm.open_resource(self.resource_name)
            self._instrument.timeout = int(self.config.timeout * 1000)

            # Configure meter
            if self.config.wavelength:
                wavelength_nm = self.config.wavelength * 1e9
                self._instrument.write(f"SENS:CORR:WAV {wavelength_nm}")

            if self.config.auto_range:
                self._instrument.write("SENS:POW:RANG:AUTO ON")

            self._connected = True
            logger.info(f"Connected to Thorlabs PM100A: {self.resource_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to power meter: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from power meter."""
        if self._instrument:
            self._instrument.close()
            self._instrument = None
        self._connected = False
        logger.info("Disconnected from power meter")

    def read_power(self) -> Optional[float]:
        """Read power measurement."""
        if not self._connected or not self._instrument:
            return None

        try:
            response = self._instrument.query("MEAS:POW?")
            power = float(response.strip())
            return power

        except Exception as e:
            logger.error(f"Power read error: {e}")
            return None

    def set_wavelength(self, wavelength: float) -> bool:
        """
        Set measurement wavelength.

        Args:
            wavelength: Wavelength in meters

        Returns:
            True if successful
        """
        if not self._connected or not self._instrument:
            return False

        try:
            wavelength_nm = wavelength * 1e9
            self._instrument.write(f"SENS:CORR:WAV {wavelength_nm}")
            self.config.wavelength = wavelength
            return True

        except Exception as e:
            logger.error(f"Failed to set wavelength: {e}")
            return False


class SimulatedPowerMeter(PowerMeter):
    """
    Simulated power meter for testing.

    Returns synthetic power readings.
    """

    def __init__(
        self,
        config: Optional[PowerMeterConfig] = None,
        base_power: float = 1e-3,  # 1 mW
        noise_level: float = 0.01,
    ):
        """
        Initialize simulated power meter.

        Args:
            config: Power meter configuration
            base_power: Base power level in watts
            noise_level: Relative noise level
        """
        super().__init__(config)
        self.base_power = base_power
        self.noise_level = noise_level
        self._current_power = base_power

    def connect(self) -> bool:
        """Simulate connection."""
        self._connected = True
        logger.info("Connected to simulated power meter")
        return True

    def disconnect(self) -> None:
        """Simulate disconnection."""
        self._connected = False
        logger.info("Disconnected from simulated power meter")

    def read_power(self) -> Optional[float]:
        """Return simulated power reading."""
        if not self._connected:
            return None

        # Add noise
        noise = np.random.normal(0, self.noise_level * self._current_power)
        return float(self._current_power + noise)

    def set_power_level(self, power: float) -> None:
        """
        Set the simulated power level.

        Args:
            power: Power level in watts
        """
        self._current_power = power

    def simulate_alignment_change(self, quality: float) -> None:
        """
        Simulate power change due to alignment.

        Args:
            quality: Alignment quality (0 to 1)
        """
        self._current_power = self.base_power * quality


def create_power_meter(
    meter_type: str = "simulated",
    **kwargs,
) -> PowerMeter:
    """
    Factory function to create power meters.

    Args:
        meter_type: Type of meter ('thorlabs', 'simulated')
        **kwargs: Additional arguments for the meter

    Returns:
        PowerMeter instance
    """
    if meter_type == "thorlabs":
        return ThorlabsPM100A(**kwargs)
    elif meter_type == "simulated":
        return SimulatedPowerMeter(**kwargs)
    else:
        raise ValueError(f"Unknown meter type: {meter_type}")
