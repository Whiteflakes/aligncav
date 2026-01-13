"""
Motor Controller Interface.

This module provides interfaces for controlling stepper motors
used in cavity mirror alignment.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class MotorConfig:
    """Configuration for motor controller."""

    num_motors: int = 4
    steps_per_revolution: int = 200
    microsteps: int = 16
    max_speed: float = 1000  # steps per second
    acceleration: float = 500  # steps per second^2
    step_size_um: float = 0.1  # micrometers per step


class MotorController(ABC):
    """
    Abstract base class for motor controllers.

    Defines the interface for controlling alignment motors.
    """

    def __init__(self, config: Optional[MotorConfig] = None):
        """
        Initialize motor controller.

        Args:
            config: Motor configuration
        """
        self.config = config or MotorConfig()
        self._positions = np.zeros(self.config.num_motors, dtype=np.int64)

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the motor controller.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the motor controller."""
        pass

    @abstractmethod
    def move_relative(self, motor_id: int, steps: int) -> bool:
        """
        Move motor by relative number of steps.

        Args:
            motor_id: Motor index (0 to num_motors-1)
            steps: Number of steps (positive or negative)

        Returns:
            True if move successful
        """
        pass

    @abstractmethod
    def move_absolute(self, motor_id: int, position: int) -> bool:
        """
        Move motor to absolute position.

        Args:
            motor_id: Motor index
            position: Target position in steps

        Returns:
            True if move successful
        """
        pass

    def move_all(self, steps: NDArray) -> bool:
        """
        Move all motors simultaneously.

        Args:
            steps: Array of steps for each motor

        Returns:
            True if all moves successful
        """
        success = True
        for motor_id, step_count in enumerate(steps):
            if step_count != 0:
                if not self.move_relative(motor_id, int(step_count)):
                    success = False
        return success

    def get_position(self, motor_id: int) -> int:
        """Get current position of motor."""
        return int(self._positions[motor_id])

    def get_all_positions(self) -> NDArray:
        """Get positions of all motors."""
        return self._positions.copy()

    def home(self, motor_id: Optional[int] = None) -> bool:
        """
        Home motor(s) to zero position.

        Args:
            motor_id: Motor to home (None for all)

        Returns:
            True if homing successful
        """
        if motor_id is not None:
            return self.move_absolute(motor_id, 0)
        else:
            success = True
            for i in range(self.config.num_motors):
                if not self.move_absolute(i, 0):
                    success = False
            return success

    @property
    def is_connected(self) -> bool:
        """Check if controller is connected."""
        return False


class ArduinoMotorController(MotorController):
    """
    Motor controller using Arduino with stepper drivers.

    Communicates via serial port with custom firmware.
    """

    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 115200,
        config: Optional[MotorConfig] = None,
    ):
        """
        Initialize Arduino motor controller.

        Args:
            port: Serial port name
            baudrate: Serial communication speed
            config: Motor configuration
        """
        super().__init__(config)
        self.port = port
        self.baudrate = baudrate
        self._serial = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to Arduino."""
        try:
            import serial

            self._serial = serial.Serial(
                self.port,
                self.baudrate,
                timeout=1,
            )
            time.sleep(2)  # Wait for Arduino reset

            # Verify connection
            self._serial.write(b"PING\n")
            response = self._serial.readline().decode().strip()

            if response == "PONG":
                self._connected = True
                logger.info(f"Connected to Arduino on {self.port}")
                return True
            else:
                logger.error(f"Unexpected response: {response}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Arduino."""
        if self._serial:
            self._serial.close()
            self._serial = None
        self._connected = False
        logger.info("Disconnected from Arduino")

    def move_relative(self, motor_id: int, steps: int) -> bool:
        """Move motor by relative steps."""
        if not self._connected or not self._serial:
            logger.error("Not connected to Arduino")
            return False

        try:
            cmd = f"MOVE {motor_id} {steps}\n"
            self._serial.write(cmd.encode())

            response = self._serial.readline().decode().strip()
            if response == "OK":
                self._positions[motor_id] += steps
                return True
            else:
                logger.error(f"Move failed: {response}")
                return False

        except Exception as e:
            logger.error(f"Move error: {e}")
            return False

    def move_absolute(self, motor_id: int, position: int) -> bool:
        """Move motor to absolute position."""
        current = self._positions[motor_id]
        relative = position - current
        return self.move_relative(motor_id, relative)

    @property
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected


class SimulatedMotorController(MotorController):
    """
    Simulated motor controller for testing.

    Mimics real motor behavior without hardware.
    """

    def __init__(
        self,
        config: Optional[MotorConfig] = None,
        move_delay: float = 0.01,
        failure_rate: float = 0.0,
    ):
        """
        Initialize simulated controller.

        Args:
            config: Motor configuration
            move_delay: Simulated delay per move
            failure_rate: Probability of move failure
        """
        super().__init__(config)
        self.move_delay = move_delay
        self.failure_rate = failure_rate
        self._connected = False

    def connect(self) -> bool:
        """Simulate connection."""
        self._connected = True
        logger.info("Connected to simulated motor controller")
        return True

    def disconnect(self) -> None:
        """Simulate disconnection."""
        self._connected = False
        logger.info("Disconnected from simulated motor controller")

    def move_relative(self, motor_id: int, steps: int) -> bool:
        """Simulate relative move."""
        if not self._connected:
            return False

        # Simulate occasional failures
        if np.random.random() < self.failure_rate:
            logger.warning(f"Simulated move failure for motor {motor_id}")
            return False

        # Simulate move time
        time.sleep(self.move_delay * abs(steps) / 100)

        self._positions[motor_id] += steps
        return True

    def move_absolute(self, motor_id: int, position: int) -> bool:
        """Simulate absolute move."""
        current = self._positions[motor_id]
        relative = position - current
        return self.move_relative(motor_id, relative)

    @property
    def is_connected(self) -> bool:
        """Check simulated connection."""
        return self._connected


def create_motor_controller(
    controller_type: str = "simulated",
    **kwargs,
) -> MotorController:
    """
    Factory function to create motor controllers.

    Args:
        controller_type: Type of controller ('arduino', 'simulated')
        **kwargs: Additional arguments for the controller

    Returns:
        MotorController instance
    """
    if controller_type == "arduino":
        return ArduinoMotorController(**kwargs)
    elif controller_type == "simulated":
        return SimulatedMotorController(**kwargs)
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
