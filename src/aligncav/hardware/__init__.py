"""
Hardware interfaces for AlignCav.

Provides interfaces for motors, cameras, and power meters
used in the alignment system.
"""

from .motor_controller import (
    ArduinoMotorController,
    MotorConfig,
    MotorController,
    SimulatedMotorController,
    create_motor_controller,
)
from .power_meter import (
    PowerMeter,
    PowerMeterConfig,
    SimulatedPowerMeter,
    ThorlabsPM100A,
    create_power_meter,
)
from .video_stream import (
    HTTPVideoStream,
    OpenCVVideoStream,
    SimulatedVideoStream,
    StreamConfig,
    VideoStream,
    create_video_stream,
)

__all__ = [
    # Motor control
    "MotorController",
    "MotorConfig",
    "ArduinoMotorController",
    "SimulatedMotorController",
    "create_motor_controller",
    # Video stream
    "VideoStream",
    "StreamConfig",
    "OpenCVVideoStream",
    "HTTPVideoStream",
    "SimulatedVideoStream",
    "create_video_stream",
    # Power meter
    "PowerMeter",
    "PowerMeterConfig",
    "ThorlabsPM100A",
    "SimulatedPowerMeter",
    "create_power_meter",
]
