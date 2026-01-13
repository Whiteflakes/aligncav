"""
Video Stream Interface.

This module provides interfaces for capturing video frames
from cameras for beam profile monitoring.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for video stream."""

    width: int = 640
    height: int = 480
    fps: float = 30.0
    exposure: float = 0.01  # seconds
    gain: float = 1.0


class VideoStream(ABC):
    """
    Abstract base class for video streams.

    Defines interface for capturing frames from various sources.
    """

    def __init__(self, config: Optional[StreamConfig] = None):
        """
        Initialize video stream.

        Args:
            config: Stream configuration
        """
        self.config = config or StreamConfig()
        self._running = False

    @abstractmethod
    def start(self) -> bool:
        """
        Start the video stream.

        Returns:
            True if started successfully
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the video stream."""
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[NDArray]]:
        """
        Read a frame from the stream.

        Returns:
            Tuple of (success, frame)
        """
        pass

    def read_gray(self) -> Tuple[bool, Optional[NDArray]]:
        """
        Read a grayscale frame.

        Returns:
            Tuple of (success, grayscale_frame)
        """
        success, frame = self.read()
        if not success or frame is None:
            return False, None

        if len(frame.shape) == 3:
            import cv2

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return True, frame

    @property
    def is_running(self) -> bool:
        """Check if stream is running."""
        return self._running

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get frame dimensions (width, height)."""
        return (self.config.width, self.config.height)


class OpenCVVideoStream(VideoStream):
    """
    Video stream using OpenCV VideoCapture.

    Supports USB cameras, built-in webcams, and IP cameras.
    """

    def __init__(
        self,
        source: int | str = 0,
        config: Optional[StreamConfig] = None,
    ):
        """
        Initialize OpenCV video stream.

        Args:
            source: Camera index or URL
            config: Stream configuration
        """
        super().__init__(config)
        self.source = source
        self._cap = None

    def start(self) -> bool:
        """Start video capture."""
        try:
            import cv2

            self._cap = cv2.VideoCapture(self.source)

            if not self._cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False

            # Configure camera
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            self._running = True
            logger.info(f"Started video stream from {self.source}")
            return True

        except Exception as e:
            logger.error(f"Error starting video stream: {e}")
            return False

    def stop(self) -> None:
        """Stop video capture."""
        if self._cap:
            self._cap.release()
            self._cap = None
        self._running = False
        logger.info("Stopped video stream")

    def read(self) -> Tuple[bool, Optional[NDArray]]:
        """Read a frame."""
        if not self._cap:
            return False, None

        ret, frame = self._cap.read()
        return ret, frame if ret else None


class HTTPVideoStream(VideoStream):
    """
    Video stream from HTTP server (e.g., Raspberry Pi camera server).

    Fetches JPEG frames from an HTTP endpoint.
    """

    def __init__(
        self,
        url: str = "http://localhost:8080/frame",
        config: Optional[StreamConfig] = None,
        timeout: float = 5.0,
    ):
        """
        Initialize HTTP video stream.

        Args:
            url: URL to fetch frames from
            config: Stream configuration
            timeout: Request timeout in seconds
        """
        super().__init__(config)
        self.url = url
        self.timeout = timeout

    def start(self) -> bool:
        """Start HTTP stream."""
        try:
            import requests

            # Test connection
            response = requests.get(self.url, timeout=self.timeout)
            if response.status_code == 200:
                self._running = True
                logger.info(f"Started HTTP video stream from {self.url}")
                return True
            else:
                logger.error(f"HTTP error: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to HTTP stream: {e}")
            return False

    def stop(self) -> None:
        """Stop HTTP stream."""
        self._running = False
        logger.info("Stopped HTTP video stream")

    def read(self) -> Tuple[bool, Optional[NDArray]]:
        """Read a frame from HTTP endpoint."""
        if not self._running:
            return False, None

        try:
            import cv2
            import requests

            response = requests.get(self.url, timeout=self.timeout)
            if response.status_code != 200:
                return False, None

            # Decode JPEG
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            return True, frame

        except Exception as e:
            logger.error(f"Error reading HTTP frame: {e}")
            return False, None


class SimulatedVideoStream(VideoStream):
    """
    Simulated video stream for testing.

    Generates synthetic beam images.
    """

    def __init__(
        self,
        config: Optional[StreamConfig] = None,
        noise_level: float = 0.05,
    ):
        """
        Initialize simulated stream.

        Args:
            config: Stream configuration
            noise_level: Amount of noise to add
        """
        super().__init__(config)
        self.noise_level = noise_level
        self._frame_count = 0

    def start(self) -> bool:
        """Start simulated stream."""
        self._running = True
        self._frame_count = 0
        logger.info("Started simulated video stream")
        return True

    def stop(self) -> None:
        """Stop simulated stream."""
        self._running = False
        logger.info("Stopped simulated video stream")

    def read(self) -> Tuple[bool, Optional[NDArray]]:
        """Generate a simulated frame."""
        if not self._running:
            return False, None

        # Generate Gaussian beam pattern
        y, x = np.ogrid[: self.config.height, : self.config.width]
        cx, cy = self.config.width // 2, self.config.height // 2

        # Add some drift
        drift_x = 10 * np.sin(self._frame_count * 0.01)
        drift_y = 10 * np.cos(self._frame_count * 0.01)

        r_squared = (x - cx - drift_x) ** 2 + (y - cy - drift_y) ** 2
        sigma = min(self.config.width, self.config.height) / 6

        frame = np.exp(-r_squared / (2 * sigma**2))

        # Add noise
        noise = np.random.normal(0, self.noise_level, frame.shape)
        frame = np.clip(frame + noise, 0, 1)

        # Convert to uint8
        frame = (frame * 255).astype(np.uint8)

        # Convert to BGR for consistency
        frame = np.stack([frame, frame, frame], axis=-1)

        self._frame_count += 1
        return True, frame


def create_video_stream(
    stream_type: str = "simulated",
    **kwargs,
) -> VideoStream:
    """
    Factory function to create video streams.

    Args:
        stream_type: Type of stream ('opencv', 'http', 'simulated')
        **kwargs: Additional arguments for the stream

    Returns:
        VideoStream instance
    """
    if stream_type == "opencv":
        return OpenCVVideoStream(**kwargs)
    elif stream_type == "http":
        return HTTPVideoStream(**kwargs)
    elif stream_type == "simulated":
        return SimulatedVideoStream(**kwargs)
    else:
        raise ValueError(f"Unknown stream type: {stream_type}")
