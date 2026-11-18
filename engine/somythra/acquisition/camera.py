"""
SOMYTHRA Acquisition — Live Camera Capture Module.

Provides safe, timestamped, resource-managed camera frame acquisition.
Gracefully fails if no camera is connected or permissions are denied.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    device_index: int = 0
    target_width: int = 640
    target_height: int = 480
    target_fps: float = 30.0
    buffer_size: int = 1
    lock_exposure: bool = False
    exposure_value: Optional[float] = None
    lock_white_balance: bool = False


class CameraCapture:
    """
    Safely captures frames from a local video device index with sub-millisecond timestamps.
    Implements Python context manager protocol for guaranteed resource cleanup.
    """

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._start_time: float = 0.0
        self._last_timestamp: float = 0.0
        self._frame_count: int = 0
        self._dropped_frames: int = 0
        self._is_open: bool = False

    @property
    def is_open(self) -> bool:
        return self._is_open and self._cap is not None and self._cap.isOpened()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    @property
    def average_fps(self) -> float:
        if self._frame_count <= 1 or self._start_time <= 0:
            return 0.0
        elapsed = time.perf_counter() - self._start_time
        return self._frame_count / max(1e-4, elapsed)

    def open(self) -> bool:
        """Attempt to open camera device index gracefully without unhandled exceptions."""
        try:
            logger.info("Opening camera device %d...", self.config.device_index)
            self._cap = cv2.VideoCapture(self.config.device_index)

            if not self._cap.isOpened():
                logger.warning(
                    "Camera device %d could not be opened. Verify permissions or device presence.",
                    self.config.device_index,
                )
                self._is_open = False
                return False

            # Set hardware requested dimensions and buffer size
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.target_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.target_height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

            # Hardware-level Auto-Exposure (AE) and Auto-White-Balance (AWB) locking
            if self.config.lock_exposure:
                try:
                    # 0.25 (or 1) sets manual mode on V4L2/UVC webcam backends
                    self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    if self.config.exposure_value is not None:
                        self._cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure_value)
                    logger.info("Applied hardware exposure lock.")
                except Exception as ex:
                    logger.warning("Could not set hardware exposure lock: %s", ex)

            if self.config.lock_white_balance:
                try:
                    self._cap.set(cv2.CAP_PROP_AUTO_WB_TEMPERATURE, 0)
                    logger.info("Applied hardware white-balance lock.")
                except Exception as ex:
                    logger.warning("Could not set hardware white-balance lock: %s", ex)

            self._start_time = time.perf_counter()
            self._last_timestamp = self._start_time
            self._frame_count = 0
            self._dropped_frames = 0
            self._is_open = True
            return True
        except Exception as e:
            logger.error("Exception occurred while opening camera device %d: %s", self.config.device_index, e)
            self._is_open = False
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Capture the next frame.
        Returns:
            (success, frame_bgr, relative_timestamp_sec)
        """
        if not self.is_open or self._cap is None:
            return False, None, 0.0

        now = time.perf_counter()
        ret, frame = self._cap.read()

        if not ret or frame is None:
            self._dropped_frames += 1
            return False, None, now - self._start_time

        self._frame_count += 1
        rel_time = now - self._start_time

        # Resize if dimensions differ from target
        h, w = frame.shape[:2]
        if (w, h) != (self.config.target_width, self.config.target_height):
            frame = cv2.resize(
                frame,
                (self.config.target_width, self.config.target_height),
                interpolation=cv2.INTER_AREA,
            )

        self._last_timestamp = now
        return True, frame, rel_time

    def release(self) -> None:
        """Safely release the camera hardware handle."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning("Error releasing camera handle: %s", e)
            finally:
                self._cap = None
                self._is_open = False
                logger.info("Camera device released cleanly.")

    def __enter__(self) -> "CameraCapture":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
