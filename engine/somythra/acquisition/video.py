"""
SOMYTHRA Acquisition — Prerecorded Video File Stream.

Provides deterministic, reproducible frame decoding from video files.
Used extensively for pipeline benchmarking and automated unit/integration tests.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoStreamConfig:
    file_path: Union[str, Path]
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    max_frames: Optional[int] = None
    start_frame: int = 0
    loop: bool = False


class VideoCaptureStream:
    """
    Decodes video frames with frame-accurate timestamps derived from native video FPS.
    Supports iteration, length discovery, resizing, and context management.
    """

    def __init__(self, config: VideoStreamConfig):
        self.config = config
        self.file_path = Path(config.file_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 30.0
        self._total_frames: int = 0
        self._current_frame_idx: int = 0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open(self) -> bool:
        """Open the target video file and extract container metadata."""
        if not self.file_path.exists():
            logger.error("Video file not found: %s", self.file_path)
            return False

        self._cap = cv2.VideoCapture(str(self.file_path))
        if not self._cap.isOpened():
            logger.error("Failed to decode video file: %s", self.file_path)
            return False

        native_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = native_fps if native_fps > 0 else 30.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.config.start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.config.start_frame)
            self._current_frame_idx = self.config.start_frame
        else:
            self._current_frame_idx = 0

        logger.info(
            "Opened video: %s (%d frames @ %.2f FPS)",
            self.file_path.name,
            self._total_frames,
            self._fps,
        )
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Fetch the next video frame.
        Returns:
            (success, frame_bgr, timestamp_seconds)
        """
        if not self.is_open or self._cap is None:
            return False, None, 0.0

        if (
            self.config.max_frames is not None
            and self._current_frame_idx >= (self.config.start_frame + self.config.max_frames)
        ):
            return False, None, self._current_frame_idx / self._fps

        ret, frame = self._cap.read()
        if not ret or frame is None:
            if self.config.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.config.start_frame)
                self._current_frame_idx = self.config.start_frame
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    return False, None, 0.0
            else:
                return False, None, self._current_frame_idx / self._fps

        timestamp = self._current_frame_idx / self._fps
        self._current_frame_idx += 1

        if self.config.target_width and self.config.target_height:
            frame = cv2.resize(
                frame,
                (self.config.target_width, self.config.target_height),
                interpolation=cv2.INTER_AREA,
            )

        return True, frame, timestamp

    def release(self) -> None:
        """Release underlying OpenCV VideoCapture resource."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            finally:
                self._cap = None

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float]]:
        """Convenient generator yielding (frame, timestamp) until stream completion."""
        while True:
            ret, frame, ts = self.read()
            if not ret or frame is None:
                break
            yield frame, ts

    def __enter__(self) -> "VideoCaptureStream":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
