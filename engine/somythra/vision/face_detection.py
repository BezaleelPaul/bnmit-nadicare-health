"""
SOMYTHRA Vision — Face Detection & Tracking Module.

Provides modular, robust face detection with temporal bounding box smoothing
and graceful handling of temporary occlusions.
"""

from abc import ABC, abstractmethod
import logging
from typing import Optional, List
import cv2
import numpy as np
from ..models.schemas import BoundingBox

logger = logging.getLogger(__name__)


class BaseFaceDetector(ABC):
    """Abstract interface for face detection backends."""

    @abstractmethod
    def detect_face(self, frame_bgr: np.ndarray) -> Optional[BoundingBox]:
        """Detect the primary face bounding box in the given BGR frame."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracking and smoothing state."""
        pass


class HaarCascadeFaceDetector(BaseFaceDetector):
    """
    OpenCV Haar Cascade Face Detector.
    Zero external network dependencies, high portability, fast execution.
    Includes temporal exponential smoothing and occlusion coasting.
    """

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (60, 60),
        smoothing_factor: float = 0.7,
        max_missing_frames: int = 4,
    ):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.smoothing_factor = smoothing_factor
        self.max_missing_frames = max_missing_frames

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from: {cascade_path}")

        self._prev_bbox: Optional[BoundingBox] = None
        self._missing_count: int = 0

    def detect_face(self, frame_bgr: np.ndarray) -> Optional[BoundingBox]:
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        # Convert to grayscale for Haar detection
        if len(frame_bgr.shape) == 3:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr

        # Equalize histogram to improve contrast in variable lighting
        gray = cv2.equalizeHist(gray)

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            self._missing_count += 1
            if self._prev_bbox is not None and self._missing_count <= self.max_missing_frames:
                # Coast using last known position during momentary blink or occlusion
                return self._prev_bbox
            self._prev_bbox = None
            return None

        self._missing_count = 0

        # Choose the largest detected face (closest to camera)
        best_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = (int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3]))

        # Constrain to frame boundaries
        img_h, img_w = frame_bgr.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))

        raw_box = BoundingBox(x=x, y=y, w=w, h=h, confidence=0.9)

        # Apply temporal exponential smoothing to stabilize boundary jitter
        if self._prev_bbox is not None and self.smoothing_factor > 0.0:
            alpha = self.smoothing_factor
            smooth_x = int(alpha * self._prev_bbox.x + (1 - alpha) * raw_box.x)
            smooth_y = int(alpha * self._prev_bbox.y + (1 - alpha) * raw_box.y)
            smooth_w = int(alpha * self._prev_bbox.w + (1 - alpha) * raw_box.w)
            smooth_h = int(alpha * self._prev_bbox.h + (1 - alpha) * raw_box.h)
            smoothed_box = BoundingBox(
                x=smooth_x,
                y=smooth_y,
                w=smooth_w,
                h=smooth_h,
                confidence=raw_box.confidence,
            )
            self._prev_bbox = smoothed_box
            return smoothed_box

        self._prev_bbox = raw_box
        return raw_box

    def reset(self) -> None:
        self._prev_bbox = None
        self._missing_count = 0


def get_default_face_detector() -> BaseFaceDetector:
    """Factory function returning the default validated face detector."""
    return HaarCascadeFaceDetector()
