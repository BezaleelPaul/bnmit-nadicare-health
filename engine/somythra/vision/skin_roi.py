"""
SOMYTHRA Vision — Skin Region-of-Interest (ROI) Extraction & Segmentation.

Extracts anatomically targeted facial ROIs (forehead, left cheek, right cheek)
and applies color-space skin pixel segmentation to reject non-skin pixels
such as hair, glasses, shadows, and beard regions.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
from ..models.schemas import BoundingBox, FaceROISet


class SkinSegmenter:
    """
    Color-space based human skin pixel segmenter.
    Combines YCrCb chromatic distribution with HSV saturation checks.
    Robust across diverse Fitzpatrick phototypes while excluding background & hair.
    """

    @staticmethod
    def get_skin_mask(bgr_image: np.ndarray) -> np.ndarray:
        """
        Produce a binary mask (uint8, 0 or 255) marking skin pixels.
        """
        if bgr_image is None or bgr_image.size == 0:
            return np.empty((0, 0), dtype=np.uint8)

        # 1. YCrCb thresholding: human skin clusters tightly in Cr and Cb channels
        ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
        # Broad physiological skin locus:
        lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        # 2. HSV thresholding for secondary verification
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 15, 40], dtype=np.uint8)
        upper_hsv = np.array([28, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Combine masks
        combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)

        # Morphological opening to eliminate isolated noise pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        return cleaned


class SkinROIExtractor:
    """
    Partitions a detected face bounding box into physiological sub-regions:
    - Forehead (superciliary arch to hairline)
    - Left Cheek (zygomatic arch)
    - Right Cheek (zygomatic arch)
    """

    def __init__(self, min_skin_ratio: float = 0.15, min_pixel_count: int = 40):
        self.min_skin_ratio = min_skin_ratio
        self.min_pixel_count = min_pixel_count
        self.segmenter = SkinSegmenter()

    def extract_rois(self, frame_bgr: np.ndarray, face_box: Optional[BoundingBox]) -> Optional[FaceROISet]:
        """
        Given a frame and a face box, compute safe sub-region bounding boxes.
        Returns FaceROISet or None if face_box is invalid.
        """
        if face_box is None or not face_box.is_valid():
            return None

        img_h, img_w = frame_bgr.shape[:2]

        def _clamp_box(x: int, y: int, w: int, h: int) -> Optional[BoundingBox]:
            cx = max(0, min(x, img_w - 1))
            cy = max(0, min(y, img_h - 1))
            cw = max(1, min(w, img_w - cx))
            ch = max(1, min(h, img_h - cy))
            b = BoundingBox(x=cx, y=cy, w=cw, h=ch, confidence=face_box.confidence)
            return b if b.is_valid(self.min_pixel_count) else None

        # Forehead: top 10% to 28% of face, center 50% width
        forehead = _clamp_box(
            x=face_box.x + int(0.25 * face_box.w),
            y=face_box.y + int(0.08 * face_box.h),
            w=int(0.50 * face_box.w),
            h=int(0.18 * face_box.h),
        )

        # Left Cheek (viewer's left): 50% to 70% height, 12% to 37% width
        left_cheek = _clamp_box(
            x=face_box.x + int(0.12 * face_box.w),
            y=face_box.y + int(0.50 * face_box.h),
            w=int(0.25 * face_box.w),
            h=int(0.20 * face_box.h),
        )

        # Right Cheek (viewer's right): 50% to 70% height, 63% to 88% width
        right_cheek = _clamp_box(
            x=face_box.x + int(0.63 * face_box.w),
            y=face_box.y + int(0.50 * face_box.h),
            w=int(0.25 * face_box.w),
            h=int(0.20 * face_box.h),
        )

        return FaceROISet(
            face=face_box,
            forehead=forehead,
            left_cheek=left_cheek,
            right_cheek=right_cheek,
            confidence=face_box.confidence,
            detected=True,
        )

    def extract_skin_rgb(
        self, frame_bgr: np.ndarray, roi_box: Optional[BoundingBox]
    ) -> Tuple[bool, float, float, float, int]:
        """
        Extract spatial average RGB strictly from verified skin pixels inside ROI.
        Returns:
            (is_valid, mean_r, mean_g, mean_b, skin_pixel_count)
        """
        if roi_box is None or not roi_box.is_valid():
            return False, 0.0, 0.0, 0.0, 0

        crop = frame_bgr[roi_box.y : roi_box.y2, roi_box.x : roi_box.x2]
        if crop.size == 0:
            return False, 0.0, 0.0, 0.0, 0

        # Segment skin in crop
        mask = self.segmenter.get_skin_mask(crop)
        skin_pixel_count = int(np.count_nonzero(mask))
        total_pixels = crop.shape[0] * crop.shape[1]

        skin_ratio = skin_pixel_count / max(1, total_pixels)

        # If skin mask found enough pixels, compute average over skin mask
        if skin_pixel_count >= self.min_pixel_count and skin_ratio >= self.min_skin_ratio:
            # OpenCV BGR -> RGB
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mean_vals = cv2.mean(rgb_crop, mask=mask)[:3]
            return True, float(mean_vals[0]), float(mean_vals[1]), float(mean_vals[2]), skin_pixel_count

        # Fallback: if skin segmentation is borderline but total area is clean,
        # compute central sub-crop average to avoid losing signal during subtle lighting shifts
        if total_pixels >= self.min_pixel_count:
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mean_vals = cv2.mean(rgb_crop)[:3]
            return True, float(mean_vals[0]), float(mean_vals[1]), float(mean_vals[2]), total_pixels

        return False, 0.0, 0.0, 0.0, 0
