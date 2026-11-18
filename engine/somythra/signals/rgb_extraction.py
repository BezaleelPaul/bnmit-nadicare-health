"""
SOMYTHRA Signals — Temporal RGB Signal Extraction & Accumulation.

Collects spatial skin color measurements over time for distinct facial sub-regions
and packages them into validated TemporalRGBSignal instances.
"""

from typing import Dict, List, Optional
import numpy as np
from ..models.schemas import FrameRGBMeasurement, TemporalRGBSignal


class RGBSignalAccumulator:
    """
    Accumulator buffer storing temporal RGB measurements across frames.
    Maintains separate buffers for forehead, left cheek, right cheek, and combined.
    """

    def __init__(self, expected_fps: float = 30.0, max_buffer_seconds: float = 60.0):
        self.expected_fps = expected_fps
        self.max_buffer_samples = int(expected_fps * max_buffer_seconds)

        self._buffers: Dict[str, List[FrameRGBMeasurement]] = {
            "forehead": [],
            "left_cheek": [],
            "right_cheek": [],
            "combined": [],
        }

    def add_frame_sample(self, sample: FrameRGBMeasurement) -> None:
        """Add a measurement sample to its respective region buffer."""
        roi = sample.roi_name if sample.roi_name in self._buffers else "combined"
        buf = self._buffers[roi]
        buf.append(sample)
        if len(buf) > self.max_buffer_samples:
            buf.pop(0)

    def add_frame_rois(
        self,
        timestamp: float,
        forehead_rgb: Optional[tuple[float, float, float, int]] = None,
        left_cheek_rgb: Optional[tuple[float, float, float, int]] = None,
        right_cheek_rgb: Optional[tuple[float, float, float, int]] = None,
    ) -> None:
        """
        Record measurements for all available ROIs in a frame and compute weighted combined skin signal.
        """
        comb_r = 0.0
        comb_g = 0.0
        comb_b = 0.0
        total_pixels = 0

        if forehead_rgb and forehead_rgb[0]:
            r, g, b, cnt = forehead_rgb[1], forehead_rgb[2], forehead_rgb[3], forehead_rgb[4]
            self.add_frame_sample(
                FrameRGBMeasurement(
                    timestamp=timestamp,
                    r=r,
                    g=g,
                    b=b,
                    roi_name="forehead",
                    skin_pixel_count=cnt,
                    is_valid=True,
                )
            )
            comb_r += r * cnt
            comb_g += g * cnt
            comb_b += b * cnt
            total_pixels += cnt

        if left_cheek_rgb and left_cheek_rgb[0]:
            r, g, b, cnt = left_cheek_rgb[1], left_cheek_rgb[2], left_cheek_rgb[3], left_cheek_rgb[4]
            self.add_frame_sample(
                FrameRGBMeasurement(
                    timestamp=timestamp,
                    r=r,
                    g=g,
                    b=b,
                    roi_name="left_cheek",
                    skin_pixel_count=cnt,
                    is_valid=True,
                )
            )
            comb_r += r * cnt
            comb_g += g * cnt
            comb_b += b * cnt
            total_pixels += cnt

        if right_cheek_rgb and right_cheek_rgb[0]:
            r, g, b, cnt = right_cheek_rgb[1], right_cheek_rgb[2], right_cheek_rgb[3], right_cheek_rgb[4]
            self.add_frame_sample(
                FrameRGBMeasurement(
                    timestamp=timestamp,
                    r=r,
                    g=g,
                    b=b,
                    roi_name="right_cheek",
                    skin_pixel_count=cnt,
                    is_valid=True,
                )
            )
            comb_r += r * cnt
            comb_g += g * cnt
            comb_b += b * cnt
            total_pixels += cnt

        # Add weighted average combined signal
        if total_pixels > 0:
            self.add_frame_sample(
                FrameRGBMeasurement(
                    timestamp=timestamp,
                    r=comb_r / total_pixels,
                    g=comb_g / total_pixels,
                    b=comb_b / total_pixels,
                    roi_name="combined",
                    skin_pixel_count=total_pixels,
                    is_valid=True,
                )
            )

    def get_signal(self, roi_name: str = "combined") -> Optional[TemporalRGBSignal]:
        """Convert buffered samples for a region into a contiguous TemporalRGBSignal."""
        buf = self._buffers.get(roi_name, [])
        if len(buf) < 3:
            return None

        # Filter valid samples
        valid_samples = [s for s in buf if s.is_valid]
        if len(valid_samples) < 3:
            return None

        ts = np.array([s.timestamp for s in valid_samples], dtype=np.float64)
        r = np.array([s.r for s in valid_samples], dtype=np.float64)
        g = np.array([s.g for s in valid_samples], dtype=np.float64)
        b = np.array([s.b for s in valid_samples], dtype=np.float64)

        # Estimate empirical sampling rate
        duration = ts[-1] - ts[0]
        fs = (len(ts) - 1) / duration if duration > 0 else self.expected_fps

        return TemporalRGBSignal(
            timestamps=ts,
            r=r,
            g=g,
            b=b,
            roi_name=roi_name,
            sampling_rate_hz=float(fs),
        )

    def clear(self) -> None:
        """Reset all accumulator buffers."""
        for k in self._buffers:
            self._buffers[k].clear()
