"""
SOMYTHRA Signals — Preprocessing, Resampling & Detrending.

Handles timestamp jitter, uniform interpolation, moving-average baseline removal,
and per-channel temporal normalization.
"""

from dataclasses import dataclass
import logging
from typing import Tuple
import numpy as np
from scipy.interpolate import interp1d
from ..models.schemas import TemporalRGBSignal

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    target_sampling_rate_hz: float = 30.0
    resample_uniform: bool = True
    detrend: bool = True
    detrend_window_sec: float = 1.0  # Window for moving baseline
    normalize_temporal: bool = True  # C_n = C / mean(C)
    motion_threshold_zscore: float = 3.5


class SignalPreprocessor:
    """
    Standardizes raw RGB optical time-series:
    1. Inter-frame timestamp jitter analysis & uniform resampling
    2. Motion disturbance spike detection
    3. Moving-average baseline detrending
    4. Temporal division normalization
    """

    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()

    def process(self, raw_signal: TemporalRGBSignal) -> Tuple[TemporalRGBSignal, float, float]:
        """
        Preprocess the raw multi-channel RGB signal.
        Returns:
            (preprocessed_signal, jitter_std_ms, motion_score)
        """
        ts = raw_signal.timestamps.copy()
        r = raw_signal.r.copy()
        g = raw_signal.g.copy()
        b = raw_signal.b.copy()

        n = len(ts)
        if n < 4:
            return raw_signal, 0.0, 0.0

        # 1. Analyze timestamp jitter
        dt = np.diff(ts)
        dt_positive = dt[dt > 0]
        jitter_std_ms = float(np.std(dt_positive) * 1000.0) if len(dt_positive) > 1 else 0.0

        # 2. Resample to uniform grid if requested
        target_fs = self.config.target_sampling_rate_hz
        if self.config.resample_uniform and (ts[-1] - ts[0]) > 0:
            if abs(target_fs - raw_signal.sampling_rate_hz) < 0.5:
                num_samples = len(ts)
            else:
                num_samples = max(4, int(np.round((ts[-1] - ts[0]) * target_fs)) + 1)

            if num_samples >= 4:
                uniform_ts = np.linspace(ts[0], ts[-1], num_samples)
                # Linear interpolation is safe and robust against edge oscillations
                interp_r = interp1d(ts, r, kind="linear", fill_value="extrapolate")
                interp_g = interp1d(ts, g, kind="linear", fill_value="extrapolate")
                interp_b = interp1d(ts, b, kind="linear", fill_value="extrapolate")

                ts = uniform_ts
                r = interp_r(uniform_ts)
                g = interp_g(uniform_ts)
                b = interp_b(uniform_ts)
                fs = target_fs
            else:
                fs = raw_signal.sampling_rate_hz
        else:
            fs = raw_signal.sampling_rate_hz

        # 3. Detect motion disturbance score
        diff_g = np.diff(g)
        std_diff = np.std(diff_g) if len(diff_g) > 1 else 1.0
        z_scores = np.abs(diff_g - np.mean(diff_g)) / max(1e-6, std_diff)
        motion_score = float(np.mean(z_scores > self.config.motion_threshold_zscore))

        # 4. Temporal normalization: C_n(t) = C(t) / mean(C)
        if self.config.normalize_temporal:
            mean_r = np.mean(r) if np.mean(r) != 0 else 1.0
            mean_g = np.mean(g) if np.mean(g) != 0 else 1.0
            mean_b = np.mean(b) if np.mean(b) != 0 else 1.0

            r = r / mean_r
            g = g / mean_g
            b = b / mean_b

        # 5. Baseline detrending
        if self.config.detrend:
            r = self._detrend_moving_average(r, fs, self.config.detrend_window_sec)
            g = self._detrend_moving_average(g, fs, self.config.detrend_window_sec)
            b = self._detrend_moving_average(b, fs, self.config.detrend_window_sec)

        preprocessed = TemporalRGBSignal(
            timestamps=ts,
            r=r,
            g=g,
            b=b,
            roi_name=raw_signal.roi_name,
            sampling_rate_hz=fs,
        )
        return preprocessed, jitter_std_ms, motion_score

    @staticmethod
    def _detrend_moving_average(signal: np.ndarray, fs: float, window_sec: float) -> np.ndarray:
        """Subtract moving average baseline to remove low-frequency DC drift."""
        window_size = max(3, int(np.round(window_sec * fs)))
        if window_size % 2 == 0:
            window_size += 1

        if len(signal) <= window_size:
            # If signal is too short for window, simple mean subtraction
            return signal - np.mean(signal)

        # Uniform moving average filter
        kernel = np.ones(window_size) / window_size
        # Pad symmetrically at boundaries
        pad_width = window_size // 2
        padded = np.pad(signal, pad_width, mode="edge")
        trend = np.convolve(padded, kernel, mode="valid")

        # Ensure exact shape alignment
        if len(trend) > len(signal):
            trend = trend[: len(signal)]

        return signal - trend
