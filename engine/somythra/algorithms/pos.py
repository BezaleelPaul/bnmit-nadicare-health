"""
SOMYTHRA Algorithms — Plane-Orthogonal-to-Skin (POS) rPPG Algorithm.

Citation:
Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
Algorithmic Principles of Remote PPG.
IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.
"""

import numpy as np
from .base import BaseRPPGAlgorithm
from ..models.schemas import TemporalRGBSignal, PulseWaveform


class POSAlgorithm(BaseRPPGAlgorithm):
    """
    Plane-Orthogonal-to-Skin (POS) pulse extraction algorithm.
    Projects temporally normalized RGB signals onto two orthogonal projection planes
    perpendicular to the skin-tone reflection direction, followed by
    adaptive standard-deviation alpha tuning.
    """

    def __init__(self, window_length_sec: float = 1.6):
        self.window_length_sec = window_length_sec

    @property
    def name(self) -> str:
        return "POS"

    def extract_pulse(self, rgb_signal: TemporalRGBSignal) -> PulseWaveform:
        ts = rgb_signal.timestamps
        r = rgb_signal.r
        g = rgb_signal.g
        b = rgb_signal.b
        fs = rgb_signal.sampling_rate_hz
        n = len(ts)

        if n < 4:
            return PulseWaveform(
                timestamps=ts.copy(),
                signal=np.zeros(n, dtype=np.float64),
                algorithm_name=self.name,
                sampling_rate_hz=fs,
            )

        # Temporal normalization: if signals are raw positive intensities (mean > 0.5),
        # normalize by channel DC level: C_n = C / mean(C).
        # If signals are already AC detrended (mean near 0), use directly without dividing by residual zero.
        mean_r = np.mean(r)
        mean_g = np.mean(g)
        mean_b = np.mean(b)

        if abs(mean_r) > 0.5 and abs(mean_g) > 0.5 and abs(mean_b) > 0.5:
            r_n = r / mean_r
            g_n = g / mean_g
            b_n = b / mean_b
        else:
            r_n = r
            g_n = g
            b_n = b

        # Plane-Orthogonal-to-Skin projections
        s1 = g_n - b_n
        s2 = g_n + b_n - 2.0 * r_n

        std_s1 = np.std(s1)
        std_s2 = np.std(s2)
        alpha = (std_s1 / std_s2) if std_s2 > 1e-6 else 1.0

        pulse_signal = s1 + alpha * s2
        pulse_signal = pulse_signal - np.mean(pulse_signal)

        # Standardize variance
        std_p = np.std(pulse_signal)
        if std_p > 1e-6:
            pulse_signal = pulse_signal / std_p

        return PulseWaveform(
            timestamps=ts.copy(),
            signal=pulse_signal,
            algorithm_name=self.name,
            sampling_rate_hz=fs,
        )
