"""
SOMYTHRA Algorithms — Chrominance-Based (CHROM) rPPG Algorithm.

Citation:
de Haan, G., & Jeanne, V. (2013).
Robust Pulse Rate From Chrominance-Based rPPG.
IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
"""

import numpy as np
from .base import BaseRPPGAlgorithm
from ..models.schemas import TemporalRGBSignal, PulseWaveform


class CHROMAlgorithm(BaseRPPGAlgorithm):
    """
    Chrominance-based (CHROM) pulse extraction algorithm.
    Builds two orthogonal chrominance signals Xs and Ys from temporally normalized RGB,
    then combines them using standard-deviation ratio cancellation to eliminate motion.
    """

    def __init__(self, window_length_sec: float = 1.6):
        self.window_length_sec = window_length_sec

    @property
    def name(self) -> str:
        return "CHROM"

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

        # Chrominance signals:
        # Xs = 3 * Rn - 2 * Gn
        # Ys = 1.5 * Rn + Gn - 1.5 * Bn
        xs = 3.0 * r_n - 2.0 * g_n
        ys = 1.5 * r_n + g_n - 1.5 * b_n

        std_xs = np.std(xs)
        std_ys = np.std(ys)
        alpha = (std_xs / std_ys) if std_ys > 1e-6 else 1.0

        pulse_signal = xs - alpha * ys
        pulse_signal = pulse_signal - np.mean(pulse_signal)

        std_p = np.std(pulse_signal)
        if std_p > 1e-6:
            pulse_signal = pulse_signal / std_p

        return PulseWaveform(
            timestamps=ts.copy(),
            signal=pulse_signal,
            algorithm_name=self.name,
            sampling_rate_hz=fs,
        )
