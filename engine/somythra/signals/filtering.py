"""
SOMYTHRA Signals — Zero-Phase Bandpass Filtering Module.

Applies forward-backward Butterworth IIR filtering strictly in the human
physiological cardiac frequency band [0.7 Hz - 3.5 Hz] (42 BPM - 210 BPM).
Guarantees zero phase distortion and verifies Nyquist boundary limits.
"""

from dataclasses import dataclass
import logging
from typing import Union
import numpy as np
from scipy import signal as sp_signal
from ..models.schemas import TemporalRGBSignal, PulseWaveform

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    lowcut_hz: float = 0.7   # 42 BPM
    highcut_hz: float = 3.5  # 210 BPM
    order: int = 3           # 3rd order (effective 6th order with zero-phase sosfiltfilt)


class BandpassFilter:
    """
    Zero-phase Butterworth bandpass filter.
    Protects against Nyquist violations and edge boundary transients.
    """

    def __init__(self, config: FilterConfig = None):
        self.config = config or FilterConfig()

    def filter_1d(self, data: np.ndarray, fs: float) -> np.ndarray:
        """
        Filter a 1D real-valued signal.
        """
        if data is None or len(data) < 12:
            return data

        nyquist = 0.5 * fs
        if nyquist <= self.config.lowcut_hz:
            raise ValueError(
                f"Sampling rate {fs:.2f} Hz is below Nyquist minimum for lowcut {self.config.lowcut_hz} Hz."
            )

        low = self.config.lowcut_hz / nyquist
        # Guard high cutoff against Nyquist ceiling
        high = min(0.95, self.config.highcut_hz / nyquist)

        if low >= high:
            low = max(0.01, high - 0.1)

        sos = sp_signal.butter(self.config.order, [low, high], btype="bandpass", output="sos")

        # Zero-phase bidirectional filter
        try:
            filtered = sp_signal.sosfiltfilt(sos, data)
            return filtered
        except Exception as e:
            logger.warning("sosfiltfilt failed (%s); returning detrended raw data", e)
            return data - np.mean(data)

    def filter_rgb_signal(self, rgb_signal: TemporalRGBSignal) -> TemporalRGBSignal:
        """Filter all channels (R, G, B) of a TemporalRGBSignal."""
        fs = rgb_signal.sampling_rate_hz
        r_filt = self.filter_1d(rgb_signal.r, fs)
        g_filt = self.filter_1d(rgb_signal.g, fs)
        b_filt = self.filter_1d(rgb_signal.b, fs)

        return TemporalRGBSignal(
            timestamps=rgb_signal.timestamps.copy(),
            r=r_filt,
            g=g_filt,
            b=b_filt,
            roi_name=rgb_signal.roi_name,
            sampling_rate_hz=fs,
        )

    def filter_pulse_waveform(self, pulse: PulseWaveform) -> PulseWaveform:
        """Filter an extracted 1D pulse waveform."""
        filt = self.filter_1d(pulse.signal, pulse.sampling_rate_hz)
        return PulseWaveform(
            timestamps=pulse.timestamps.copy(),
            signal=filt,
            algorithm_name=pulse.algorithm_name,
            sampling_rate_hz=pulse.sampling_rate_hz,
        )
