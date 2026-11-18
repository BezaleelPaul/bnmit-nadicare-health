"""
SOMYTHRA Physiology — Heart Rate Estimation from Pulse Waveforms.

Extracts cardiac pulse frequency using high-resolution spectral analysis,
parabolic peak refinement, and strict quality verification.
Rejects corrupted, noisy, or brief recordings without fabricating values.
"""

from dataclasses import dataclass
import logging
from typing import Optional, Tuple
import numpy as np
from scipy import signal as sp_signal
from ..models.schemas import PulseWaveform, HeartRateEstimate, QualityAssessment, QualityStatus
from ..quality.assessment import QualityAssessmentEngine, QualityConfig

logger = logging.getLogger(__name__)


@dataclass
class HeartRateConfig:
    min_bpm: float = 42.0   # 0.7 Hz
    max_bpm: float = 210.0  # 3.5 Hz
    fft_zero_pad_factor: int = 4
    welch_nperseg_sec: float = 4.0


class HeartRateEstimator:
    """
    Physiological pulse frequency determination.
    Combines Welch periodogram estimation with parabolic interpolation
    and comprehensive SNR / quality gatekeeper checks.
    """

    def __init__(
        self,
        config: HeartRateConfig = None,
        quality_config: QualityConfig = None,
    ):
        self.config = config or HeartRateConfig()
        self.quality_engine = QualityAssessmentEngine(quality_config)

    @property
    def min_freq_hz(self) -> float:
        return self.config.min_bpm / 60.0

    @property
    def max_freq_hz(self) -> float:
        return self.config.max_bpm / 60.0

    def estimate_heart_rate(
        self,
        pulse: PulseWaveform,
        valid_frame_ratio: float = 1.0,
        jitter_std_ms: float = 0.0,
        motion_score: float = 0.0,
    ) -> Tuple[HeartRateEstimate, QualityAssessment, np.ndarray, np.ndarray]:
        """
        Estimate Heart Rate in BPM from a PulseWaveform.
        Returns:
            (HeartRateEstimate, QualityAssessment, freqs, psd)
        """
        signal_data = pulse.signal
        fs = pulse.sampling_rate_hz
        duration = pulse.duration_seconds
        n = len(signal_data)

        # 1. Compute PSD using zero-padded FFT / Welch
        freqs, psd = self.compute_psd(signal_data, fs)

        # 2. Restrict to physiological band [min_freq_hz, max_freq_hz]
        phys_mask = (freqs >= self.min_freq_hz) & (freqs <= self.max_freq_hz)
        if not np.any(phys_mask) or len(psd) == 0:
            quality = self.quality_engine.assess_signal(
                freqs=freqs,
                psd=psd,
                peak_freq=None,
                duration_seconds=duration,
                valid_frame_ratio=valid_frame_ratio,
                jitter_std_ms=jitter_std_ms,
                motion_score=motion_score,
            )
            return (
                HeartRateEstimate(
                    bpm=None,
                    confidence=0.0,
                    snr_db=-20.0,
                    is_reliable=False,
                    quality_status=quality.status,
                    rejection_reason=quality.rejection_reason,
                    window_duration_sec=duration,
                    algorithm_name=pulse.algorithm_name,
                ),
                quality,
                freqs,
                psd,
            )

        phys_freqs = freqs[phys_mask]
        phys_psd = psd[phys_mask]

        # Find dominant peak index in physiological band
        peak_idx_in_band = int(np.argmax(phys_psd))
        raw_peak_freq = phys_freqs[peak_idx_in_band]

        # Map back to full spectrum index for sub-bin parabolic interpolation
        global_peak_idx = np.where(phys_mask)[0][peak_idx_in_band]
        refined_peak_freq = self._parabolic_peak_interpolation(freqs, psd, global_peak_idx)

        # 3. Assess signal quality & SNR
        quality = self.quality_engine.assess_signal(
            freqs=freqs,
            psd=psd,
            peak_freq=refined_peak_freq,
            duration_seconds=duration,
            valid_frame_ratio=valid_frame_ratio,
            jitter_std_ms=jitter_std_ms,
            motion_score=motion_score,
            lowcut_hz=self.min_freq_hz,
            highcut_hz=self.max_freq_hz,
        )

        # 4. Final reliability decision
        if quality.is_reliable:
            bpm_val = round(refined_peak_freq * 60.0, 1)
            # Extra safety check against unphysiological outputs
            if not (self.config.min_bpm <= bpm_val <= self.config.max_bpm):
                return (
                    HeartRateEstimate(
                        bpm=None,
                        confidence=quality.confidence,
                        snr_db=quality.snr_db,
                        is_reliable=False,
                        quality_status=QualityStatus.OUT_OF_PHYSIOLOGICAL_RANGE,
                        rejection_reason=f"Estimated frequency {bpm_val} BPM exceeds human range.",
                        peak_frequency_hz=refined_peak_freq,
                        window_duration_sec=duration,
                        algorithm_name=pulse.algorithm_name,
                    ),
                    quality,
                    freqs,
                    psd,
                )

            return (
                HeartRateEstimate(
                    bpm=bpm_val,
                    confidence=quality.confidence,
                    snr_db=quality.snr_db,
                    is_reliable=True,
                    quality_status=quality.status,
                    rejection_reason=None,
                    peak_frequency_hz=round(refined_peak_freq, 3),
                    window_duration_sec=duration,
                    algorithm_name=pulse.algorithm_name,
                ),
                quality,
                freqs,
                psd,
            )
        else:
            # Signal rejected: do not output fabricated reading
            return (
                HeartRateEstimate(
                    bpm=None,
                    confidence=quality.confidence,
                    snr_db=quality.snr_db,
                    is_reliable=False,
                    quality_status=quality.status,
                    rejection_reason=quality.rejection_reason,
                    peak_frequency_hz=round(refined_peak_freq, 3),
                    window_duration_sec=duration,
                    algorithm_name=pulse.algorithm_name,
                ),
                quality,
                freqs,
                psd,
            )

    def compute_psd(self, signal_data: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute zero-padded power spectrum using Hann window."""
        n = len(signal_data)
        if n < 4:
            return np.array([]), np.array([])

        # Detrend and window
        detrended = signal_data - np.mean(signal_data)
        window = np.hanning(n)
        windowed = detrended * window

        # Zero-pad to next power of 2 for smooth sub-bin interpolation
        n_fft = max(2048, int(2 ** np.ceil(np.log2(n * self.config.fft_zero_pad_factor))))

        fft_vals = np.fft.rfft(windowed, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        psd = (np.abs(fft_vals) ** 2) / (np.sum(window**2) or 1.0)

        return freqs, psd

    @staticmethod
    def _parabolic_peak_interpolation(freqs: np.ndarray, psd: np.ndarray, idx: int) -> float:
        """Parabolic interpolation of the true peak frequency around maximum FFT bin."""
        if idx <= 0 or idx >= len(psd) - 1:
            return float(freqs[idx])

        alpha = psd[idx - 1]
        beta = psd[idx]
        gamma = psd[idx + 1]

        denom = 2.0 * (alpha - 2.0 * beta + gamma)
        if abs(denom) < 1e-12:
            return float(freqs[idx])

        delta = (alpha - gamma) / denom
        df = freqs[1] - freqs[0]
        return float(freqs[idx] + delta * df)
