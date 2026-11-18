"""
SOMYTHRA Quality — Signal-to-Noise Ratio (SNR) & Reliability Assessment.

Computes physical frequency-domain SNR, spectral entropy, and multi-factor
quality classification to prevent presentation of fabricated or corrupted readings.
"""

from dataclasses import dataclass
import logging
from typing import Tuple, Optional
import numpy as np
from ..models.schemas import QualityAssessment, QualityStatus

logger = logging.getLogger(__name__)


@dataclass
class QualityConfig:
    min_duration_seconds: float = 4.0
    min_snr_db: float = -3.0
    min_confidence: float = 0.35
    min_valid_frame_ratio: float = 0.60
    max_jitter_ms: float = 45.0
    max_motion_score: float = 0.35
    freq_tolerance_hz: float = 0.15  # +/- 9 BPM around peak and 1st harmonic


class QualityAssessmentEngine:
    """
    Evaluates pulse signal fidelity using the standard de Haan / Wang rPPG SNR formulation:
    Ratio of fundamental and first harmonic pulse energy to in-band residual noise energy.
    """

    def __init__(self, config: QualityConfig = None):
        self.config = config or QualityConfig()

    def assess_signal(
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
        peak_freq: Optional[float],
        duration_seconds: float,
        valid_frame_ratio: float = 1.0,
        jitter_std_ms: float = 0.0,
        motion_score: float = 0.0,
        lowcut_hz: float = 0.7,
        highcut_hz: float = 3.5,
    ) -> QualityAssessment:
        """
        Evaluate full quality metrics for the physiological pulse spectrum.
        """
        # Check duration
        if duration_seconds < self.config.min_duration_seconds:
            return QualityAssessment(
                snr_db=-20.0,
                confidence=0.0,
                status=QualityStatus.INSUFFICIENT_DURATION,
                valid_frame_ratio=valid_frame_ratio,
                jitter_std_ms=jitter_std_ms,
                rejection_reason=(
                    f"Insufficient duration ({duration_seconds:.1f}s < {self.config.min_duration_seconds}s required)"
                ),
            )

        # Check valid frame tracking ratio
        if valid_frame_ratio < self.config.min_valid_frame_ratio:
            return QualityAssessment(
                snr_db=-15.0,
                confidence=0.1,
                status=QualityStatus.LOW_SKIN_COVERAGE,
                valid_frame_ratio=valid_frame_ratio,
                jitter_std_ms=jitter_std_ms,
                rejection_reason=(
                    f"Face/skin lost across {int((1.0 - valid_frame_ratio) * 100)}% of frames."
                ),
            )

        # Check motion artifacts
        if motion_score > self.config.max_motion_score:
            return QualityAssessment(
                snr_db=-10.0,
                confidence=0.15,
                status=QualityStatus.HIGH_MOTION_ARTIFACT,
                valid_frame_ratio=valid_frame_ratio,
                jitter_std_ms=jitter_std_ms,
                rejection_reason="Excessive subject or camera motion detected.",
            )

        # Check timestamp jitter
        if jitter_std_ms > self.config.max_jitter_ms:
            return QualityAssessment(
                snr_db=-8.0,
                confidence=0.2,
                status=QualityStatus.IRREGULAR_SAMPLING,
                valid_frame_ratio=valid_frame_ratio,
                jitter_std_ms=jitter_std_ms,
                rejection_reason=f"Severe camera frame timing jitter ({jitter_std_ms:.1f}ms std dev).",
            )

        if peak_freq is None or len(freqs) < 10 or len(psd) < 10:
            return QualityAssessment(
                snr_db=-20.0,
                confidence=0.0,
                status=QualityStatus.POOR_SNR,
                valid_frame_ratio=valid_frame_ratio,
                jitter_std_ms=jitter_std_ms,
                rejection_reason="No coherent peak found in physiological frequency band.",
            )

        # Calculate SNR and Spectral Entropy
        snr_db = self.compute_snr(freqs, psd, peak_freq, lowcut_hz, highcut_hz)
        entropy = self.compute_spectral_entropy(freqs, psd, lowcut_hz, highcut_hz)

        # Calculate confidence [0.0, 1.0] from SNR
        # Sigmoid scaling around -1.0 dB threshold
        snr_norm = 1.0 / (1.0 + np.exp(-0.40 * (snr_db - (-1.0))))
        confidence = float(np.clip(snr_norm, 0.0, 1.0))

        # Reliability decision
        if snr_db < self.config.min_snr_db or confidence < self.config.min_confidence:
            return QualityAssessment(
                snr_db=round(snr_db, 2),
                confidence=round(confidence, 2),
                status=QualityStatus.POOR_SNR,
                valid_frame_ratio=valid_frame_ratio,
                jitter_std_ms=jitter_std_ms,
                spectral_entropy=round(entropy, 3),
                rejection_reason=(
                    f"Low pulse signal-to-noise ratio ({snr_db:.1f} dB < {self.config.min_snr_db} dB limit)."
                ),
            )

        status = QualityStatus.OPTIMAL if snr_db >= 3.0 else QualityStatus.ACCEPTABLE

        return QualityAssessment(
            snr_db=round(snr_db, 2),
            confidence=round(confidence, 2),
            status=status,
            valid_frame_ratio=valid_frame_ratio,
            jitter_std_ms=jitter_std_ms,
            spectral_entropy=round(entropy, 3),
            rejection_reason=None,
        )

    def compute_snr(
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
        peak_freq: float,
        lowcut_hz: float,
        highcut_hz: float,
    ) -> float:
        """
        Calculates rPPG SNR in dB according to Wang/de Haan:
        Signal Power: Energy in [f0 - tol, f0 + tol] and [2*f0 - tol, 2*f0 + tol].
        Noise Power: Total energy in [lowcut, highcut] minus Signal Power.
        """
        in_band = (freqs >= lowcut_hz) & (freqs <= highcut_hz)
        if not np.any(in_band):
            return -20.0

        band_freqs = freqs[in_band]
        band_psd = psd[in_band]
        total_energy = np.sum(band_psd)

        tol = self.config.freq_tolerance_hz
        fund_mask = (band_freqs >= (peak_freq - tol)) & (band_freqs <= (peak_freq + tol))
        harm_mask = (band_freqs >= (2.0 * peak_freq - tol)) & (band_freqs <= (2.0 * peak_freq + tol))
        signal_mask = fund_mask | harm_mask

        signal_energy = np.sum(band_psd[signal_mask])
        noise_energy = max(1e-12, total_energy - signal_energy)

        if signal_energy <= 1e-12:
            return -20.0

        snr = 10.0 * np.log10(signal_energy / noise_energy)
        return float(np.clip(snr, -30.0, 30.0))

    @staticmethod
    def compute_spectral_entropy(
        freqs: np.ndarray, psd: np.ndarray, lowcut_hz: float, highcut_hz: float
    ) -> float:
        """Shannon entropy of the normalized power distribution in the physiological band."""
        mask = (freqs >= lowcut_hz) & (freqs <= highcut_hz)
        if not np.any(mask):
            return 0.0

        p = psd[mask]
        p_sum = np.sum(p)
        if p_sum <= 0:
            return 0.0

        p_norm = p / p_sum
        p_norm = p_norm[p_norm > 1e-12]
        return float(-np.sum(p_norm * np.log2(p_norm)))
