import pytest
import numpy as np
from somythra.physiology.heart_rate import HeartRateEstimator, HeartRateConfig
from somythra.quality.assessment import QualityConfig
from somythra.models.schemas import PulseWaveform, QualityStatus
from somythra.signals.synthetic import generate_synthetic_rgb_signal
from somythra.algorithms.pos import POSAlgorithm
from somythra.signals.preprocessing import SignalPreprocessor
from somythra.signals.filtering import BandpassFilter


def test_heart_rate_estimation_accuracy():
    target_bpm = 76.5
    sig, _ = generate_synthetic_rgb_signal(
        duration_seconds=10.0,
        sampling_rate_hz=30.0,
        heart_rate_bpm=target_bpm,
        snr_factor=25.0,
    )
    pre = SignalPreprocessor()
    preprocessed, _, _ = pre.process(sig)
    pos = POSAlgorithm()
    pulse = pos.extract_pulse(preprocessed)
    filt = BandpassFilter()
    filtered_pulse = filt.filter_pulse_waveform(pulse)

    estimator = HeartRateEstimator()
    est, quality, freqs, psd = estimator.estimate_heart_rate(filtered_pulse)

    assert est.is_reliable is True
    assert est.bpm is not None
    assert abs(est.bpm - target_bpm) <= 1.0
    assert quality.status in (QualityStatus.OPTIMAL, QualityStatus.ACCEPTABLE)
    assert quality.snr_db > 2.0


def test_rejection_of_pure_noise():
    # Signal with no coherent cardiac rhythm
    rng = np.random.RandomState(99)
    fs = 30.0
    t = np.arange(0, 10.0, 1.0 / fs)
    noise = rng.normal(0, 1.0, len(t))

    pulse = PulseWaveform(timestamps=t, signal=noise, algorithm_name="POS", sampling_rate_hz=fs)
    estimator = HeartRateEstimator()
    est, quality, _, _ = estimator.estimate_heart_rate(pulse)

    # Must reject without fabricating a reading
    assert est.is_reliable is False
    assert est.bpm is None
    assert quality.status == QualityStatus.POOR_SNR
    assert est.rejection_reason is not None


def test_rejection_of_insufficient_duration():
    # Only 2 seconds of data (< 4.0s required)
    fs = 30.0
    t = np.arange(0, 2.0, 1.0 / fs)
    sig = np.sin(2 * np.pi * 1.2 * t)
    pulse = PulseWaveform(timestamps=t, signal=sig, algorithm_name="POS", sampling_rate_hz=fs)

    estimator = HeartRateEstimator(quality_config=QualityConfig(min_duration_seconds=4.0))
    est, quality, _, _ = estimator.estimate_heart_rate(pulse)

    assert est.is_reliable is False
    assert est.bpm is None
    assert quality.status == QualityStatus.INSUFFICIENT_DURATION
    assert "Insufficient duration" in est.rejection_reason


def test_rejection_of_high_motion():
    fs = 30.0
    t = np.arange(0, 8.0, 1.0 / fs)
    sig = np.sin(2 * np.pi * 1.2 * t)
    pulse = PulseWaveform(timestamps=t, signal=sig, algorithm_name="POS", sampling_rate_hz=fs)

    estimator = HeartRateEstimator()
    est, quality, _, _ = estimator.estimate_heart_rate(pulse, motion_score=0.8)

    assert est.is_reliable is False
    assert est.bpm is None
    assert quality.status == QualityStatus.HIGH_MOTION_ARTIFACT
