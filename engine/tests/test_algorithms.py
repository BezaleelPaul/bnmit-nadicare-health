import pytest
import numpy as np
from somythra.algorithms.pos import POSAlgorithm
from somythra.algorithms.chrom import CHROMAlgorithm
from somythra.signals.synthetic import generate_synthetic_rgb_signal
from somythra.signals.preprocessing import SignalPreprocessor


@pytest.mark.parametrize("target_bpm", [66.0, 75.0, 90.0])
def test_pos_algorithm_frequency_fidelity(target_bpm):
    sig, _ = generate_synthetic_rgb_signal(
        duration_seconds=10.0,
        sampling_rate_hz=30.0,
        heart_rate_bpm=target_bpm,
        snr_factor=30.0,
    )
    pre = SignalPreprocessor()
    preprocessed, _, _ = pre.process(sig)

    pos = POSAlgorithm()
    pulse = pos.extract_pulse(preprocessed)

    assert pulse.algorithm_name == "POS"
    assert len(pulse.signal) == len(preprocessed.timestamps)
    assert not np.any(np.isnan(pulse.signal))

    # Zero-padded spectral analysis for high sub-bin resolution
    fft_vals = np.abs(np.fft.rfft(pulse.signal, n=4096))
    freqs = np.fft.rfftfreq(4096, d=1.0 / pulse.sampling_rate_hz)

    mask = (freqs >= 0.7) & (freqs <= 3.5)
    peak_freq = freqs[mask][np.argmax(fft_vals[mask])]
    est_bpm = peak_freq * 60.0

    # Assert within 1.0 BPM of target
    assert abs(est_bpm - target_bpm) <= 1.0


@pytest.mark.parametrize("target_bpm", [66.0, 75.0, 90.0])
def test_chrom_algorithm_frequency_fidelity(target_bpm):
    sig, _ = generate_synthetic_rgb_signal(
        duration_seconds=10.0,
        sampling_rate_hz=30.0,
        heart_rate_bpm=target_bpm,
        snr_factor=30.0,
    )
    pre = SignalPreprocessor()
    preprocessed, _, _ = pre.process(sig)

    chrom = CHROMAlgorithm()
    pulse = chrom.extract_pulse(preprocessed)

    assert pulse.algorithm_name == "CHROM"
    assert len(pulse.signal) == len(preprocessed.timestamps)
    assert not np.any(np.isnan(pulse.signal))

    fft_vals = np.abs(np.fft.rfft(pulse.signal, n=4096))
    freqs = np.fft.rfftfreq(4096, d=1.0 / pulse.sampling_rate_hz)

    mask = (freqs >= 0.7) & (freqs <= 3.5)
    peak_freq = freqs[mask][np.argmax(fft_vals[mask])]
    est_bpm = peak_freq * 60.0

    assert abs(est_bpm - target_bpm) <= 1.0


def test_algorithm_short_signal_handling():
    sig, _ = generate_synthetic_rgb_signal(duration_seconds=0.1, sampling_rate_hz=30.0)
    pos = POSAlgorithm()
    pulse = pos.extract_pulse(sig)
    assert len(pulse.signal) == sig.frame_count
