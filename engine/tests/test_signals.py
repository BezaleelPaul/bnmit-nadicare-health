import pytest
import numpy as np
from somythra.signals.rgb_extraction import RGBSignalAccumulator
from somythra.signals.preprocessing import SignalPreprocessor, PreprocessingConfig
from somythra.signals.filtering import BandpassFilter, FilterConfig
from somythra.models.schemas import FrameRGBMeasurement, TemporalRGBSignal
from somythra.signals.synthetic import generate_synthetic_rgb_signal


def test_rgb_accumulator_buffering():
    accum = RGBSignalAccumulator(expected_fps=30.0)

    for i in range(10):
        t = i * 0.033
        accum.add_frame_rois(
            timestamp=t,
            forehead_rgb=(True, 150.0 + i, 100.0, 80.0, 500),
            left_cheek_rgb=(True, 140.0, 95.0, 75.0, 300),
            right_cheek_rgb=(True, 142.0, 96.0, 76.0, 300),
        )

    forehead = accum.get_signal("forehead")
    assert forehead is not None
    assert forehead.frame_count == 10
    assert forehead.r[0] == 150.0

    combined = accum.get_signal("combined")
    assert combined is not None
    assert combined.frame_count == 10
    # Combined should be weighted average: (150*500 + 140*300 + 142*300) / 1100 = 145.09
    assert pytest.approx(combined.r[0], 0.1) == 145.1

    accum.clear()
    assert accum.get_signal("forehead") is None


def test_preprocessor_resampling_and_detrending():
    sig, _ = generate_synthetic_rgb_signal(
        duration_seconds=5.0,
        sampling_rate_hz=30.0,
        jitter_std_sec=0.005,  # Introduce noticeable jitter
    )

    pre = SignalPreprocessor(
        PreprocessingConfig(
            target_sampling_rate_hz=30.0,
            resample_uniform=True,
            detrend=True,
            normalize_temporal=True,
        )
    )
    processed, jitter_ms, motion = pre.process(sig)

    assert jitter_ms > 0.0
    assert processed.frame_count > 100
    assert pytest.approx(processed.sampling_rate_hz, 0.1) == 30.0
    # Detrending centers signal around 0
    assert abs(np.mean(processed.g)) < 0.05


def test_bandpass_filter_frequency_response():
    filt = BandpassFilter(FilterConfig(lowcut_hz=0.7, highcut_hz=3.5, order=3))
    fs = 30.0
    t = np.arange(0, 8.0, 1.0 / fs)

    # Signal with in-band component (1.2 Hz = 72 BPM) and out-of-band components (0.1 Hz and 8.0 Hz)
    f_inband = 1.2
    f_low = 0.1
    f_high = 8.0

    pure_cardiac = np.sin(2 * np.pi * f_inband * t)
    composite = pure_cardiac + 2.0 * np.sin(2 * np.pi * f_low * t) + 1.5 * np.sin(2 * np.pi * f_high * t)

    filtered = filt.filter_1d(composite, fs)

    # In-band component should be preserved; out-of-band attenuated
    # Correlation between filtered and true in-band cardiac component should be high (> 0.90)
    corr = np.corrcoef(filtered[30:-30], pure_cardiac[30:-30])[0, 1]
    assert corr > 0.88


def test_bandpass_filter_nyquist_guard():
    filt = BandpassFilter(FilterConfig(lowcut_hz=0.7, highcut_hz=3.5))
    # Unviably low sampling rate below 2 * lowcut (e.g. 1.0 Hz)
    with pytest.raises(ValueError, match="below Nyquist minimum"):
        filt.filter_1d(np.ones(20), fs=1.0)
