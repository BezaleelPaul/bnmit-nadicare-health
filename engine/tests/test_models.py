import pytest
import numpy as np
from somythra.models.schemas import (
    BoundingBox,
    FaceROISet,
    TemporalRGBSignal,
    PulseWaveform,
    QualityStatus,
    QualityAssessment,
    HeartRateEstimate,
)


def test_bounding_box_geometry():
    box = BoundingBox(x=10, y=20, w=100, h=80, confidence=0.95)
    assert box.x2 == 110
    assert box.y2 == 100
    assert box.area == 8000
    assert box.is_valid(min_area=100) is True

    invalid_box = BoundingBox(x=0, y=0, w=0, h=50)
    assert invalid_box.is_valid() is False


def test_temporal_rgb_signal_validation():
    ts = np.array([0.0, 0.033, 0.066, 0.099])
    r = np.array([120.0, 121.0, 122.0, 123.0])
    g = np.array([90.0, 91.0, 92.0, 93.0])
    b = np.array([80.0, 81.0, 82.0, 83.0])

    sig = TemporalRGBSignal(timestamps=ts, r=r, g=g, b=b, roi_name="forehead", sampling_rate_hz=30.0)
    assert sig.frame_count == 4
    assert pytest.approx(sig.duration_seconds, 0.001) == 0.099

    # Length mismatch must raise ValueError
    with pytest.raises(ValueError, match="Array length mismatch"):
        TemporalRGBSignal(timestamps=ts, r=r[:2], g=g, b=b, roi_name="mismatch")


def test_pulse_waveform_length_mismatch():
    ts = np.array([0.0, 0.1, 0.2])
    data = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Timestamps length"):
        PulseWaveform(timestamps=ts, signal=data, algorithm_name="POS")


def test_heart_rate_estimate_disclaimer():
    est = HeartRateEstimate(
        bpm=72.0,
        confidence=0.88,
        snr_db=6.5,
        is_reliable=True,
        quality_status=QualityStatus.OPTIMAL,
    )
    assert est.is_reliable is True
    assert "Not calibrated for medical diagnosis" in est.disclaimer
