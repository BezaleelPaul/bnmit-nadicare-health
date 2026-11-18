import pytest
from pathlib import Path
import cv2
import numpy as np

from somythra.pipeline import RPPGPipeline, PipelineConfig
from somythra.models.schemas import QualityStatus
from somythra.signals.synthetic import generate_synthetic_rgb_signal, generate_synthetic_face_frame
from somythra.acquisition.camera import CameraCapture, CameraConfig


def test_pipeline_synthetic_rgb_signal():
    target_bpm = 70.0
    sig, _ = generate_synthetic_rgb_signal(
        duration_seconds=8.0,
        sampling_rate_hz=30.0,
        heart_rate_bpm=target_bpm,
        snr_factor=25.0,
    )

    pipeline = RPPGPipeline(PipelineConfig(algorithm="POS"))
    result = pipeline.process_rgb_signal(sig)

    assert result.heart_rate.is_reliable is True
    assert result.heart_rate.bpm is not None
    assert abs(result.heart_rate.bpm - target_bpm) <= 1.5
    assert result.quality.snr_db > 0.0
    assert result.metadata["algorithm"] == "POS"


def test_pipeline_chrom_algorithm():
    target_bpm = 80.0
    sig, _ = generate_synthetic_rgb_signal(
        duration_seconds=8.0,
        sampling_rate_hz=30.0,
        heart_rate_bpm=target_bpm,
        snr_factor=25.0,
    )

    pipeline = RPPGPipeline(PipelineConfig(algorithm="CHROM"))
    result = pipeline.process_rgb_signal(sig)

    assert result.heart_rate.bpm is not None
    assert abs(result.heart_rate.bpm - target_bpm) <= 1.5
    assert result.metadata["algorithm"] == "CHROM"


def test_pipeline_blank_frames_graceful_handling():
    # Stream of blank frames (no face detected)
    blank_frames = [(np.zeros((240, 320, 3), dtype=np.uint8), i * 0.033) for i in range(30)]

    pipeline = RPPGPipeline()
    result = pipeline.process_frame_stream(iter(blank_frames), measured_fps=30.0)

    # Must complete cleanly without crash, reporting no face detected
    assert result.heart_rate.bpm is None
    assert result.heart_rate.is_reliable is False
    assert result.total_frames_processed == 30


def test_camera_graceful_failure():
    # Attempting to open an impossible hardware index (e.g. 999) must return False cleanly
    cam = CameraCapture(CameraConfig(device_index=999))
    success = cam.open()
    assert success is False
    assert cam.is_open is False
    # Safe release must not crash
    cam.release()
