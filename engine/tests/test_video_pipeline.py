import pytest
from pathlib import Path
import cv2
import numpy as np

from somythra.pipeline import RPPGPipeline, PipelineConfig
from somythra.signals.synthetic import generate_synthetic_rgb_signal, generate_synthetic_face_frame


def test_video_pipeline_end_to_end(tmp_path: Path):
    """
    Generate a real MP4 video file containing a synthetic face with animated skin plethysmography.
    Decode using VideoCaptureStream and verify the full rPPG pipeline end-to-end.
    """
    video_path = tmp_path / "synthetic_face_test.mp4"
    fps = 30.0
    duration_sec = 6.0
    width = 320
    height = 240
    n_frames = int(fps * duration_sec)
    target_bpm = 72.0

    sig, _ = generate_synthetic_rgb_signal(
        duration_seconds=duration_sec,
        sampling_rate_hz=fps,
        heart_rate_bpm=target_bpm,
        jitter_std_sec=0.0,
        snr_factor=30.0,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    assert writer.isOpened(), "Could not initialize video writer"

    face_coords = (40, 20, 240, 200)

    for i in range(n_frames):
        r_val = sig.r[i]
        g_val = sig.g[i]
        b_val = sig.b[i]
        frame = generate_synthetic_face_frame(
            width=width,
            height=height,
            r_val=r_val,
            g_val=g_val,
            b_val=b_val,
            face_box_coords=face_coords,
        )
        writer.write(frame)

    writer.release()
    assert video_path.exists()
    assert video_path.stat().st_size > 0

    # Execute pipeline on the video file
    pipeline = RPPGPipeline(PipelineConfig(algorithm="POS", min_duration_seconds=3.5))
    result = pipeline.process_video_file(video_path)

    assert result.total_frames_processed == n_frames
    assert result.fps_measured > 0.0
    assert result.raw_signal.frame_count > 0
    assert result.pulse_waveform is not None
    assert result.heart_rate is not None
    assert result.quality is not None
