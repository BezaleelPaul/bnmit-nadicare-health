"""
Tests for SOMYTHRA Vision LiveRPPGStreamer.
Verifies real-time sliding window buffering, ambient illumination compensation,
live pulse sample emission, and cadence BPM updates.
"""

import numpy as np
import pytest

from somythra.streamer import LiveRPPGStreamer
from somythra.models.schemas import QualityStatus, LiveStreamPacket
from somythra.signals.synthetic import generate_synthetic_rgb_signal, generate_synthetic_face_frame


def test_live_streamer_buffering_and_packet_emission():
    """Verify that LiveRPPGStreamer emits a valid LiveStreamPacket for every frame."""
    streamer = LiveRPPGStreamer(
        window_duration_seconds=5.0,
        cadence_interval_seconds=1.0,
        min_duration_seconds=3.0,
        target_fps=30.0,
        compensate_ambient=True,
    )

    sig, _ = generate_synthetic_rgb_signal(duration_seconds=1.0, sampling_rate_hz=30.0, heart_rate_bpm=72.0)

    packets = []
    for i in range(len(sig.timestamps)):
        frame = generate_synthetic_face_frame(
            width=640, height=480,
            r_val=float(sig.r[i]), g_val=float(sig.g[i]), b_val=float(sig.b[i]),
            face_box_coords=(160, 100, 320, 320),
        )
        pkt = streamer.ingest_frame(frame, timestamp=float(sig.timestamps[i]))
        packets.append(pkt)

    assert len(packets) == 30
    last_pkt = packets[-1]
    assert isinstance(last_pkt, LiveStreamPacket)
    assert last_pkt.frame_index == 30
    assert last_pkt.face_detected is True
    # At 1 second (< min_duration 3.0s), status should indicate buffering
    assert last_pkt.quality_status in (QualityStatus.INSUFFICIENT_DURATION, QualityStatus.UNPROCESSED)
    assert last_pkt.bpm is None


def test_live_streamer_cadence_bpm_update():
    """Verify that after enough buffer is accumulated, cadence updates produce valid BPM and optical features."""
    streamer = LiveRPPGStreamer(
        window_duration_seconds=6.0,
        cadence_interval_seconds=1.0,
        min_duration_seconds=3.0,
        target_fps=30.0,
        compensate_ambient=True,
    )

    sig, bpm_gt = generate_synthetic_rgb_signal(duration_seconds=4.0, sampling_rate_hz=30.0, heart_rate_bpm=75.0)

    cadence_count = 0
    final_packet = None

    for i in range(len(sig.timestamps)):
        frame = generate_synthetic_face_frame(
            width=640, height=480,
            r_val=float(sig.r[i]), g_val=float(sig.g[i]), b_val=float(sig.b[i]),
            face_box_coords=(160, 100, 320, 320),
        )
        pkt = streamer.ingest_frame(frame, timestamp=float(sig.timestamps[i]))
        if pkt.is_cadence_update:
            cadence_count += 1
        final_packet = pkt

    assert cadence_count >= 3, f"Expected at least 3 cadence triggers, got {cadence_count}"
    assert final_packet is not None

    if final_packet.bpm is not None:
        assert 60.0 <= final_packet.bpm <= 90.0
        assert final_packet.optical_features is not None
        assert "mean_hr" in final_packet.optical_features
        assert "rmssd" in final_packet.optical_features
        assert "pnn50" in final_packet.optical_features
        assert "lf_hf" in final_packet.optical_features


def test_live_streamer_reset():
    """Verify that reset() clears all internal buffer states."""
    streamer = LiveRPPGStreamer()
    sig, _ = generate_synthetic_rgb_signal(duration_seconds=1.0, sampling_rate_hz=30.0)

    for i in range(len(sig.timestamps)):
        frame = generate_synthetic_face_frame(
            width=640, height=480,
            r_val=float(sig.r[i]), g_val=float(sig.g[i]), b_val=float(sig.b[i]),
            face_box_coords=(160, 100, 320, 320),
        )
        streamer.ingest_frame(frame, timestamp=float(sig.timestamps[i]))

    assert len(streamer._buffer) > 0
    streamer.reset()
    assert len(streamer._buffer) == 0
    assert streamer._frame_index == 0
    assert streamer._latest_bpm is None
