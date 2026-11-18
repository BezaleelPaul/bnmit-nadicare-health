"""
SOMYTHRA Backend — Vision Engine Service Bridge.
Integrates engine.somythra computer vision rPPG processing with FastAPI WebSockets and REST.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import cv2

# Ensure engine package is importable
ENGINE_PATH = Path(__file__).resolve().parents[3] / "engine"
if str(ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(ENGINE_PATH))

from somythra.streamer import LiveRPPGStreamer
from somythra.pipeline import RPPGPipeline, PipelineConfig
from somythra.models.schemas import LiveStreamPacket, RPPGPipelineResult
from app.engine.stress import SomythraStressEngine
from app.models.schemas import StressPredictRequest

logger = logging.getLogger(__name__)


class SomythraVisionService:
    """
    Manages live camera frame ingestion, sliding FIFO rPPG processing,
    and automatic coupling with SOMYTHRA Twin autonomic stress classification.
    """

    @staticmethod
    def create_streamer(
        window_duration: float = 8.0,
        cadence_interval: float = 1.0,
        fps: float = 30.0,
        algorithm: str = "pos",
        compensate_ambient: bool = True,
    ) -> LiveRPPGStreamer:
        """Create a stateful live rPPG streamer for a client session."""
        return LiveRPPGStreamer(
            window_duration_seconds=window_duration,
            cadence_interval_seconds=cadence_interval,
            target_fps=fps,
            algorithm_name=algorithm,
            compensate_ambient=compensate_ambient,
        )

    @classmethod
    def process_frame_bytes(
        cls, image_bytes: bytes, streamer: LiveRPPGStreamer
    ) -> Dict[str, Any]:
        """
        Decode raw JPEG/PNG image buffer, ingest into streamer, and attach stress telemetry.
        """
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("Invalid image bytes; could not decode JPEG/PNG frame.")

        pkt: LiveStreamPacket = streamer.ingest_frame(frame_bgr)

        # Autonomic stress prediction when optical features are available
        stress_info: Optional[Dict[str, Any]] = None
        if pkt.bpm is not None and pkt.optical_features is not None:
            opt = pkt.optical_features
            stress_res = SomythraStressEngine.predict_stress(
                StressPredictRequest(
                    hr=opt["mean_hr"],
                    hrv_rmssd=opt["rmssd"],
                    sdnn=opt["rmssd"] * np.sqrt(opt["lf_hf"]),
                )
            )
            stress_info = {
                "stress_level": stress_res.stress_level,
                "confidence": stress_res.confidence,
                "stress_index": stress_res.stress_index,
                "autonomic_state": stress_res.autonomic_state,
            }

        face_box_dict = None
        if pkt.face_box is not None and pkt.face_box.is_valid():
            face_box_dict = {
                "x": pkt.face_box.x,
                "y": pkt.face_box.y,
                "w": pkt.face_box.w,
                "h": pkt.face_box.h,
            }

        return {
            "frame_index": pkt.frame_index,
            "timestamp": round(pkt.timestamp, 4),
            "pulse_sample": round(pkt.pulse_sample, 4),
            "bpm": round(pkt.bpm, 1) if pkt.bpm is not None else None,
            "confidence": round(pkt.confidence, 3),
            "snr_db": round(pkt.snr_db, 2),
            "quality_status": pkt.quality_status.value,
            "is_cadence_update": pkt.is_cadence_update,
            "rejection_reason": pkt.rejection_reason,
            "face_detected": pkt.face_detected,
            "face_box": face_box_dict,
            "optical_features": pkt.optical_features,
            "stress": stress_info,
        }

    @classmethod
    def process_video_file(
        cls, video_path: str, algorithm: str = "pos"
    ) -> Dict[str, Any]:
        """
        Run the complete SOMYTHRA Vision RPPGPipeline on a saved video recording.
        """
        pipeline = RPPGPipeline(config=PipelineConfig(algorithm=algorithm))
        result: RPPGPipelineResult = pipeline.process_video(video_path)

        stress_info = None
        if result.heart_rate.is_reliable and result.heart_rate.bpm is not None:
            hr_val = float(result.heart_rate.bpm)
            stress_res = SomythraStressEngine.predict_stress(
                StressPredictRequest(hr=hr_val, hrv_rmssd=40.0)
            )
            stress_info = {
                "stress_level": stress_res.stress_level,
                "confidence": stress_res.confidence,
                "stress_index": stress_res.stress_index,
                "autonomic_state": stress_res.autonomic_state,
            }

        return {
            "heart_rate_bpm": result.heart_rate.bpm,
            "confidence": result.heart_rate.confidence,
            "snr_db": result.quality.snr_db,
            "quality_status": result.quality.status.value,
            "is_reliable": result.heart_rate.is_reliable,
            "rejection_reason": result.heart_rate.rejection_reason,
            "total_frames": result.total_frames_processed,
            "fps_measured": round(result.fps_measured, 2),
            "duration_seconds": round(result.pulse_waveform.duration_seconds, 2),
            "stress": stress_info,
        }
