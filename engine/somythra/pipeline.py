"""
SOMYTHRA Pipeline — End-to-End Camera-Based Physiological Signal Processor.

Orchestrates:
Camera / Video Input
  → Face Detection & Tracking
  → Skin ROI Extraction & Segmentation
  → Multichannel RGB Time-Series Accumulation
  → Signal Preprocessing (Resampling & Detrending)
  → Butterworth Bandpass Filtering
  → rPPG Algorithm (POS / CHROM)
  → Sub-Bin Heart Rate Estimation
  → SNR & Quality Gatekeeper Decision
  → Structured RPPGPipelineResult
"""

from dataclasses import dataclass, field
import logging
import time
from typing import Optional, Union, List, Iterator, Tuple
from pathlib import Path
import numpy as np

from .models.schemas import (
    BoundingBox,
    FaceROISet,
    TemporalRGBSignal,
    PulseWaveform,
    HeartRateEstimate,
    QualityAssessment,
    RPPGPipelineResult,
)
from .acquisition.video import VideoCaptureStream, VideoStreamConfig
from .acquisition.camera import CameraCapture, CameraConfig
from .vision.face_detection import BaseFaceDetector, get_default_face_detector
from .vision.skin_roi import SkinROIExtractor
from .signals.rgb_extraction import RGBSignalAccumulator
from .signals.preprocessing import SignalPreprocessor, PreprocessingConfig
from .signals.filtering import BandpassFilter, FilterConfig
from .algorithms.base import BaseRPPGAlgorithm
from .algorithms.pos import POSAlgorithm
from .algorithms.chrom import CHROMAlgorithm
from .physiology.heart_rate import HeartRateEstimator, HeartRateConfig
from .quality.assessment import QualityConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    algorithm: str = "POS"  # "POS" or "CHROM"
    target_sampling_rate_hz: float = 30.0
    min_duration_seconds: float = 4.0
    lowcut_hz: float = 0.7
    highcut_hz: float = 3.5
    resample_uniform: bool = True
    detrend: bool = True
    save_raw_video: bool = False  # Privacy default: do not store raw face frames


class RPPGPipeline:
    """
    Unified modular rPPG processing pipeline.
    Processes video files, live cameras, or synthetic signals deterministically.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        face_detector: Optional[BaseFaceDetector] = None,
    ):
        self.config = config or PipelineConfig()
        self.face_detector = face_detector or get_default_face_detector()
        self.roi_extractor = SkinROIExtractor()
        self.accumulator = RGBSignalAccumulator(expected_fps=self.config.target_sampling_rate_hz)

        self.preprocessor = SignalPreprocessor(
            PreprocessingConfig(
                target_sampling_rate_hz=self.config.target_sampling_rate_hz,
                resample_uniform=self.config.resample_uniform,
                detrend=self.config.detrend,
            )
        )

        self.bandpass_filter = BandpassFilter(
            FilterConfig(
                lowcut_hz=self.config.lowcut_hz,
                highcut_hz=self.config.highcut_hz,
            )
        )

        if self.config.algorithm.upper() == "CHROM":
            self.algorithm: BaseRPPGAlgorithm = CHROMAlgorithm()
        else:
            self.algorithm = POSAlgorithm()

        self.hr_estimator = HeartRateEstimator(
            config=HeartRateConfig(
                min_bpm=self.config.lowcut_hz * 60.0,
                max_bpm=self.config.highcut_hz * 60.0,
            ),
            quality_config=QualityConfig(
                min_duration_seconds=self.config.min_duration_seconds,
            ),
        )

    def process_video_file(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None,
    ) -> RPPGPipelineResult:
        """Process an entire video file from start to finish."""
        stream_cfg = VideoStreamConfig(file_path=video_path, max_frames=max_frames)
        with VideoCaptureStream(stream_cfg) as stream:
            if not stream.is_open:
                raise RuntimeError(f"Could not open video file: {video_path}")
            return self.process_frame_stream(stream, measured_fps=stream.fps)

    # Alias for convenience
    process_video = process_video_file

    def process_frame_stream(
        self,
        frame_iterator: Iterator[Tuple[np.ndarray, float]],
        measured_fps: float = 30.0,
    ) -> RPPGPipelineResult:
        """
        Process a stream of (frame_bgr, timestamp_seconds).
        Extracts face, accumulates skin RGB, and runs pulse estimation.
        """
        self.accumulator.clear()
        self.face_detector.reset()

        total_frames = 0
        detected_frames = 0

        for frame_bgr, ts in frame_iterator:
            total_frames += 1
            face_box = self.face_detector.detect_face(frame_bgr)

            if face_box is not None and face_box.is_valid():
                detected_frames += 1
                rois = self.roi_extractor.extract_rois(frame_bgr, face_box)

                if rois is not None:
                    fh_rgb = self.roi_extractor.extract_skin_rgb(frame_bgr, rois.forehead)
                    lc_rgb = self.roi_extractor.extract_skin_rgb(frame_bgr, rois.left_cheek)
                    rc_rgb = self.roi_extractor.extract_skin_rgb(frame_bgr, rois.right_cheek)

                    self.accumulator.add_frame_rois(
                        timestamp=ts,
                        forehead_rgb=fh_rgb if fh_rgb[0] else None,
                        left_cheek_rgb=lc_rgb if lc_rgb[0] else None,
                        right_cheek_rgb=rc_rgb if rc_rgb[0] else None,
                    )

        valid_ratio = (detected_frames / max(1, total_frames))
        return self._finish_processing(
            valid_ratio=valid_ratio,
            total_frames=total_frames,
            dropped_frames=total_frames - detected_frames,
            fps_measured=measured_fps,
        )

    def process_rgb_signal(
        self,
        raw_signal: TemporalRGBSignal,
        valid_ratio: float = 1.0,
    ) -> RPPGPipelineResult:
        """
        Process a pre-extracted or synthetic TemporalRGBSignal directly.
        Bypasses vision front-end for unit testing and offline algorithmic research.
        """
        return self._run_signal_pipeline(
            raw_signal=raw_signal,
            valid_ratio=valid_ratio,
            total_frames=raw_signal.frame_count,
            dropped_frames=0,
            fps_measured=raw_signal.sampling_rate_hz,
        )

    def _finish_processing(
        self,
        valid_ratio: float,
        total_frames: int,
        dropped_frames: int,
        fps_measured: float,
    ) -> RPPGPipelineResult:
        raw_combined = self.accumulator.get_signal("combined")
        if raw_combined is None or raw_combined.frame_count < 4:
            # Handle case where no face or skin was detected
            empty_ts = np.array([0.0, 0.1, 0.2])
            empty_zeros = np.zeros(3)
            dummy_raw = TemporalRGBSignal(empty_ts, empty_zeros, empty_zeros, empty_zeros, "empty", fps_measured)
            dummy_pulse = PulseWaveform(empty_ts, empty_zeros, self.algorithm.name, fps_measured)
            hr, quality, _, _ = self.hr_estimator.estimate_heart_rate(
                dummy_pulse, valid_frame_ratio=0.0
            )
            return RPPGPipelineResult(
                raw_signal=dummy_raw,
                preprocessed_signal=dummy_raw,
                pulse_waveform=dummy_pulse,
                heart_rate=hr,
                quality=quality,
                fps_measured=fps_measured,
                dropped_frames=dropped_frames,
                total_frames_processed=total_frames,
            )

        forehead_signal = self.accumulator.get_signal("forehead")
        cheeks_signal = self.accumulator.get_signal("left_cheek")

        return self._run_signal_pipeline(
            raw_signal=raw_combined,
            valid_ratio=valid_ratio,
            total_frames=total_frames,
            dropped_frames=dropped_frames,
            fps_measured=fps_measured,
            forehead_signal=forehead_signal,
            cheeks_signal=cheeks_signal,
        )

    def _run_signal_pipeline(
        self,
        raw_signal: TemporalRGBSignal,
        valid_ratio: float,
        total_frames: int,
        dropped_frames: int,
        fps_measured: float,
        forehead_signal: Optional[TemporalRGBSignal] = None,
        cheeks_signal: Optional[TemporalRGBSignal] = None,
    ) -> RPPGPipelineResult:
        # 1. Preprocessing: jitter analysis, uniform resampling, detrending
        preprocessed, jitter_std_ms, motion_score = self.preprocessor.process(raw_signal)

        # 2. rPPG Pulse Extraction: POS or CHROM
        raw_pulse = self.algorithm.extract_pulse(preprocessed)

        # 3. Butterworth Zero-Phase Bandpass Filter
        filtered_pulse = self.bandpass_filter.filter_pulse_waveform(raw_pulse)

        # 4. Heart Rate Estimation & Quality Assessment
        hr_estimate, quality, freqs, psd = self.hr_estimator.estimate_heart_rate(
            pulse=filtered_pulse,
            valid_frame_ratio=valid_ratio,
            jitter_std_ms=jitter_std_ms,
            motion_score=motion_score,
        )

        return RPPGPipelineResult(
            raw_signal=raw_signal,
            preprocessed_signal=preprocessed,
            pulse_waveform=filtered_pulse,
            heart_rate=hr_estimate,
            quality=quality,
            forehead_signal=forehead_signal,
            cheeks_signal=cheeks_signal,
            fps_measured=fps_measured,
            dropped_frames=dropped_frames,
            total_frames_processed=total_frames,
            metadata={
                "algorithm": self.algorithm.name,
                "jitter_std_ms": jitter_std_ms,
                "motion_score": motion_score,
                "duration_seconds": raw_signal.duration_seconds,
                "psd_peak_freq_hz": hr_estimate.peak_frequency_hz,
            },
        )
