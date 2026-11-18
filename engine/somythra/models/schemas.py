"""
SOMYTHRA R001 — Core Data Models & Signal Schemas.

All physiological measurement representations enforce explicit quality status,
modality metadata, timestamp precision, and non-fabricated outputs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np


class QualityStatus(str, Enum):
    """Quality and reliability classification for camera physiological estimation."""
    OPTIMAL = "OPTIMAL"
    ACCEPTABLE = "ACCEPTABLE"
    POOR_SNR = "POOR_SNR"
    INSUFFICIENT_DURATION = "INSUFFICIENT_DURATION"
    HIGH_MOTION_ARTIFACT = "HIGH_MOTION_ARTIFACT"
    LOW_SKIN_COVERAGE = "LOW_SKIN_COVERAGE"
    IRREGULAR_SAMPLING = "IRREGULAR_SAMPLING"
    OUT_OF_PHYSIOLOGICAL_RANGE = "OUT_OF_PHYSIOLOGICAL_RANGE"
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    UNPROCESSED = "UNPROCESSED"


@dataclass(frozen=True)
class BoundingBox:
    """Rectangular image region in pixel coordinates."""
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def is_valid(self, min_area: int = 100) -> bool:
        return self.w > 0 and self.h > 0 and self.area >= min_area


@dataclass
class FaceROISet:
    """Distinct regions of interest extracted from a detected face."""
    face: BoundingBox
    forehead: Optional[BoundingBox] = None
    left_cheek: Optional[BoundingBox] = None
    right_cheek: Optional[BoundingBox] = None
    confidence: float = 1.0
    detected: bool = True


@dataclass
class FrameRGBMeasurement:
    """Spatial mean RGB values from a single video frame for one region."""
    timestamp: float  # Seconds (relative or absolute)
    r: float
    g: float
    b: float
    roi_name: str = "forehead"
    skin_pixel_count: int = 0
    is_valid: bool = True
    lighting_mean: float = 0.0


@dataclass
class TemporalRGBSignal:
    """
    Multichannel raw or preprocessed optical time-series.
    r, g, b, and timestamps are 1D numpy arrays of identical length.
    """
    timestamps: np.ndarray
    r: np.ndarray
    g: np.ndarray
    b: np.ndarray
    roi_name: str = "combined"
    sampling_rate_hz: float = 30.0

    def __post_init__(self):
        n = len(self.timestamps)
        if len(self.r) != n or len(self.g) != n or len(self.b) != n:
            raise ValueError(
                f"Array length mismatch: timestamps={n}, r={len(self.r)}, g={len(self.g)}, b={len(self.b)}"
            )

    @property
    def duration_seconds(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def frame_count(self) -> int:
        return len(self.timestamps)


@dataclass
class PulseWaveform:
    """1D physiological pulse signal extracted from camera color fluctuations."""
    timestamps: np.ndarray
    signal: np.ndarray
    algorithm_name: str  # e.g., 'POS', 'CHROM'
    sampling_rate_hz: float = 30.0

    def __post_init__(self):
        if len(self.timestamps) != len(self.signal):
            raise ValueError(
                f"Timestamps length ({len(self.timestamps)}) != signal length ({len(self.signal)})"
            )

    @property
    def duration_seconds(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])


@dataclass
class QualityAssessment:
    """Rigorous signal-to-noise ratio and confidence evaluation."""
    snr_db: float
    confidence: float  # Normalized [0.0, 1.0]
    status: QualityStatus
    valid_frame_ratio: float = 1.0
    jitter_std_ms: float = 0.0
    spectral_entropy: float = 0.0
    rejection_reason: Optional[str] = None

    @property
    def is_reliable(self) -> bool:
        return self.status in (QualityStatus.OPTIMAL, QualityStatus.ACCEPTABLE)


@dataclass
class HeartRateEstimate:
    """
    Physiological pulse frequency determination.
    If the signal fails quality verification, bpm is None and rejection_reason explains why.
    """
    bpm: Optional[float]
    confidence: float
    snr_db: float
    is_reliable: bool
    quality_status: QualityStatus
    rejection_reason: Optional[str] = None
    peak_frequency_hz: Optional[float] = None
    window_duration_sec: float = 0.0
    algorithm_name: str = "POS"
    disclaimer: str = (
        "Research-grade camera pulse estimate. "
        "Not calibrated for medical diagnosis or clinical monitoring."
    )


@dataclass
class RPPGPipelineResult:
    """Complete structured output from the SOMYTHRA rPPG processing pipeline."""
    raw_signal: TemporalRGBSignal
    preprocessed_signal: TemporalRGBSignal
    pulse_waveform: PulseWaveform
    heart_rate: HeartRateEstimate
    quality: QualityAssessment
    forehead_signal: Optional[TemporalRGBSignal] = None
    cheeks_signal: Optional[TemporalRGBSignal] = None
    fps_measured: float = 30.0
    dropped_frames: int = 0
    total_frames_processed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveStreamPacket:
    """Real-time telemetry packet emitted per frame by LiveRPPGStreamer."""
    frame_index: int
    timestamp: float
    pulse_sample: float
    bpm: Optional[float]
    confidence: float
    snr_db: float
    quality_status: QualityStatus
    is_cadence_update: bool
    rejection_reason: Optional[str] = None
    optical_features: Optional[Dict[str, float]] = None
    face_detected: bool = True
    face_box: Optional[BoundingBox] = None
