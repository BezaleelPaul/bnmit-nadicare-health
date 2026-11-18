"""
SOMYTHRA — R001: Camera-Based Physiological Signal Processing Foundation.

The science of understanding the human body as an interconnected living system.
Research-grade camera pulse extraction and digital-twin signal foundation.
"""

__version__ = "SOMYTHRA-R001-0.1.0"
__identity__ = "SOMYTHRA"
__milestone__ = "R001"

from .pipeline import RPPGPipeline
from .streamer import LiveRPPGStreamer
from .models.schemas import (
    LiveStreamPacket,
    RPPGPipelineResult,
    HeartRateEstimate,
    QualityStatus,
    PulseWaveform,
    TemporalRGBSignal,
)

__all__ = [
    "RPPGPipeline",
    "LiveRPPGStreamer",
    "LiveStreamPacket",
    "RPPGPipelineResult",
    "HeartRateEstimate",
    "QualityStatus",
    "PulseWaveform",
    "TemporalRGBSignal",
]
