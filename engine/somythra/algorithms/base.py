"""
SOMYTHRA Algorithms — Base rPPG Algorithm Interface.

Defines the contract for remote photoplethysmography (rPPG) optical pulse extractors.
"""

from abc import ABC, abstractmethod
from ..models.schemas import TemporalRGBSignal, PulseWaveform


class BaseRPPGAlgorithm(ABC):
    """Abstract base class for all rPPG signal extraction methods."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algorithm (e.g. POS, CHROM)."""
        pass

    @abstractmethod
    def extract_pulse(self, rgb_signal: TemporalRGBSignal) -> PulseWaveform:
        """
        Extract 1D blood volume pulse waveform from temporal RGB color time-series.
        """
        pass
