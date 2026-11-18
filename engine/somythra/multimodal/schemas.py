"""
SOMYTHRA Multimodal — Sensor Modality Schemas.

Defines research-grade data structures for:
- Near-Infrared (NIR) 850nm / 940nm optical absorption
- Long-Wave Infrared (LWIR) Thermal Thermography
- Multimodal Fused Physiological Telemetry
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class NIRFrame:
    """Optical Near-Infrared frame (e.g., 850nm or 940nm band)."""
    timestamp: float
    wavelength_nm: int
    data: np.ndarray  # 2D grayscale intensity
    exposure_ms: float = 16.6
    emitter_power_mw: float = 50.0

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape


@dataclass
class ThermalFrame:
    """Long-wave infrared microbolometer thermal frame in degrees Celsius."""
    timestamp: float
    temperature_matrix_c: np.ndarray  # 2D calibrated temperature values
    ambient_temp_c: float = 24.0
    emissivity: float = 0.98  # Human skin emissivity ~ 0.98

    @property
    def min_temp(self) -> float:
        return float(np.min(self.temperature_matrix_c))

    @property
    def max_temp(self) -> float:
        return float(np.max(self.temperature_matrix_c))

    @property
    def mean_temp(self) -> float:
        return float(np.mean(self.temperature_matrix_c))


@dataclass
class MultimodalFusedSample:
    """Synchronized fused multi-sensor telemetry."""
    timestamp: float
    rgb_pulse_sample: float
    nir_pulse_sample: Optional[float]
    nasal_respiration_bpm: Optional[float]
    periorbital_temp_c: Optional[float]
    casi_stress_score: float  # Composite Autonomic Strain Index (0 - 100)
    autonomic_dominance: str  # Sympathetic vs Parasympathetic vs Homeostasis
    modality_confidence: Dict[str, float] = field(default_factory=dict)
