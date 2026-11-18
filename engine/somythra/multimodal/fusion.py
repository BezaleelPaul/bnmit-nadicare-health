"""
SOMYTHRA Multimodal — Sensor Fusion & Autonomic Strain Integration.

Fuses:
1. RGB camera rPPG pulsatile signals (superficial microvasculature)
2. NIR 850nm / 940nm absorption (deeper vascular bed, melanin-invariant)
3. Thermal nasal airflow dynamics (respiratory rate from inhalation cooling)
4. Thermal supraorbital temperature shifts (sympathetic microvascular vasoconstriction)

Outputs:
Composite Autonomic Strain Index (CASI: 0 - 100) and autonomic balance state.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np

from .schemas import NIRFrame, ThermalFrame, MultimodalFusedSample


class MultimodalFusionEngine:
    """
    Multimodal physiological sensor fusion engine.
    Integrates optical (RGB + NIR) and thermal thermographic cues.
    """

    def __init__(self, baseline_temp_c: float = 34.5, baseline_hr_bpm: float = 70.0):
        self.baseline_temp_c = baseline_temp_c
        self.baseline_hr_bpm = baseline_hr_bpm

        # Historical buffers for respiration and vasoconstriction
        self._thermal_nasal_buffer = []
        self._last_fused_sample: Optional[MultimodalFusedSample] = None

    def extract_nasal_respiration(
        self, thermal_matrix: np.ndarray, nasal_roi: Optional[Tuple[int, int, int, int]] = None
    ) -> float:
        """
        Extract nasal temperature. Cyclical temperature drops indicate inhalation of cooler ambient air;
        temperature increases indicate exhalation of warmed core air.
        """
        if thermal_matrix is None or thermal_matrix.size == 0:
            return 32.0

        if nasal_roi is not None:
            x, y, w, h = nasal_roi
            patch = thermal_matrix[y : y + h, x : x + w]
            if patch.size > 0:
                return float(np.mean(patch))

        # Default center region approximation
        h, w = thermal_matrix.shape[:2]
        center_patch = thermal_matrix[int(h * 0.45) : int(h * 0.65), int(w * 0.4) : int(w * 0.6)]
        return float(np.mean(center_patch)) if center_patch.size > 0 else 32.0

    def extract_periorbital_temp(
        self, thermal_matrix: np.ndarray, forehead_roi: Optional[Tuple[int, int, int, int]] = None
    ) -> float:
        """
        Extract supraorbital / forehead thermal value.
        Sympathetic fight-or-flight triggers microvascular peripheral vasoconstriction,
        causing a measurable drop in facial skin temperature.
        """
        if thermal_matrix is None or thermal_matrix.size == 0:
            return self.baseline_temp_c

        if forehead_roi is not None:
            x, y, w, h = forehead_roi
            patch = thermal_matrix[y : y + h, x : x + w]
            if patch.size > 0:
                return float(np.mean(patch))

        h, w = thermal_matrix.shape[:2]
        top_patch = thermal_matrix[int(h * 0.15) : int(h * 0.35), int(w * 0.3) : int(w * 0.7)]
        return float(np.mean(top_patch)) if top_patch.size > 0 else self.baseline_temp_c

    def fuse_step(
        self,
        timestamp: float,
        rgb_pulse: float,
        rgb_hr: Optional[float] = None,
        rgb_confidence: float = 1.0,
        nir_frame: Optional[NIRFrame] = None,
        thermal_frame: Optional[ThermalFrame] = None,
    ) -> MultimodalFusedSample:
        """
        Fuse a single synchronized time instant.
        """
        nir_pulse_val: Optional[float] = None
        nir_conf = 0.0
        if nir_frame is not None and nir_frame.data.size > 0:
            # NIR mean absorption amplitude
            nir_pulse_val = float(np.mean(nir_frame.data))
            nir_conf = 0.9

        nasal_temp: Optional[float] = None
        periorbital_temp: Optional[float] = None
        thermal_conf = 0.0
        respiration_rate_bpm = 16.0  # Nomative resting baseline

        if thermal_frame is not None and thermal_frame.temperature_matrix_c.size > 0:
            mat = thermal_frame.temperature_matrix_c
            nasal_temp = self.extract_nasal_respiration(mat)
            periorbital_temp = self.extract_periorbital_temp(mat)
            thermal_conf = 0.85

            self._thermal_nasal_buffer.append((timestamp, nasal_temp))
            if len(self._thermal_nasal_buffer) > 300:
                self._thermal_nasal_buffer.pop(0)

            # Estimate respiration from thermal oscillations if sufficient buffer
            if len(self._thermal_nasal_buffer) >= 60:
                temps = [t[1] for t in self._thermal_nasal_buffer]
                # Zero-crossing rate
                detrended = temps - np.mean(temps)
                crossings = np.where(np.diff(np.sign(detrended)))[0]
                dt = self._thermal_nasal_buffer[-1][0] - self._thermal_nasal_buffer[0][0]
                if dt > 2.0 and len(crossings) > 1:
                    cycles = len(crossings) / 2.0
                    respiration_rate_bpm = float(np.clip((cycles / dt) * 60.0, 8.0, 35.0))

        # Calculate Composite Autonomic Strain Index (CASI: 0 - 100)
        # Factor 1: Heart Rate elevation relative to baseline
        hr_val = rgb_hr if rgb_hr is not None else self.baseline_hr_bpm
        hr_strain = float(np.clip(((hr_val - self.baseline_hr_bpm) / 40.0) * 50.0, 0.0, 60.0))

        # Factor 2: Thermal Vasoconstriction (drop in periorbital temp indicates sympathetic tone)
        temp_strain = 0.0
        if periorbital_temp is not None:
            temp_delta = self.baseline_temp_c - periorbital_temp
            temp_strain = float(np.clip(temp_delta * 15.0, 0.0, 30.0))

        # Factor 3: Elevated Tachypnea (> 20 breaths/min indicates strain)
        resp_strain = float(np.clip((respiration_rate_bpm - 16.0) * 1.5, 0.0, 20.0))

        casi = float(np.clip(20.0 + hr_strain + temp_strain + resp_strain, 5.0, 95.0))

        if casi > 65.0:
            dominance = "Sympathetic Hyperarousal"
        elif casi > 45.0:
            dominance = "Moderate Sympathetic Arousal"
        elif casi < 25.0:
            dominance = "Parasympathetic Recovery"
        else:
            dominance = "Autonomic Homeostasis"

        sample = MultimodalFusedSample(
            timestamp=timestamp,
            rgb_pulse_sample=rgb_pulse,
            nir_pulse_sample=nir_pulse_val,
            nasal_respiration_bpm=round(respiration_rate_bpm, 1) if thermal_frame else None,
            periorbital_temp_c=round(periorbital_temp, 2) if periorbital_temp else None,
            casi_stress_score=round(casi, 1),
            autonomic_dominance=dominance,
            modality_confidence={
                "rgb_camera": rgb_confidence,
                "nir_optical": nir_conf,
                "thermal_lwir": thermal_conf,
            },
        )
        self._last_fused_sample = sample
        return sample
