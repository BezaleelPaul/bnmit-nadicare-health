"""
Tests for SOMYTHRA Multimodal Fusion Module (RGB + NIR + Thermal).
"""

import pytest
import numpy as np
from somythra.multimodal.schemas import NIRFrame, ThermalFrame, MultimodalFusedSample
from somythra.multimodal.fusion import MultimodalFusionEngine


def test_nir_frame_properties():
    data = np.full((120, 160), 180, dtype=np.uint8)
    frame = NIRFrame(timestamp=1.0, wavelength_nm=850, data=data)

    assert frame.wavelength_nm == 850
    assert frame.shape == (120, 160)
    assert np.mean(frame.data) == 180.0


def test_thermal_frame_properties():
    mat = np.full((60, 80), 34.2, dtype=np.float32)
    mat[30, 40] = 36.5  # Warm region
    mat[10, 10] = 31.0  # Cool region

    frame = ThermalFrame(timestamp=1.0, temperature_matrix_c=mat)
    assert frame.max_temp == pytest.approx(36.5, abs=0.01)
    assert frame.min_temp == pytest.approx(31.0, abs=0.01)
    assert frame.emissivity == 0.98


def test_multimodal_fusion_engine_step():
    engine = MultimodalFusionEngine(baseline_temp_c=34.5, baseline_hr_bpm=70.0)

    # Synthetic frames
    nir_data = np.full((100, 100), 150, dtype=np.uint8)
    nir_f = NIRFrame(timestamp=1.0, wavelength_nm=940, data=nir_data)

    thermal_mat = np.full((80, 80), 34.0, dtype=np.float32)
    thermal_f = ThermalFrame(timestamp=1.0, temperature_matrix_c=thermal_mat)

    # 1. Baseline resting step
    sample = engine.fuse_step(
        timestamp=1.0,
        rgb_pulse=0.15,
        rgb_hr=72.0,
        rgb_confidence=0.92,
        nir_frame=nir_f,
        thermal_frame=thermal_f,
    )

    assert isinstance(sample, MultimodalFusedSample)
    assert sample.casi_stress_score >= 10.0
    assert sample.casi_stress_score <= 50.0
    assert sample.nir_pulse_sample == 150.0
    assert "rgb_camera" in sample.modality_confidence
    assert sample.modality_confidence["nir_optical"] == 0.9

    # 2. Elevated strain step (high HR + vasoconstriction temperature drop)
    thermal_stress = np.full((80, 80), 32.0, dtype=np.float32)  # 2.5 deg drop (vasoconstriction)
    thermal_stress_f = ThermalFrame(timestamp=2.0, temperature_matrix_c=thermal_stress)

    strain_sample = engine.fuse_step(
        timestamp=2.0,
        rgb_pulse=-0.4,
        rgb_hr=115.0,  # Elevated HR
        rgb_confidence=0.95,
        nir_frame=nir_f,
        thermal_frame=thermal_stress_f,
    )

    assert strain_sample.casi_stress_score > sample.casi_stress_score
    assert "Sympathetic" in strain_sample.autonomic_dominance
