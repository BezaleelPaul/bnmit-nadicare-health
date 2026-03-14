import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path so 'src' and 'data_gen' can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import UserProfile
from src.analytics import cardiac_enhancement_score
from src.safety_monitor import check_safety_boundaries, AlertLevel
from data_gen import generate_24h_demo


def test_generate_24h_demo_has_expected_shape_and_columns(tmp_path):
    """Demo generator should produce 1440 rows and all required columns."""
    out_path = tmp_path / "demo_test.csv"
    df = generate_24h_demo(output_path=str(out_path))

    assert len(df) == 1440
    for col in ["timestamp", "hr", "hrv", "activity_load", "label", "strategy", "safety_level"]:
        assert col in df.columns

    # All timestamps should be monotonically increasing
    assert pd.to_datetime(df["timestamp"]).is_monotonic_increasing


def test_ces_is_bounded_and_behaves_sensibly():
    """CES must always be within [0, 100] and respond to HR/HRV changes."""
    baseline_hrv = 60.0

    # Ideal case: actual == predicted, HRV == baseline, low load
    ces_ideal = cardiac_enhancement_score(
        actual_hr=70,
        predicted_hr=70,
        actual_hrv=baseline_hrv,
        baseline_hrv=baseline_hrv,
        activity_load=1.0,
    )
    assert 0 <= ces_ideal <= 100

    # Worse HR and lower HRV should reduce CES
    ces_stressed = cardiac_enhancement_score(
        actual_hr=90,
        predicted_hr=70,
        actual_hrv=baseline_hrv * 0.5,
        baseline_hrv=baseline_hrv,
        activity_load=8.0,
    )
    assert 0 <= ces_stressed <= 100
    assert ces_stressed < ces_ideal


def test_safety_monitor_hr_thresholds():
    """Safety monitor should escalate from SAFE → WARNING → CRITICAL with HR."""
    profile = UserProfile(
        name="Test",
        age=30,
        weight_kg=70,
        baseline_hr=60,
        baseline_hrv=60,
    )
    max_hr = profile.max_hr

    # Clearly safe HR and HRV
    alert_safe = check_safety_boundaries(hr=max_hr * 0.5, hrv=60, profile=profile)
    assert alert_safe.level == AlertLevel.SAFE

    # Warning zone HR (>80% max) but safe HRV
    alert_warning = check_safety_boundaries(hr=max_hr * 0.82, hrv=60, profile=profile)
    assert alert_warning.level == AlertLevel.WARNING

    # Critical HR (>95% max) irrespective of HRV
    alert_critical = check_safety_boundaries(hr=max_hr * 0.97, hrv=60, profile=profile)
    assert alert_critical.level == AlertLevel.CRITICAL


def test_safety_monitor_hrv_thresholds():
    """Very low HRV should trigger WARNING/CRITICAL even at modest HR."""
    profile = UserProfile(
        name="Test",
        age=40,
        weight_kg=80,
        baseline_hr=65,
        baseline_hrv=50,
    )

    # HRV between 20 and 30 ms → WARNING
    alert_warning = check_safety_boundaries(hr=profile.baseline_hr + 5, hrv=25, profile=profile)
    assert alert_warning.level == AlertLevel.WARNING

    # HRV < 20 ms → CRITICAL
    alert_critical = check_safety_boundaries(hr=profile.baseline_hr + 5, hrv=15, profile=profile)
    assert alert_critical.level == AlertLevel.CRITICAL

