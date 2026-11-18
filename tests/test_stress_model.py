import os
import sys
from pathlib import Path
import pytest
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import UserProfile
from src.stress_predictor import CardiacIntelligence, AlertLevel


def test_optical_ppg_model_artifact_integrity():
    """Verify that the WESAD optical PPG model artifact has all required components and valid LOSO metrics."""
    model_path = ROOT / "somythra_optical_stress_model.pkl"
    assert model_path.exists(), "Model artifact somythra_optical_stress_model.pkl should exist in root."

    artifact = joblib.load(str(model_path))
    assert isinstance(artifact, dict)
    assert "model" in artifact
    assert "scaler" in artifact
    assert "feature_columns" in artifact
    assert "classes" in artifact
    assert "loso_mean_accuracy" in artifact

    # Check optical features
    feature_cols = artifact["feature_columns"]
    assert feature_cols == ["mean_hr", "rmssd", "pnn50", "lf_hf"], f"Unexpected features: {feature_cols}"

    # Verify true human LOSO cross-validation score
    assert artifact["loso_mean_accuracy"] >= 0.80, f"Expected LOSO accuracy >= 0.80, got {artifact['loso_mean_accuracy']}"
    assert artifact["loso_mean_roc_auc"] >= 0.85, f"Expected LOSO ROC-AUC >= 0.85, got {artifact['loso_mean_roc_auc']}"
    assert "stress" in artifact["classes"]


def test_cardiac_intelligence_optical_ppg_inference():
    """Verify that CardiacIntelligence produces valid predictions using the optical PPG model."""
    profile = UserProfile(name="Test Subject", age=30, weight_kg=70, baseline_hr=60, baseline_hrv=65)
    ci = CardiacIntelligence(profile, model_path=str(ROOT / "somythra_optical_stress_model.pkl"))
    assert ci.model_bundle is not None

    # Normal resting condition (low HR, healthy high RMSSD)
    alert_normal = ci.analyze(hr=62.0, hrv_rmssd=60.0, mean_rr=967.0, sdnn=68.0)
    assert alert_normal.level in [AlertLevel.SAFE, AlertLevel.WARNING]
    assert alert_normal.risk_score < 75.0

    # High stress condition (elevated HR, suppressed RMSSD)
    alert_stress = ci.analyze(hr=110.0, hrv_rmssd=12.0, mean_rr=545.0, sdnn=18.0)
    assert alert_stress.level in [AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.STRESS_HIGH]
    assert alert_stress.risk_score > 50.0

    # Sensor failure guard (NaN input)
    alert_fail = ci.analyze(hr=float("nan"), hrv_rmssd=50.0)
    assert alert_fail.level == AlertLevel.CRITICAL
    assert alert_fail.triggered_by == "SENSOR"


def test_legacy_stress_model_artifact_integrity():
    """Verify backward compatibility of the legacy 34-feature model."""
    model_path = ROOT / "stress_prediction_model.pkl"
    assert model_path.exists(), "Model artifact stress_prediction_model.pkl should exist in root."

    artifact = joblib.load(str(model_path))
    assert isinstance(artifact, dict)
    assert "model" in artifact
    assert "scaler" in artifact
    assert "feature_columns" in artifact
    assert len(artifact["feature_columns"]) == 34
    assert artifact["test_accuracy"] >= 0.95
