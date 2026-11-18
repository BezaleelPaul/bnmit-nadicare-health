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


def test_stress_model_artifact_integrity():
    """Verify that the trained custom model artifact has all required components and metadata."""
    model_path = ROOT / "stress_prediction_model.pkl"
    assert model_path.exists(), "Model artifact stress_prediction_model.pkl should exist in root."

    artifact = joblib.load(str(model_path))
    assert isinstance(artifact, dict)
    assert "model" in artifact
    assert "scaler" in artifact
    assert "label_encoder" in artifact
    assert "feature_columns" in artifact
    assert "test_accuracy" in artifact

    # Check features and classes
    feature_cols = artifact["feature_columns"]
    assert len(feature_cols) == 34, f"Expected 34 physiological features, got {len(feature_cols)}"
    assert "datasetId" not in feature_cols, "Data leakage: datasetId should not be a feature!"

    expected_classes = {"interruption", "no stress", "time pressure"}
    assert set(artifact["label_encoder"].classes_) == expected_classes
    assert artifact["test_accuracy"] >= 0.95, f"Expected test accuracy >= 0.95, got {artifact['test_accuracy']}"


def test_cardiac_intelligence_ml_stress_prediction():
    """Verify that CardiacIntelligence loads the model and produces valid predictions."""
    profile = UserProfile(name="Test Subject", age=30, weight_kg=70, baseline_hr=60, baseline_hrv=65)
    ci = CardiacIntelligence(profile, model_path=str(ROOT / "stress_prediction_model.pkl"))
    assert ci.model_bundle is not None

    # Normal resting condition
    alert_normal = ci.analyze(hr=62.0, hrv_rmssd=60.0, mean_rr=967.0, sdnn=68.0)
    assert alert_normal.level in [AlertLevel.SAFE, AlertLevel.WARNING]
    assert alert_normal.risk_score < 75.0

    # Sensor failure guard (NaN input)
    alert_fail = ci.analyze(hr=float("nan"), hrv_rmssd=50.0)
    assert alert_fail.level == AlertLevel.CRITICAL
    assert alert_fail.triggered_by == "SENSOR"


def test_cardiac_intelligence_feature_conversion():
    """Verify that 34 features are correctly derived from raw physiological inputs."""
    profile = UserProfile(name="Test Subject", age=30, weight_kg=70, baseline_hr=60, baseline_hrv=65)
    ci = CardiacIntelligence(profile, model_path=str(ROOT / "stress_prediction_model.pkl"))

    features = ci._hrv_to_features(rmssd=45.0, mean_rr=800.0, hr=75.0, sdnn=55.0)
    assert "MEAN_RR" in features
    assert "RMSSD" in features
    assert "HR" in features
    assert "LF_HF" in features
    assert "pNN50" in features
    assert features["MEAN_RR"] == 800.0
    assert features["RMSSD"] == 45.0
    assert features["HR"] == 75.0
