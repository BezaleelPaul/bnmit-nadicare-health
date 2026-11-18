import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.schemas import AlertLevel

client = TestClient(app)

def test_root_and_health():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["project"] == "SOMYTHRA"

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

def test_twin_simulate():
    payload = {
        "profile": {
            "name": "Alex",
            "age": 28,
            "weight_kg": 68.0,
            "baseline_hr": 62.0,
            "baseline_hrv": 58.0,
            "fitness_level": "intermediate",
        },
        "peak_hr": 165.0,
        "duration_seconds": 180,
        "time_step_seconds": 10,
    }
    res = client.post("/api/v1/twin/simulate", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert len(body["points"]) == 19
    assert body["points"][0]["predicted_hr"] == 165.0
    assert body["points"][-1]["predicted_hr"] < 165.0
    assert body["half_recovery_sec"] > 0

def test_ces_calculation():
    payload = {
        "actual_hr": 70.0,
        "predicted_hr": 70.0,
        "actual_hrv": 60.0,
        "baseline_hrv": 60.0,
        "activity_load": 1.0,
    }
    res = client.post("/api/v1/metrics/ces", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body["score"] <= 100
    assert "Optimal" in body["rating"] or "Balanced" in body["rating"]

def test_safety_check_safe():
    payload = {
        "hr": 72.0,
        "hrv_rmssd": 52.0,
        "profile": {
            "name": "Alex",
            "age": 25,
            "weight_kg": 70.0,
            "baseline_hr": 65.0,
            "baseline_hrv": 55.0,
            "fitness_level": "intermediate",
        },
    }
    res = client.post("/api/v1/safety/check", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["level"] == AlertLevel.SAFE.value
    assert body["sensor_valid"] is True

def test_safety_check_critical_hr():
    # 25 year old: max HR Tanaka = 208 - 0.7 * 25 = 190.5. 90% is ~171.45.
    payload = {
        "hr": 185.0,
        "hrv_rmssd": 45.0,
        "profile": {
            "name": "Alex",
            "age": 25,
            "weight_kg": 70.0,
            "baseline_hr": 65.0,
            "baseline_hrv": 55.0,
        },
    }
    res = client.post("/api/v1/safety/check", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["level"] == AlertLevel.CRITICAL.value
    assert body["is_critical"] is True

def test_safety_check_invalid_reading():
    payload = {
        "hr": 290.0,  # Beyond MAX_HR (250)
        "hrv_rmssd": 50.0,
    }
    res = client.post("/api/v1/safety/check", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["sensor_valid"] is False
    assert "Sensor Input Fault" in body["message"]

def test_stress_prediction():
    payload = {
        "hr": 82.0,
        "hrv_rmssd": 38.0,
        "activity_load": 2.0,
    }
    res = client.post("/api/v1/stress/predict", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert "stress_level" in body
    assert body["stress_index"] > 0

def test_vision_analyze_frame():
    import cv2
    import numpy as np
    import io

    # Create synthetic test frame
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.ellipse(frame, (320, 240), (120, 150), 0, 0, 360, (110, 130, 180), -1)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok

    files = {"file": ("test.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}
    res = client.post("/api/v1/vision/analyze-frame", files=files)
    assert res.status_code == 200
    data = res.json()
    assert "frame_index" in data
    assert "pulse_sample" in data
    assert "quality_status" in data

def test_vision_websocket_stream():
    import cv2
    import numpy as np

    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.ellipse(frame, (320, 240), (120, 150), 0, 0, 360, (110, 130, 180), -1)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    jpg_bytes = buf.tobytes()

    with client.websocket_connect("/ws/vision/live") as ws:
        ws.send_bytes(jpg_bytes)
        resp_txt = ws.receive_text()
        import json
        packet = json.loads(resp_txt)
        assert "frame_index" in packet
        assert packet["frame_index"] == 1
        assert "pulse_sample" in packet

def test_multimodal_fuse_endpoint():
    payload = {
        "timestamp": 10.5,
        "rgb_pulse": 0.22,
        "rgb_hr": 78.0,
        "rgb_confidence": 0.95,
        "nir_value": 160.0,
        "thermal_nasal_temp": 32.5,
        "thermal_forehead_temp": 34.8,
    }
    res = client.post("/api/v1/multimodal/fuse", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "casi_stress_score" in data
    assert "autonomic_dominance" in data
    assert "modality_confidence" in data
    assert data["casi_stress_score"] > 0.0
