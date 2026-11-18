import asyncio
import json
import random
import tempfile
import base64
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.models.schemas import (
    UserProfile,
    RecoveryRequest,
    RecoveryResponse,
    CESRequest,
    CESResponse,
    SafetyCheckRequest,
    SafetyCheckResponse,
    StressPredictRequest,
    StressPredictResponse,
    TelemetryReading,
)
from app.engine.twin import SomythraTwinEngine
from app.engine.analytics import SomythraAnalytics
from app.engine.safety import SomythraSafetyMonitor
from app.engine.stress import SomythraStressEngine
from app.engine.vision import SomythraVisionService

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
)

# Enable CORS for Flutter web, local desktop, and mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def get_root():
    return {
        "project": settings.PROJECT_IDENTITY,
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "line": "Life is not a collection of parts. It is a system of relationships.",
        "status": "operational",
    }

@app.get("/health")
def get_health():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.VERSION,
    }

@app.post(f"{settings.API_V1_STR}/twin/simulate", response_model=RecoveryResponse)
def simulate_twin(req: RecoveryRequest):
    try:
        return SomythraTwinEngine.simulate_recovery(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"{settings.API_V1_STR}/metrics/ces", response_model=CESResponse)
def calculate_ces(req: CESRequest):
    try:
        return SomythraAnalytics.calculate_ces(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"{settings.API_V1_STR}/safety/check", response_model=SafetyCheckResponse)
def check_safety(req: SafetyCheckRequest):
    return SomythraSafetyMonitor.check_safety(req)

@app.post(f"{settings.API_V1_STR}/stress/predict", response_model=StressPredictResponse)
def predict_stress(req: StressPredictRequest):
    try:
        return SomythraStressEngine.predict_stress(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"{settings.API_V1_STR}/vision/analyze-video")
async def analyze_video(file: UploadFile = File(...), algorithm: str = "pos"):
    """
    Process an uploaded video file (MP4, AVI) through the full SOMYTHRA rPPG pipeline
    and predict autonomic stress from the resulting optical pulse metrics.
    """
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = SomythraVisionService.process_video_file(tmp_path, algorithm=algorithm)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

@app.post(f"{settings.API_V1_STR}/vision/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    """
    Process a single frame image (JPEG / PNG) through rPPG extraction.
    """
    try:
        content = await file.read()
        streamer = SomythraVisionService.create_streamer()
        return SomythraVisionService.process_frame_bytes(content, streamer)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.websocket("/ws/vision/live")
async def websocket_vision_live(websocket: WebSocket):
    """
    Bidirectional real-time rPPG streaming WebSocket.
    Client sends raw JPEG/PNG frame bytes or base64 encoded strings;
    Server returns 30 FPS pulse waveforms, SNR, instantaneous BPM, and autonomic stress state.
    """
    await websocket.accept()
    streamer = SomythraVisionService.create_streamer(
        window_duration=8.0, cadence_interval=1.0, fps=30.0, compensate_ambient=True
    )

    try:
        while True:
            data = await websocket.receive()
            if "bytes" in data and data["bytes"]:
                frame_bytes = data["bytes"]
            elif "text" in data and data["text"]:
                txt = data["text"].strip()
                # If JSON sent with base64 'image' field
                if txt.startswith("{"):
                    parsed = json.loads(txt)
                    b64_str = parsed.get("image", "")
                    if "," in b64_str:
                        b64_str = b64_str.split(",", 1)[1]
                    frame_bytes = base64.b64decode(b64_str)
                else:
                    if "," in txt:
                        txt = txt.split(",", 1)[1]
                    frame_bytes = base64.b64decode(txt)
            else:
                continue

            packet = SomythraVisionService.process_frame_bytes(frame_bytes, streamer)
            await websocket.send_text(json.dumps(packet))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """
    WebSocket endpoint streaming live synthetic or hardware-interfaced physiological telemetry
    to the Flutter cross-platform client in real time (every 1 second).
    """
    await websocket.accept()
    base_hr = 68.0
    base_hrv = 55.0
    load = 1.0

    try:
        while True:
            # Generate realistic fluctuating physiological parameters
            delta_hr = random.uniform(-2.5, 2.5)
            delta_hrv = random.uniform(-3.0, 3.0)
            cur_hr = round(max(50.0, min(140.0, base_hr + delta_hr)), 1)
            cur_hrv = round(max(20.0, min(95.0, base_hrv + delta_hrv)), 1)

            # Drift base slightly
            base_hr = 0.95 * base_hr + 0.05 * (65.0 + random.uniform(-5, 10))
            base_hrv = 0.95 * base_hrv + 0.05 * (55.0 + random.uniform(-5, 10))

            # Calculate safety alert
            safety_res = SomythraSafetyMonitor.check_safety(
                SafetyCheckRequest(hr=cur_hr, hrv_rmssd=cur_hrv)
            )

            # Calculate CES score against expected twin
            expected_hr = round(base_hr, 1)
            ces_res = SomythraAnalytics.calculate_ces(
                CESRequest(
                    actual_hr=cur_hr,
                    predicted_hr=expected_hr,
                    actual_hrv=cur_hrv,
                    baseline_hrv=55.0,
                    activity_load=load,
                )
            )

            # Stress calculation
            stress_res = SomythraStressEngine.predict_stress(
                StressPredictRequest(hr=cur_hr, hrv_rmssd=cur_hrv)
            )

            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "hr": cur_hr,
                "hrv": cur_hrv,
                "mean_rr": round(60000.0 / cur_hr, 1),
                "sdnn": round(cur_hrv * 1.15, 1),
                "activity_load": load,
                "ces_score": ces_res.score,
                "ces_rating": ces_res.rating,
                "safety_level": safety_res.level.value,
                "safety_message": safety_res.message,
                "stress_level": stress_res.stress_level,
                "stress_index": stress_res.stress_index,
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
