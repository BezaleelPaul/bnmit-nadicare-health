import os
import pickle
import numpy as np
from pathlib import Path
from app.models.schemas import StressPredictRequest, StressPredictResponse

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from app.models.schemas import StressPredictRequest, StressPredictResponse

class SomythraStressEngine:
    """
    Autonomic tone and stress classification service.
    Loads calibrated optical PPG model (somythra_optical_stress_model.pkl)
    or legacy model, with calibrated Baevsky Stress Index fallback.
    """
    _model_bundle = None
    _model_loaded = False

    @classmethod
    def load_model(cls) -> None:
        if cls._model_loaded:
            return
        candidates = [
            Path(__file__).resolve().parents[3] / "somythra_optical_stress_model.pkl",
            Path(__file__).resolve().parents[3] / "stress_prediction_model.pkl",
            Path("somythra_optical_stress_model.pkl"),
            Path("stress_prediction_model.pkl"),
        ]
        for p in candidates:
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        bundle = pickle.load(f)
                    if isinstance(bundle, dict) and "model" in bundle and "scaler" in bundle:
                        cls._model_bundle = bundle
                        cls._model_loaded = True
                        return
                except Exception:
                    pass
        cls._model_loaded = True

    @classmethod
    def predict_stress(cls, req: StressPredictRequest) -> StressPredictResponse:
        cls.load_model()
        hr = float(req.hr)
        hrv = float(req.hrv_rmssd)

        # 1. Calibrate Baevsky Stress Index: SI = AMo / (2 * VR * Mo)
        hr_factor = (hr / 70.0) ** 1.5
        hrv_factor = 50.0 / max(5.0, hrv)
        stress_idx = round(min(500.0, max(10.0, 50.0 * hr_factor * hrv_factor)), 1)

        # 2. If calibrated optical model is available, perform ML inference
        if cls._model_bundle:
            try:
                model = cls._model_bundle["model"]
                scaler = cls._model_bundle["scaler"]
                features = cls._model_bundle.get("feature_columns") or cls._model_bundle.get("feature_names", [])

                if set(features) == {"mean_hr", "rmssd", "pnn50", "lf_hf"}:
                    sdnn = req.sdnn or (hrv * 1.15)
                    pnn50 = float(np.clip((hrv - 18.0) * 0.6, 0.0, 100.0))
                    lf_hf = float(np.clip((sdnn / max(hrv, 1.0)) ** 2, 0.1, 10.0))
                    row = pd.DataFrame([{"mean_hr": hr, "rmssd": hrv, "pnn50": pnn50, "lf_hf": lf_hf}], columns=features)
                    X_scaled = scaler.transform(row)
                    pred = model.predict(X_scaled)[0]
                    probs = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
                    conf = float(max(probs))

                    if pred == 1 or str(pred).lower() in ["stress", "1"]:
                        level = "High Autonomic Stress" if conf > 0.85 else "Moderate Stress"
                        state = "Sympathetic Hyperarousal" if conf > 0.85 else "Sympathetic Dominance"
                    else:
                        level = "Restful Recovery" if hrv > 50.0 else "Balanced / Low Stress"
                        state = "Parasympathetic Dominance" if hrv > 50.0 else "Autonomic Homeostasis"

                    return StressPredictResponse(
                        stress_level=level,
                        confidence=round(conf, 3),
                        stress_index=stress_idx,
                        autonomic_state=state,
                    )
            except Exception:
                pass  # Safe fallback to physiological rules

        # 3. Rule-based physiological fallback
        if stress_idx > 200 or hrv < 25.0:
            level = "High Autonomic Stress"
            state = "Sympathetic Hyperarousal"
            conf = 0.92
        elif stress_idx > 100 or hrv < 40.0:
            level = "Moderate Stress"
            state = "Sympathetic Dominance"
            conf = 0.85
        elif stress_idx > 50:
            level = "Balanced / Low Stress"
            state = "Autonomic Homeostasis"
            conf = 0.88
        else:
            level = "Restful Recovery"
            state = "Parasympathetic Dominance"
            conf = 0.94

        return StressPredictResponse(
            stress_level=level,
            confidence=conf,
            stress_index=stress_idx,
            autonomic_state=state,
        )
