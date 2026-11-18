import math
from typing import List
from app.models.schemas import (
    UserProfile,
    RecoveryRequest,
    RecoveryPoint,
    RecoveryResponse,
)

class SomythraTwinEngine:
    """
    Physiological digital-twin recovery model.
    Models cardiac recovery dynamics post-exercise using exponential decay:
    HR(t) = HR_base + (HR_peak - HR_base) * exp(-lambda * t)
    """

    @staticmethod
    def predict_hr(t_seconds: float, baseline_hr: float, peak_hr: float, decay_rate: float) -> float:
        """Calculate predicted HR at elapsed time t in seconds."""
        if t_seconds < 0:
            return peak_hr
        delta = max(0.0, peak_hr - baseline_hr)
        decayed = delta * math.exp(-decay_rate * t_seconds)
        return round(baseline_hr + decayed, 1)

    @classmethod
    def simulate_recovery(cls, req: RecoveryRequest) -> RecoveryResponse:
        """Simulate continuous recovery points given profile and peak HR."""
        decay_rate = req.profile.twin_decay_rate
        baseline = req.profile.baseline_hr
        peak = max(req.peak_hr, baseline)
        total_delta = max(1.0, peak - baseline)

        points: List[RecoveryPoint] = []
        for t in range(0, req.duration_seconds + 1, req.time_step_seconds):
            predicted = cls.predict_hr(t, baseline, peak, decay_rate)
            recovered_amount = peak - predicted
            pct = min(100.0, max(0.0, (recovered_amount / total_delta) * 100.0))
            points.append(
                RecoveryPoint(
                    time_sec=t,
                    predicted_hr=predicted,
                    recovery_pct=round(pct, 1),
                )
            )

        half_recovery = round(math.log(2) / decay_rate, 1)
        full_estimate = round(3.0 / decay_rate, 1)

        return RecoveryResponse(
            points=points,
            half_recovery_sec=half_recovery,
            full_recovery_estimate_sec=full_estimate,
            decay_constant=decay_rate,
            max_hr=req.profile.max_hr,
        )
