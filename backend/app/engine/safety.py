import math
from typing import Optional
from app.models.schemas import (
    AlertLevel,
    UserProfile,
    SafetyCheckRequest,
    SafetyCheckResponse,
)
from app.core.config import settings

class SomythraSafetyMonitor:
    """
    Physiological safety monitoring adhering to SOMYTHRA principles:
    - Never clamp sensor faults into fake plausible values.
    - Expose missing, noisy, or implausible signals.
    - Tanaka age-adjusted heart rate thresholds.
    """

    @classmethod
    def validate_reading(
        cls,
        hr: float,
        hrv_rmssd: float,
        mean_rr: Optional[float] = None,
        sdnn: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """Validate all numeric values are finite and within biological observation limits."""
        numeric_dict = {
            "hr": hr,
            "hrv_rmssd": hrv_rmssd,
            "mean_rr": mean_rr,
            "sdnn": sdnn,
            "temperature": temperature,
        }
        for name, val in numeric_dict.items():
            if val is None:
                continue
            try:
                finite = math.isfinite(float(val))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                raise ValueError(f"{name} must be a finite number")

        if not settings.MIN_HR <= float(hr) <= settings.MAX_HR:
            raise ValueError(f"hr must be between {settings.MIN_HR} and {settings.MAX_HR} BPM")
        if not settings.MIN_HRV <= float(hrv_rmssd) <= settings.MAX_HRV:
            raise ValueError(f"hrv_rmssd must be between {settings.MIN_HRV} and {settings.MAX_HRV} ms")
        if mean_rr is not None and float(mean_rr) <= 0:
            raise ValueError("mean_rr must be greater than 0")
        if sdnn is not None and float(sdnn) <= 0:
            raise ValueError("sdnn must be greater than 0")

    @classmethod
    def check_safety(cls, req: SafetyCheckRequest) -> SafetyCheckResponse:
        """Evaluate cardiac state against age-adjusted Tanaka and autonomic boundaries."""
        try:
            cls.validate_reading(
                hr=req.hr,
                hrv_rmssd=req.hrv_rmssd,
                mean_rr=req.mean_rr,
                sdnn=req.sdnn,
                temperature=req.temperature,
            )
        except ValueError as e:
            return SafetyCheckResponse(
                level=AlertLevel.CRITICAL,
                message=f"Sensor Input Fault: {str(e)}",
                recommendation="Inspect wearable electrode/optical sensor placement. Data flagged invalid.",
                is_critical=True,
                is_warning=False,
                sensor_valid=False,
            )

        profile = req.profile or UserProfile()
        critical_hr = profile.critical_hr_threshold
        warning_hr = profile.warning_hr_threshold
        hr = float(req.hr)
        hrv = float(req.hrv_rmssd)

        # Check Critical Conditions
        if hr >= critical_hr:
            return SafetyCheckResponse(
                level=AlertLevel.CRITICAL,
                message=f"Extreme Tachycardia: Heart rate ({hr:.0f} BPM) exceeds 90% Tanaka maximum ({critical_hr:.0f} BPM).",
                recommendation="Cease vigorous exertion immediately, sit down, and engage in slow nasal breathing.",
                is_critical=True,
                is_warning=False,
                sensor_valid=True,
            )
        if hrv < settings.CRITICAL_HRV_MS:
            return SafetyCheckResponse(
                level=AlertLevel.CRITICAL,
                message=f"Severe Autonomic Depletion: HRV RMSSD ({hrv:.1f} ms) critically below {settings.CRITICAL_HRV_MS} ms.",
                recommendation="High sympathetic overactivation or acute systemic fatigue. Rest and hydrate.",
                is_critical=True,
                is_warning=False,
                sensor_valid=True,
            )

        # Check Warning Conditions
        if hr >= warning_hr:
            return SafetyCheckResponse(
                level=AlertLevel.WARNING,
                message=f"Elevated Cardiac Demand: Heart rate ({hr:.0f} BPM) exceeds 75% Tanaka threshold ({warning_hr:.0f} BPM).",
                recommendation="Moderate workload intensity to maintain physiological equilibrium.",
                is_warning=True,
                is_critical=False,
                sensor_valid=True,
            )
        if hrv < settings.WARNING_HRV_MS:
            return SafetyCheckResponse(
                level=AlertLevel.WARNING,
                message=f"Mild Autonomic Strain: HRV RMSSD ({hrv:.1f} ms) below {settings.WARNING_HRV_MS} ms.",
                recommendation="Sympathetic dominance observed. Monitor recovery rate.",
                is_warning=True,
                is_critical=False,
                sensor_valid=True,
            )

        # Normal State
        return SafetyCheckResponse(
            level=AlertLevel.SAFE,
            message="Physiological vitals stable within safe regulatory limits.",
            recommendation="Continue normal activity. Autonomic balance is well-maintained.",
            is_critical=False,
            is_warning=False,
            sensor_valid=True,
        )
