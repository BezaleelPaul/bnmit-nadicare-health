# src/models.py
# Pydantic data schemas for NadiCare Digital Twin
# Defines validated data structures used across the entire pipeline

from pydantic import BaseModel, Field, ConfigDict, field_validator, field_serializer
from datetime import datetime
from typing import Literal, Optional


# ── Shared threshold constants ───────────────────────────────────────────────
# Single source of truth. analytics.py and safety_monitor.py import from here.
# Never redefine these values elsewhere.

HRV_CRITICAL_MS: float = 25.0    # ms — severe autonomic stress → CRITICAL
HRV_WARNING_MS:  float = 35.0    # ms — elevated sympathetic tone → WARNING
HRV_HIGH_MS:     float = 100.0   # ms — unusually high; verify sensor

HR_CRITICAL_PCT: float = 0.90    # fraction of max HR → CRITICAL
HR_WARNING_PCT:  float = 0.75    # fraction of max HR → WARNING

# Fitness-level decay rates for the Digital Twin (1/s).
# Higher = faster HR recovery = better fitness.
FITNESS_DECAY_RATES: dict[str, float] = {
    "beginner":     0.020,
    "intermediate": 0.040,
    "advanced":     0.065,
    "elite":        0.100,
}


class HeartData(BaseModel):
    """
    Schema for a single heart data reading.
    Represents one timestamped snapshot of the user's cardiac state.
    """
    timestamp:     datetime = Field(..., description="UTC timestamp of the reading")
    hr:            float    = Field(..., ge=20,  le=250, description="Heart Rate in BPM")
    hrv:           float    = Field(..., ge=0,   le=200, description="HRV in ms (RMSSD)")
    activity_load: float    = Field(..., ge=0,   le=10,  description="Activity load 0–10")
    label:         Optional[str] = Field(None, description="Optional event label")

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v

    @field_serializer("timestamp")
    def serialize_timestamp(self, v: datetime):
        return v.isoformat()

    model_config = ConfigDict()


class UserProfile(BaseModel):
    """
    Schema for user biometric profile.
    Used to personalise safety thresholds and Twin calibration.
    """
    name:          str   = Field(default="User", description="User's name")
    age:           int   = Field(..., ge=10, le=100, description="Age in years")
    weight_kg:     float = Field(..., ge=20, le=300, description="Weight in kg")
    baseline_hr:   float = Field(..., ge=30, le=120, description="Resting HR in BPM")
    baseline_hrv:  float = Field(default=50.0, ge=5, le=200, description="Resting HRV in ms")
    fitness_level: Literal["beginner", "intermediate", "advanced", "elite"] = Field(
        default="intermediate",
        description="Fitness level — used to personalise Twin decay rate and context",
    )

    @property
    def max_hr(self) -> float:
        """
        Age-predicted maximum heart rate using the Tanaka formula (2001).
        Tanaka formula: 208 - 0.7 × age
        Validated across 351 studies (n=18,712) — more accurate than Fox (220-age),
        which overestimates max HR in older adults by 5-7 BPM.
        """
        return round(208.0 - 0.7 * self.age, 1)

    @property
    def critical_hr_threshold(self) -> float:
        """
        HR above which training is CRITICAL — 90% of max HR.
        Aligned with safety_monitor.py HR_CRITICAL_PCT constant.
        """
        return round(self.max_hr * HR_CRITICAL_PCT, 1)

    @property
    def warning_hr_threshold(self) -> float:
        """
        HR above which training is WARNING — 75% of max HR.
        Aligned with safety_monitor.py HR_WARNING_PCT constant.
        """
        return round(self.max_hr * HR_WARNING_PCT, 1)

    @property
    def twin_decay_rate(self) -> float:
        """
        Personalised decay rate (lambda) for the Digital Twin exponential model.
        Looked up from FITNESS_DECAY_RATES by fitness_level.
        """
        return FITNESS_DECAY_RATES[self.fitness_level]


class TwinState(BaseModel):
    """
    Represents the Digital Twin's predicted physiological state at a given moment.
    """
    timestamp:     datetime
    predicted_hr:  float = Field(..., description="Twin-predicted HR in BPM")
    actual_hr:     float = Field(..., description="Measured HR in BPM")
    predicted_hrv: float = Field(..., description="Twin-predicted HRV in ms")
    actual_hrv:    float = Field(..., description="Measured HRV in ms")
    ces_score:     float = Field(..., ge=0, le=100, description="Cardiac Enhancement Score")
    alert:         Optional[str] = Field(None, description="Safety alert message if triggered")
