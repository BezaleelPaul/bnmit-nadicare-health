from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional, List
from pydantic import BaseModel, Field, field_validator, field_serializer
from app.core.config import settings

def _now_utc():
    return datetime.now(timezone.utc)

class AlertLevel(str, Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class FitnessLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    ELITE = "elite"

FITNESS_DECAY_RATES: dict[str, float] = {
    "beginner": 0.020,
    "intermediate": 0.040,
    "advanced": 0.065,
    "elite": 0.100,
}

class UserProfile(BaseModel):
    name: str = Field(default="Somythra Explorer", description="User's display name")
    age: int = Field(default=25, ge=10, le=100, description="Age in years")
    weight_kg: float = Field(default=70.0, ge=20, le=300, description="Weight in kg")
    baseline_hr: float = Field(default=65.0, ge=30, le=120, description="Resting baseline HR in BPM")
    baseline_hrv: float = Field(default=55.0, ge=5, le=200, description="Resting baseline HRV in ms (RMSSD)")
    fitness_level: FitnessLevel = Field(default=FitnessLevel.INTERMEDIATE, description="Fitness tier")

    @property
    def max_hr(self) -> float:
        """Age-predicted maximum HR using Tanaka formula: 208 - 0.7 * age."""
        return round(settings.TANAKA_BASE - settings.TANAKA_FACTOR * self.age, 1)

    @property
    def critical_hr_threshold(self) -> float:
        return round(self.max_hr * settings.CRITICAL_HR_RATIO, 1)

    @property
    def warning_hr_threshold(self) -> float:
        return round(self.max_hr * settings.WARNING_HR_RATIO, 1)

    @property
    def twin_decay_rate(self) -> float:
        return FITNESS_DECAY_RATES.get(self.fitness_level.value, 0.040)

class TelemetryReading(BaseModel):
    timestamp: datetime = Field(default_factory=_now_utc)
    hr: float = Field(..., ge=20, le=250, description="Heart rate in BPM")
    hrv: float = Field(..., ge=0, le=200, description="HRV (RMSSD) in ms")
    activity_load: float = Field(default=0.0, ge=0.0, le=10.0, description="Activity intensity (0-10)")
    mean_rr: Optional[float] = Field(default=None, description="Mean RR interval in ms")
    sdnn: Optional[float] = Field(default=None, description="SDNN in ms")
    temperature: Optional[float] = Field(default=None, description="Skin/body temp in Celsius")
    signal_quality: float = Field(default=1.0, ge=0.0, le=1.0, description="Signal quality confidence")

    @field_serializer("timestamp")
    def serialize_timestamp(self, v: datetime) -> str:
        return v.isoformat()

class RecoveryRequest(BaseModel):
    profile: UserProfile
    peak_hr: float = Field(default=160.0, ge=40, le=230)
    duration_seconds: int = Field(default=300, ge=30, le=3600)
    time_step_seconds: int = Field(default=5, ge=1, le=60)

class RecoveryPoint(BaseModel):
    time_sec: int
    predicted_hr: float
    recovery_pct: float

class RecoveryResponse(BaseModel):
    points: List[RecoveryPoint]
    half_recovery_sec: float
    full_recovery_estimate_sec: float
    decay_constant: float
    max_hr: float

class CESRequest(BaseModel):
    actual_hr: float = Field(..., ge=20, le=250)
    predicted_hr: float = Field(..., ge=20, le=250)
    actual_hrv: float = Field(..., ge=1, le=200)
    baseline_hrv: float = Field(default=50.0, ge=5, le=200)
    activity_load: float = Field(default=1.0, ge=0, le=10)

class CESResponse(BaseModel):
    score: float = Field(..., ge=0, le=100)
    rating: str
    cardiac_strain_pct: float
    autonomic_balance_pct: float
    load_penalty: float
    summary: str

class SafetyCheckRequest(BaseModel):
    hr: float
    hrv_rmssd: float
    profile: Optional[UserProfile] = None
    mean_rr: Optional[float] = None
    sdnn: Optional[float] = None
    temperature: Optional[float] = None

class SafetyCheckResponse(BaseModel):
    level: AlertLevel
    message: str
    recommendation: str
    is_critical: bool
    is_warning: bool
    sensor_valid: bool
    timestamp: datetime = Field(default_factory=_now_utc)

    @field_serializer("timestamp")
    def serialize_timestamp(self, v: datetime) -> str:
        return v.isoformat()

class StressPredictRequest(BaseModel):
    hr: float = Field(default=75.0, ge=30, le=220)
    hrv_rmssd: float = Field(default=45.0, ge=5, le=200)
    mean_rr: Optional[float] = None
    sdnn: Optional[float] = None
    activity_load: float = Field(default=1.0, ge=0, le=10)

class StressPredictResponse(BaseModel):
    stress_level: str
    confidence: float
    stress_index: float
    autonomic_state: str
