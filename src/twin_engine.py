# src/twin_engine.py
# Digital Twin Engine — models physiological homeostasis using exponential decay.
# Core equation: HR(t) = HR_baseline + (HR_peak - HR_baseline) × exp(−λ × t)

import numpy as np
from datetime import datetime
from src.models import UserProfile, HeartData, TwinState


class DigitalTwin:
    """
    Simulates the expected cardiac response of an individual.

    Uses a first-order exponential decay model:
        HR(t)  = HR_baseline  + (HR_peak  − HR_baseline)  × exp(−λ × t)
        HRV(t) = HRV_baseline − (HRV_baseline − HRV_dip)  × exp(−λ × t)

    The decay_rate λ is personalised automatically from UserProfile.fitness_level
    via the FITNESS_DECAY_RATES lookup in models.py. Callers do not need to set it.
    """

    def __init__(self, profile: UserProfile, decay_rate: float | None = None):
        """
        Args:
            profile:    UserProfile — provides baseline HR/HRV and fitness level.
            decay_rate: Override λ (1/s). If None, uses profile.twin_decay_rate
                        which is looked up from fitness_level.
                        Typical range: 0.02 (beginner) → 0.10 (elite).
        """
        if decay_rate is not None and decay_rate <= 0:
            raise ValueError(f"decay_rate must be > 0, got {decay_rate}")

        self.profile    = profile
        # BUG-20 fix: auto-set from fitness_level instead of hardcoding 0.05
        self.decay_rate = decay_rate if decay_rate is not None else profile.twin_decay_rate

        # Internal state: set to baseline until apply_stress_event() is called
        self._peak_hr:  float = profile.baseline_hr
        self._peak_hrv: float = profile.baseline_hrv
        # BUG-17 fix: _event_time removed — it was written but never read.
        # seconds_elapsed is passed directly to predict(), which is the correct pattern.

    def apply_stress_event(self, hr_peak: float, hrv_dip: float) -> None:
        """
        Record a stress event (e.g. sprint, anxiety spike) so predict() can
        model recovery from this new peak state.

        Args:
            hr_peak:  Maximum HR reached during the event (BPM).
                      Must be ≥ profile.baseline_hr.
            hrv_dip:  Minimum HRV reached during the event (ms). Must be ≥ 0.
        """
        if hr_peak < self.profile.baseline_hr:
            raise ValueError(
                f"hr_peak ({hr_peak}) cannot be below baseline HR "
                f"({self.profile.baseline_hr})"
            )
        if hrv_dip < 0:
            raise ValueError(f"hrv_dip cannot be negative, got {hrv_dip}")

        self._peak_hr  = hr_peak
        self._peak_hrv = hrv_dip

    def predict(self, seconds_elapsed: float) -> tuple[float, float]:
        """
        Predict HR and HRV at `seconds_elapsed` seconds after the most recent
        stress event was applied via apply_stress_event().

        Args:
            seconds_elapsed: Time since stress event (seconds). Must be ≥ 0.

        Returns:
            (predicted_hr, predicted_hrv) rounded to 2 decimal places.
        """
        if seconds_elapsed < 0:
            raise ValueError(f"seconds_elapsed must be ≥ 0, got {seconds_elapsed}")

        decay = np.exp(-self.decay_rate * seconds_elapsed)

        predicted_hr = (
            self.profile.baseline_hr
            + (self._peak_hr - self.profile.baseline_hr) * decay
        )

        # HRV recovers inversely: rises from dip back toward baseline
        predicted_hrv = (
            self.profile.baseline_hrv
            - (self.profile.baseline_hrv - self._peak_hrv) * decay
        )

        return round(float(predicted_hr), 2), round(float(predicted_hrv), 2)

    def generate_recovery_curve(
        self,
        duration_seconds: int = 600,
        step: int = 10,
    ) -> dict:
        """
        Generate the full predicted recovery curve from t=0 to t=duration_seconds.

        Args:
            duration_seconds: Total window in seconds (default 10 min). Must be > 0.
            step:             Sampling interval in seconds. Must be > 0.

        Returns:
            Dict with keys:
                'time_seconds'  : list[float] — t values from 0 to duration_seconds inclusive
                'predicted_hr'  : list[float]
                'predicted_hrv' : list[float]
        """
        if duration_seconds <= 0:
            raise ValueError(f"duration_seconds must be > 0, got {duration_seconds}")
        if step <= 0:
            raise ValueError(f"step must be > 0, got {step}")

        # BUG-18 fix: arange(0, duration+step, step) overshoots by one step.
        # Use linspace or integer range to get exactly [0, step, 2*step, ..., duration].
        n_points = duration_seconds // step + 1
        times    = [float(i * step) for i in range(n_points)]
        # Ensure the final point is exactly duration_seconds (guards floating-point drift)
        if times[-1] != float(duration_seconds):
            times.append(float(duration_seconds))

        predicted_hrs  = []
        predicted_hrvs = []
        for t in times:
            hr, hrv = self.predict(t)
            predicted_hrs.append(hr)
            predicted_hrvs.append(hrv)

        return {
            "time_seconds":  times,
            "predicted_hr":  predicted_hrs,
            "predicted_hrv": predicted_hrvs,
        }

    def update_decay_rate(self, actual_recovery_time_s: float) -> None:
        """
        Self-calibrate λ from observed recovery data.

        The decay_rate is the reciprocal of the time-constant (τ), defined as
        the time for the signal to close 63.2% of the gap back to baseline.
        This is the standard definition of a first-order system time constant.

            λ = 1 / τ₆₃

        Args:
            actual_recovery_time_s: Seconds observed for HR to drop by 63.2%
                                    of (HR_peak − HR_baseline). Must be > 0.

        Note:
            BUG-19 fix: removed the unused `delta_hr` parameter — it was
            documented as required but was never used in the calculation.
            The 1/τ formula is independent of amplitude; only time matters.
        """
        if actual_recovery_time_s <= 0:
            raise ValueError(
                f"actual_recovery_time_s must be > 0, got {actual_recovery_time_s}"
            )
        self.decay_rate = round(1.0 / actual_recovery_time_s, 6)

    @property
    def estimated_recovery_seconds(self) -> float:
        """
        Estimate seconds to reach 95% recovery (≈ 3 time constants).
        Useful for telling users "you should be recovered in X minutes".
        """
        # 3τ ≈ 95% recovery for a first-order system
        tau_95 = 3.0 / self.decay_rate
        return round(tau_95, 1)
