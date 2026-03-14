# src/analytics.py
# Analytics engine for the Cardiac Enhancement Score (CES) + Advanced Features
# CES quantifies cardiovascular resilience — not just raw performance.

import math
import numpy as np
import statistics
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta
from src.models import UserProfile, HRV_CRITICAL_MS, HRV_WARNING_MS, HRV_HIGH_MS

# CES is hard-capped when HRV enters CRITICAL territory
_CES_CRITICAL_HRV_CAP = 20.0
_CES_ELITE_THRESHOLD = 90.0  # New: Elite athlete benchmark

class AnalyticsEngine:
    """Centralized analytics with stateful trend tracking."""
    
    def __init__(self, user_profile: UserProfile):
        self.user_profile = user_profile
        self.ces_history: List[float] = []
        self.timestamps: List[datetime] = []
        self.activity_history: List[float] = []
    
    # 🎯 CORE CES CALCULATION (Enhanced)
    def cardiac_enhancement_score(
        self,
        actual_hr: float,
        predicted_hr: float,
        actual_hrv: float,
        baseline_hrv: Optional[float] = None,
        activity_load: float = 0.0,
        temperature: Optional[float] = None,  # 🆕 Environmental factor
    ) -> float:
        """
        Enhanced CES with environmental compensation and auto-baseline.
        """
        # Auto-fallback for baseline_hrv
        baseline_hrv = baseline_hrv or self.user_profile.baseline_hrv or 45.0
        
        # 1. Finite + Range Validation
        inputs = [actual_hr, predicted_hr, actual_hrv, baseline_hrv, activity_load]
        if temperature is not None:
            inputs.append(temperature)
        if not all(math.isfinite(x) and x >= 0 for x in inputs):
            return 0.0

        # Input bounds
        if predicted_hr <= 30 or predicted_hr > 220:
            raise ValueError(f"Invalid predicted_hr: {predicted_hr}")
        if not (0 <= activity_load <= 10):
            activity_load = max(0, min(10, activity_load))

        # 🆕 Environmental Compensation (Temperature effect on HR)
        temp_penalty = 1.0
        if temperature is not None:
            if temperature > 28:  # Hot
                temp_penalty *= 0.95 ** (temperature - 28)
            elif temperature < 10:  # Cold
                temp_penalty *= 0.98 ** (10 - temperature)

        # Component 1: HRV Stability (with recovery debt tracking)
        hrv_ratio = actual_hrv / baseline_hrv
        hrv_component = min(hrv_ratio, 2.0)
        
        # 🆕 Recovery Debt Multiplier (if recent trend is down)
        recovery_debt = self._calculate_recovery_debt()
        hrv_component *= (1.0 - min(recovery_debt * 0.1, 0.2))

        # Component 2: Asymmetric HR Efficiency
        hr_ratio = actual_hr / predicted_hr
        if hr_ratio <= 1.0:
            hr_component = min(1.0 + (1.0 - hr_ratio) * 0.5, 1.05)
        else:
            hr_deviation = (actual_hr - predicted_hr) / predicted_hr
            hr_component = max(0.0, 1.0 - hr_deviation * 1.2)  # Harsher penalty

        # Component 3: Load + Environmental Penalty
        load_penalty = 1.0 / (1.0 + activity_load * 0.02) * temp_penalty

        ces = 100.0 * hrv_component * hr_component * load_penalty

        # Safety caps
        if actual_hrv < HRV_CRITICAL_MS:
            ces = min(ces, _CES_CRITICAL_HRV_CAP)
        elif actual_hrv > 200:  # 🆕 Physiological limit
            ces = min(ces, 95.0)

        # Store for trend analysis
        self._store_reading(ces, activity_load)
        
        return round(float(np.clip(ces, 0.0, 100.0)), 2)
    
    # 🆕 INTELLIGENT PREDICTION ENGINE
    def predict_hr_for_load(
        self, 
        target_load: float, 
        recent_activity: Optional[List[Tuple[float, float]]] = None
    ) -> float:
        """
        ML-inspired HR prediction using recent data patterns.
        """
        if recent_activity:
            # Simple linear regression on load vs HR
            loads, hrs = zip(*recent_activity)
            if len(loads) > 2:
                slope, intercept = np.polyfit(loads, hrs, 1)
                return max(50, slope * target_load + intercept)
        
        # Fallback: User baseline + load scaling
        resting_hr = self.user_profile.baseline_hr or 65
        return max(50, resting_hr + target_load * 3.5)
    
    # 🎨 ENHANCED EXPLANATION WITH TRENDS
    def ces_explanation(
        self,
        ces: float,
        actual_hr: float,
        predicted_hr: float,
        actual_hrv: float,
        baseline_hrv: float,
        activity_load: float = 0.0,
        show_trends: bool = True,
    ) -> str:
        """Rich explanation with trend insights and recommendations."""
        
        if not math.isfinite(ces):
            return "⚠️ **Data Error**: Invalid sensor readings."

        lines = [f"**CES: {ces:.1f}/100** | Load: {activity_load:.1f}/10\n"]

        # HR Efficiency (Enhanced)
        hr_diff = actual_hr - predicted_hr
        if abs(hr_diff) < 3:
            lines.append("✅ **HR**: Spot-on prediction")
        elif hr_diff > 0:
            lines.append(f"⚠️ **HR**: +{hr_diff:.0f} BPM (fatigue signal)")
        else:
            lines.append(f"✅ **HR**: {abs(hr_diff):.0f} BPM *under* (super-efficient!)")

        # HRV Status (Enhanced)
        hrv_ratio = actual_hrv / baseline_hrv
        if actual_hrv < HRV_CRITICAL_MS:
            lines.append(f"🔴 **HRV**: {actual_hrv:.0f}ms CRITICAL - REST NOW")
        elif actual_hrv < HRV_WARNING_MS:
            lines.append(f"⚠️ **HRV**: {actual_hrv:.0f}ms LOW")
        elif hrv_ratio > 1.2:
            lines.append(f"🟢 **HRV**: {actual_hrv:.0f}ms PEAK ({hrv_ratio*100:.0f}% baseline)")
        else:
            lines.append(f"✅ **HRV**: {actual_hrv:.0f}ms ({hrv_ratio*100:.0f}% baseline)")

        # 🆕 TREND INSIGHTS
        if show_trends and len(self.ces_history) >= 3:
            trend = self.get_trend_summary()
            lines.append(f"📈 **Trend**: {trend}")

        # 🆕 SMART RECOMMENDATIONS
        recommendation = self.generate_recommendation(ces, actual_hrv, activity_load)
        lines.append(f"\n💡 **Next**: {recommendation}")

        return "\n".join(lines)
    
    # 🆕 TREND ANALYSIS
    def get_trend_summary(self, days: int = 7) -> str:
        """3-day momentum analysis."""
        if len(self.ces_history) < 3:
            return "Insufficient data"
        
        recent = self.ces_history[-3:]
        trend_pct = ((recent[-1] - recent[0]) / recent[0]) * 100 if recent[0] > 0 else 0
        
        if trend_pct > 5:
            return f"🔥 UP {trend_pct:+.0f}% (gaining fitness!)"
        elif trend_pct < -5:
            return f"📉 DOWN {trend_pct:+.0f}% (recovery needed)"
        else:
            return f"➡️ Stable {trend_pct:+.0f}%"

    # 🆕 RECOVERY INTELLIGENCE
    def generate_recommendation(
        self, 
        ces: float, 
        hrv: float, 
        load: float
    ) -> str:
        """Context-aware training prescription."""
        if hrv < HRV_CRITICAL_MS:
            return "🚨 FULL REST 48h - Critical recovery"
        elif ces >= _CES_ELITE_THRESHOLD:
            return "🏆 ELITE - Push harder tomorrow"
        elif ces >= 80:
            return f"🟢 Maintain {load+1:.0f}/10 load"
        elif ces >= 60:
            return "🟡 Active recovery (Zone 1-2 only)"
        else:
            return "🔴 Deload 50% + sleep focus"

    def _calculate_recovery_debt(self) -> float:
        """Quantifies cumulative fatigue from recent sessions."""
        if len(self.ces_history) < 5:
            return 0.0
        
        # Use last 5 for recent, first 5 for baseline (distinct windows)
        recent_avg = np.mean(self.ces_history[-5:])
        
        # For baseline, use the earliest 5 readings, not overlapping with recent
        if len(self.ces_history) >= 10:
            baseline_avg = np.mean(self.ces_history[:5])
        elif len(self.ces_history) >= 5:
            # If we have 5-9 readings, use the middle portion as baseline
            baseline_avg = np.mean(self.ces_history[:len(self.ces_history)-5])
        else:
            baseline_avg = 70  # Default baseline
        
        return max(0, (baseline_avg - recent_avg) / baseline_avg)

    def _store_reading(self, ces: float, activity_load: float):
        """Append to history with timestamp."""
        self.ces_history.append(ces)
        self.timestamps.append(datetime.now())
        self.activity_history.append(activity_load)
        
        # Prune old data (>30 days)
        cutoff = datetime.now() - timedelta(days=30)
        valid_indices = [i for i, t in enumerate(self.timestamps) if t > cutoff]
        self.ces_history = [self.ces_history[i] for i in valid_indices]
        self.timestamps = [self.timestamps[i] for i in valid_indices]
        self.activity_history = [self.activity_history[i] for i in valid_indices]

# 🛡️ BACKWARD COMPATIBILITY
def cardiac_enhancement_score(
    actual_hr: float,
    predicted_hr: float,
    actual_hrv: float,
    baseline_hrv: float,
    activity_load: float,
    temperature: Optional[float] = None,
) -> float:
    """Legacy standalone function."""
    # Minimal profile for compatibility - capture values first
    _baseline_hrv = baseline_hrv
    _baseline_hr = predicted_hr * 0.6  # Estimate
    
    class DummyProfile:
        baseline_hrv = _baseline_hrv
        baseline_hr = _baseline_hr
    
    engine = AnalyticsEngine(DummyProfile())
    return engine.cardiac_enhancement_score(
        actual_hr, predicted_hr, actual_hrv, 
        baseline_hrv, activity_load, temperature
    )

def ces_explanation(
    ces: float, actual_hr: float, predicted_hr: float, 
    actual_hrv: float, baseline_hrv: float, activity_load: float = 0.0
) -> str:
    """Legacy standalone function."""
    _baseline_hrv = baseline_hrv
    
    class DummyProfile:
        baseline_hrv = _baseline_hrv
    
    engine = AnalyticsEngine(DummyProfile())
    return engine.ces_explanation(ces, actual_hr, predicted_hr, actual_hrv, baseline_hrv, activity_load)

def rolling_ces_trend(ces_history: List[float], window: int = 5) -> List[float]:
    """Enhanced rolling trend with nanmean."""
    if window < 1 or not ces_history:
        return list(ces_history)
    
    result = []
    for i in range(len(ces_history)):
        if i < window - 1:
            result.append(float("nan"))
        else:
            window_data = ces_history[i - window + 1:i + 1]
            avg = np.nanmean(window_data)
            result.append(round(float(avg), 2) if math.isfinite(avg) else float("nan"))
    return result