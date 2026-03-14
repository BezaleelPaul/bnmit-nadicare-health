# data_gen.py
# Demo data generator for NadiCare Digital Twin
# Simulates a realistic 24-hour cardiac cycle: Sleep → Morning → Workout → Recovery → Evening

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models import UserProfile
<<<<<<< HEAD
from src.safety_monitor import CardiacMonitor
=======
from src.safety_monitor import check_safety_boundaries
>>>>>>> 05adc39bba754c5158ab0f4dada08bb46ab65556

np.random.seed(42)


def generate_24h_demo(output_path: str = "demo_data.csv"):
    """
    Generate a 24-hour synthetic cardiac dataset and save as CSV.

    Phases:
        00:00 - 06:00  Deep Sleep (low HR, high HRV, zero load)
        06:00 - 07:00  Morning Wake-Up (gradual HR rise)
        07:00 - 08:00  Warm-Up Workout (moderate effort)
        08:00 - 08:30  Peak Sprint (high HR, low HRV)
        08:30 - 10:00  Recovery (exponential HR decay)
        10:00 - 17:00  Normal Day (mild fluctuations)
        17:00 - 18:00  Evening Walk (light effort)
        18:00 - 00:00  Wind-Down & Sleep Prep
    """

    # 1-minute intervals over 24 hours = 1440 rows
    start = datetime(2024, 6, 1, 0, 0, 0)
    timestamps = [start + timedelta(minutes=i) for i in range(1440)]

    # Demo profile used for safety boundary labelling
    demo_profile = UserProfile(
        name="Demo User",
        age=30,
        weight_kg=70,
        baseline_hr=60,
        baseline_hrv=70,
    )

<<<<<<< HEAD
    # Create monitor once for efficiency
    monitor = CardiacMonitor(demo_profile)

=======
>>>>>>> 05adc39bba754c5158ab0f4dada08bb46ab65556
    hrs, hrvs, loads, labels = [], [], [], []
    strategies, safety_levels = [], []

    for i, ts in enumerate(timestamps):
        hour = ts.hour + ts.minute / 60.0

        # --- Deep Sleep (00:00–06:00) ---
        if 0 <= hour < 6:
            hr = 52 + np.random.normal(0, 1.5)
            hrv = 72 + np.random.normal(0, 3)
            load = 0.0
            label = "Sleep"

        # --- Wake-Up (06:00–07:00) ---
        elif 6 <= hour < 7:
            t = (hour - 6) * 60  # minutes into phase
            hr = 52 + t * 0.25 + np.random.normal(0, 2)
            hrv = 72 - t * 0.3 + np.random.normal(0, 3)
            load = 0.5
            label = "Wake-Up"

        # --- Warm-Up (07:00–08:00) ---
        elif 7 <= hour < 8:
            t = (hour - 7) * 60
            hr = 67 + t * 0.4 + np.random.normal(0, 2.5)
            hrv = 63 - t * 0.25 + np.random.normal(0, 4)
            load = 3 + t * 0.04
            label = "Warm-Up"

        # --- Sprint (08:00–08:30) ---
        elif 8 <= hour < 8.5:
            t = (hour - 8) * 60
            hr = 91 + t * 1.8 + np.random.normal(0, 3)
            hrv = 38 - t * 0.6 + np.random.normal(0, 3)
            load = 8 + t * 0.06
            label = "Sprint"

        # --- Recovery (08:30–10:00) ---
        elif 8.5 <= hour < 10:
            t = (hour - 8.5) * 60  # 0–90 minutes
            # Exponential decay back to baseline
<<<<<<< HEAD
            baseline_hr = demo_profile.baseline_hr
            baseline_hrv = demo_profile.baseline_hrv
            hr = baseline_hr + (145 - baseline_hr) * np.exp(-0.03 * t) + np.random.normal(0, 2)
            hrv = baseline_hrv - (baseline_hrv - 22) * np.exp(-0.025 * t) + np.random.normal(0, 3)
=======
            hr = 65 + (145 - 65) * np.exp(-0.03 * t) + np.random.normal(0, 2)
            hrv = 65 - (65 - 22) * np.exp(-0.025 * t) + np.random.normal(0, 3)
>>>>>>> 05adc39bba754c5158ab0f4dada08bb46ab65556
            load = max(0, 5 - t * 0.06)
            label = "Recovery"

        # --- Normal Day (10:00–17:00) ---
        elif 10 <= hour < 17:
            hr = 68 + np.random.normal(0, 3)
            hrv = 58 + np.random.normal(0, 5)
            load = 1.0 + np.random.normal(0, 0.3)
            label = "Normal"

        # --- Evening Walk (17:00–18:00) ---
        elif 17 <= hour < 18:
            hr = 82 + np.random.normal(0, 3)
            hrv = 48 + np.random.normal(0, 4)
            load = 3.5
            label = "Evening Walk"

        # --- Wind-Down (18:00–00:00) ---
        elif 18 <= hour < 24:
            t = (hour - 18) * 60
            hr = 82 - t * 0.05 + np.random.normal(0, 2)
            hrv = 48 + t * 0.06 + np.random.normal(0, 3)
            load = max(0, 2 - t * 0.01)
            label = "Wind-Down"
        else:
            # Fallback (should not be hit for a 24h window, but keeps model safe)
            hr = 65 + np.random.normal(0, 3)
            hrv = 60 + np.random.normal(0, 5)
            load = 1.0
            label = "Normal"

        hr_val = round(float(np.clip(hr, 40, 200)), 1)
        hrv_val = round(float(np.clip(hrv, 5, 120)), 1)
        load_val = round(float(np.clip(load, 0, 10)), 2)

        # Map phase → high-level strategy category used by the dashboard
        if label in {"Warm-Up", "Sprint", "Recovery"}:
            strategy = "Interval Training Block"
        elif label in {"Evening Walk"}:
            strategy = "Recovery Optimisation"
        elif label in {"Sleep", "Wind-Down"}:
            strategy = "Deep Recovery / Sleep"
        elif label in {"Wake-Up"}:
            strategy = "Transition"
        else:
            strategy = "Baseline Day"

        # Safety boundary level for this minute given the demo profile
<<<<<<< HEAD
        alert = monitor.check_all(hr_val, hrv_val)
=======
        alert = check_safety_boundaries(hr=hr_val, hrv=hrv_val, profile=demo_profile)
>>>>>>> 05adc39bba754c5158ab0f4dada08bb46ab65556

        hrs.append(hr_val)
        hrvs.append(hrv_val)
        loads.append(load_val)
        labels.append(label)
        strategies.append(strategy)
        safety_levels.append(alert.level.value)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "hr": hrs,
        "hrv": hrvs,
        "activity_load": loads,
        "label": labels,
        "strategy": strategies,
        "safety_level": safety_levels,
    })

    df.to_csv(output_path, index=False)
    print(f"✅ Demo data saved to '{output_path}' ({len(df)} rows)")
    return df


if __name__ == "__main__":
    df = generate_24h_demo()
    print(df.head(10))
    print("\nPhase distribution:")
    print(df["label"].value_counts())
