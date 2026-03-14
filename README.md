# ❤️ NadiCare — Cardio-Fitness Digital Twin

> *Predict. Monitor. Optimise. Your heart's virtual replica.*

## 🚀 Quick Start

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv

# 2. Activate it
# Windows (PowerShell): .\.venv\Scripts\Activate.ps1
# Windows (CMD):        .\.venv\Scripts\activate.bat

# 3. Install dependencies
python -m pip install -r requirements.txt

# 4. Generate demo data
python data_gen.py

# 5. Launch the dashboard
streamlit run app.py
```

## 📁 Project Structure

```
nadicare/
├── app.py                  # Streamlit dashboard (main entry point)
├── data_gen.py             # 24-hour demo data generator
├── requirements.txt        # Python dependencies
├── demo_data.csv           # Generated after running data_gen.py
└── src/
    ├── __init__.py
    ├── models.py           # Pydantic schemas (HeartData, UserProfile, TwinState)
    ├── twin_engine.py      # DigitalTwin class — exponential decay model
    ├── analytics.py        # Cardiac Enhancement Score (CES) calculation
    └── safety_monitor.py   # Boundary alert system
```

## 🧬 Core Algorithm

```
Recovery:  HR(t) = HR_baseline + (HR_peak - HR_baseline) × e^(−λt)

CES:       100 × (HRV_actual / HRV_baseline)
               × (1 − |HR_actual − HR_predicted| / HR_predicted)
               × (1 / (1 + load × 0.05))
```

## 🔴 Safety Thresholds

| Condition              | Threshold        | Alert Level |
|------------------------|-----------------|-------------|
| HR > 90% Max HR        | (208 − 0.7×age)×0.90 | CRITICAL |
| HR > 75% Max HR        | (208 − 0.7×age)×0.75 | WARNING  |
| HRV < 25 ms (RMSSD)    | absolute         | CRITICAL    |
| HRV < 35 ms (RMSSD)    | absolute         | WARNING     |

Notes:
- Max HR uses the **Tanaka** age-predicted formula (\(208 - 0.7 \times age\)) as implemented in `src/models.py`.
- HRV here is treated as **RMSSD (ms)** (common wearable-style HRV output).

## 👥 Team

- **Lead Developer:** Bezaleel Paul N.
- **University:** CMR University, Bangalore
- **Hackathon:** NadiCare Submission
