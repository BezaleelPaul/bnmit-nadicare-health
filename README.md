# NadiCare — Cardio-Fitness Digital Twin Prototype

NadiCare is an exploratory Python and Streamlit project for visualising heart-rate and heart-rate-variability signals, simulating a simplified recovery response, and comparing modelled trends.

> **Research and education prototype only.** NadiCare is not a medical device, diagnostic system, emergency monitor, or substitute for advice from a qualified healthcare professional. Its models, thresholds, predictions, and visualisations have not been established as clinically safe or validated for individual medical decisions. Do not use the application to decide whether exercise is safe or to delay seeking medical care.

## What is included

- Streamlit dashboard for interactive exploration.
- Synthetic/demo data generation.
- A simplified exponential-decay recovery model.
- Analytics and visualisation utilities.
- A rule-based monitoring module and an optional stress-prediction model path.

The output depends on the input data and assumptions encoded in the project. A simulated curve is not a measurement of a person's actual physiology.

## Technology

- Python
- Streamlit
- NumPy and pandas
- Plotly
- SciPy
- scikit-learn
- Pydantic

Dependencies are listed in [requirements.txt](./requirements.txt).

## Run locally

```bash
git clone https://github.com/BezaleelPaul/bnmit-nadicare-health.git
cd bnmit-nadicare-health

python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

Install dependencies and start the dashboard:

```bash
python -m pip install -r requirements.txt
python data_gen.py
streamlit run app.py
```

Streamlit prints the local address to open in your browser.

## Project structure

```text
.
├── app.py                  # Streamlit dashboard
├── data_gen.py             # Demo data generator
├── requirements.txt        # Python dependencies
└── src/
    ├── models.py           # Input models and configured constants
    ├── twin_engine.py      # Simplified recovery simulation
    ├── analytics.py        # Derived analytics
    ├── safety_monitor.py   # Experimental rule-based monitoring
    └── stress_predictor.py # Stress prediction utilities
```

## Model overview

The recovery simulation uses a first-order exponential response:

```text
HR(t) = HR_baseline + (HR_peak - HR_baseline) × exp(-λt)
```

The implementation also models an HRV recovery curve. The decay parameter is selected from the configured profile unless explicitly overridden. These are simplifying assumptions for software experimentation, not a complete physiological model.

## Screenshots

The repository includes example dashboard captures under `screenshot/`. They illustrate the interface and should not be interpreted as clinical validation or evidence of real-time medical monitoring.

## Limitations and responsible use

- Demo data may be synthetic and should be labelled as such in any analysis.
- Model outputs are estimates produced by simplified assumptions.
- The configured alert thresholds are experimental software rules, not universally applicable clinical thresholds.
- Sensor accuracy, missing data, individual variation, and external validation are not established by a successful application run.
- Do not use this project for diagnosis, treatment, emergency decisions, or exercise clearance.

## Team

- **Lead developer:** Bezaleel Paul N.
- **Researchers:** Madhusudhana S and Aditya S
- **Institution:** CMR University, Bengaluru
- **Context:** NadiCare project submission

## Licence

See the repository for licensing and permitted reuse.
