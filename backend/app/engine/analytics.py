import math
from app.models.schemas import CESRequest, CESResponse

class SomythraAnalytics:
    """
    Physiological efficiency and strain analytics.
    Calculates the Cardiac Enhancement Score (CES) comparing observed physiology
    against digital twin expectation and autonomic reserve.
    """

    @staticmethod
    def calculate_ces(req: CESRequest) -> CESResponse:
        """
        Calculate bounded CES in [0, 100].
        Components:
        1. Autonomic component: (actual_hrv / baseline_hrv)
        2. Precision/Deviation component: 1 - |actual_hr - predicted_hr| / predicted_hr
        3. Load penalty factor: 1 / (1 + load * 0.05)
        """
        # 1. Autonomic balance ratio
        hrv_ratio = req.actual_hrv / max(1.0, req.baseline_hrv)
        # Cap high ratio to prevent runaway
        autonomic_factor = min(1.5, max(0.1, hrv_ratio))

        # 2. Cardiac strain deviation
        hr_err = abs(req.actual_hr - req.predicted_hr)
        hr_precision = max(0.0, 1.0 - (hr_err / max(30.0, req.predicted_hr)))
        strain_pct = round(min(100.0, (hr_err / max(30.0, req.predicted_hr)) * 100.0), 1)

        # 3. Load modulation
        load_mod = 1.0 / (1.0 + max(0.0, req.activity_load) * 0.05)
        load_penalty_pct = round((1.0 - load_mod) * 100.0, 1)

        raw_score = 100.0 * autonomic_factor * hr_precision * load_mod
        score = round(min(100.0, max(0.0, raw_score)), 1)

        if score >= 80:
            rating = "Optimal Interconnected State"
            summary = "High vagal tone, cardiac output aligns with physiological baseline expectation."
        elif score >= 60:
            rating = "Balanced Adaptation"
            summary = "Cardiovascular system adapting adequately to current autonomic demand."
        elif score >= 40:
            rating = "Mild Strain / Elevated Load"
            summary = "Slight divergence between expected twin model and observed heart rate."
        else:
            rating = "Significant Physiological Strain"
            summary = "Autonomic tone depressed relative to heart rate divergence; consider rest."

        return CESResponse(
            score=score,
            rating=rating,
            cardiac_strain_pct=strain_pct,
            autonomic_balance_pct=round(autonomic_factor * 100.0, 1),
            load_penalty=load_penalty_pct,
            summary=summary,
        )
