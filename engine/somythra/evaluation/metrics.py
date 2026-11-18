"""
SOMYTHRA Evaluation — Statistical and Ground Truth Validation Metrics.

Implements clinical and research-grade benchmark metrics:
- Mean Absolute Error (MAE in BPM)
- Root Mean Squared Error (RMSE in BPM)
- Pearson's Correlation Coefficient (r)
- Bland-Altman Agreement Analysis (Mean bias, Upper LoA, Lower LoA, Outlier Rate)
"""

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class BlandAltmanResult:
    """Statistical summary of Bland-Altman agreement."""
    mean_bias: float
    sd_diff: float
    upper_loa: float  # Mean bias + 1.96 * SD
    lower_loa: float  # Mean bias - 1.96 * SD
    outlier_count: int
    outlier_percentage: float
    total_pairs: int


@dataclass
class EvaluationMetrics:
    """Comprehensive performance metrics comparing rPPG estimates against reference ground truth."""
    mae: float
    rmse: float
    pearson_r: float
    r_squared: float
    mean_bias: float
    sd_diff: float
    upper_loa: float
    lower_loa: float
    percentage_within_5bpm: float
    percentage_within_10bpm: float
    total_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mae_bpm": round(self.mae, 2),
            "rmse_bpm": round(self.rmse, 2),
            "pearson_r": round(self.pearson_r, 4),
            "r_squared": round(self.r_squared, 4),
            "mean_bias_bpm": round(self.mean_bias, 2),
            "sd_diff_bpm": round(self.sd_diff, 2),
            "upper_loa_bpm": round(self.upper_loa, 2),
            "lower_loa_bpm": round(self.lower_loa, 2),
            "within_5bpm_pct": round(self.percentage_within_5bpm, 1),
            "within_10bpm_pct": round(self.percentage_within_10bpm, 1),
            "sample_count": self.total_samples,
        }


def compute_bland_altman(
    estimated: Sequence[float],
    reference: Sequence[float],
) -> BlandAltmanResult:
    """
    Compute Bland-Altman 95% limits of agreement (LoA).
    Differences = Estimated - Reference.
    """
    est_arr = np.asarray(estimated, dtype=float)
    ref_arr = np.asarray(reference, dtype=float)

    if len(est_arr) != len(ref_arr):
        raise ValueError(f"Array length mismatch: {len(est_arr)} vs {len(ref_arr)}")

    valid_mask = ~np.isnan(est_arr) & ~np.isnan(ref_arr)
    est_clean = est_arr[valid_mask]
    ref_clean = ref_arr[valid_mask]

    n = len(est_clean)
    if n < 2:
        return BlandAltmanResult(0.0, 0.0, 0.0, 0.0, 0, 0.0, n)

    diffs = est_clean - ref_clean
    mean_bias = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))

    upper_loa = mean_bias + 1.96 * sd_diff
    lower_loa = mean_bias - 1.96 * sd_diff

    outliers = np.sum((diffs > upper_loa) | (diffs < lower_loa))
    outlier_pct = float(outliers / n * 100.0)

    return BlandAltmanResult(
        mean_bias=mean_bias,
        sd_diff=sd_diff,
        upper_loa=upper_loa,
        lower_loa=lower_loa,
        outlier_count=int(outliers),
        outlier_percentage=outlier_pct,
        total_pairs=n,
    )


def compute_evaluation_metrics(
    estimated: Sequence[float],
    reference: Sequence[float],
) -> EvaluationMetrics:
    """
    Compute full benchmarking evaluation metrics.
    """
    est_arr = np.asarray(estimated, dtype=float)
    ref_arr = np.asarray(reference, dtype=float)

    valid_mask = ~np.isnan(est_arr) & ~np.isnan(ref_arr)
    est_c = est_arr[valid_mask]
    ref_c = ref_arr[valid_mask]

    n = len(est_c)
    if n == 0:
        return EvaluationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    errors = np.abs(est_c - ref_c)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((est_c - ref_c) ** 2)))

    # Correlation
    if n > 1 and np.std(est_c) > 1e-6 and np.std(ref_c) > 1e-6:
        r = float(np.corrcoef(est_c, ref_c)[0, 1])
        r_squared = r ** 2
    else:
        r = 0.0
        r_squared = 0.0

    ba = compute_bland_altman(est_c, ref_c)
    within_5 = float(np.sum(errors <= 5.0) / n * 100.0)
    within_10 = float(np.sum(errors <= 10.0) / n * 100.0)

    return EvaluationMetrics(
        mae=mae,
        rmse=rmse,
        pearson_r=r,
        r_squared=r_squared,
        mean_bias=ba.mean_bias,
        sd_diff=ba.sd_diff,
        upper_loa=ba.upper_loa,
        lower_loa=ba.lower_loa,
        percentage_within_5bpm=within_5,
        percentage_within_10bpm=within_10,
        total_samples=n,
    )
