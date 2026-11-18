"""
Tests for SOMYTHRA Evaluation & Benchmarking Module.
"""

import pytest
import numpy as np
from somythra.evaluation.metrics import (
    compute_bland_altman,
    compute_evaluation_metrics,
    BlandAltmanResult,
    EvaluationMetrics,
)


def test_bland_altman_perfect_agreement():
    ref = [70.0, 72.0, 75.0, 80.0, 68.0]
    est = [70.0, 72.0, 75.0, 80.0, 68.0]

    ba = compute_bland_altman(est, ref)
    assert isinstance(ba, BlandAltmanResult)
    assert ba.mean_bias == pytest.approx(0.0, abs=1e-5)
    assert ba.sd_diff == pytest.approx(0.0, abs=1e-5)
    assert ba.outlier_count == 0
    assert ba.total_pairs == 5


def test_bland_altman_constant_bias():
    ref = np.array([60.0, 70.0, 80.0, 90.0])
    est = ref + 3.0  # Constant +3 BPM bias

    ba = compute_bland_altman(est, ref)
    assert ba.mean_bias == pytest.approx(3.0, abs=1e-5)
    assert ba.sd_diff == pytest.approx(0.0, abs=1e-5)
    assert ba.upper_loa == pytest.approx(3.0, abs=1e-5)
    assert ba.lower_loa == pytest.approx(3.0, abs=1e-5)


def test_compute_evaluation_metrics():
    ref = [60.0, 70.0, 80.0, 90.0, 100.0]
    est = [61.0, 69.0, 82.0, 88.0, 101.0]

    metrics = compute_evaluation_metrics(est, ref)
    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.mae == pytest.approx(1.4, abs=0.1)
    assert metrics.rmse < 2.0
    assert metrics.pearson_r > 0.99
    assert metrics.percentage_within_5bpm == 100.0
    assert metrics.total_samples == 5

    dict_repr = metrics.to_dict()
    assert "mae_bpm" in dict_repr
    assert "pearson_r" in dict_repr
