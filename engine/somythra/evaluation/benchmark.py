"""
SOMYTHRA Evaluation — rPPG Ground Truth Benchmark Suite.

Executes repeatable evaluation against synchronized reference sensors (e.g. ECG/contact PPG).
Computes windowed error progression and aggregate Bland-Altman statistics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np

from ..pipeline import RPPGPipeline, PipelineConfig
from ..models.schemas import RPPGPipelineResult, QualityStatus
from .metrics import compute_evaluation_metrics, EvaluationMetrics


@dataclass
class BenchmarkSubjectResult:
    """Individual subject benchmark evaluation result."""
    subject_id: str
    algorithm: str
    estimated_hr: Optional[float]
    ground_truth_hr: float
    error_bpm: Optional[float]
    quality_status: str
    snr_db: float
    confidence: float
    duration_seconds: float
    is_valid: bool


class RPPGBenchmarkSuite:
    """
    Automated benchmark harness for camera physiological pulse extraction.
    Supports directory scanning, ground-truth pairing, and performance summarization.
    """

    def __init__(self, algorithm: str = "pos", max_frames_per_video: Optional[int] = 300):
        self.algorithm = algorithm
        self.max_frames = max_frames_per_video
        self.pipeline = RPPGPipeline(config=PipelineConfig(algorithm=algorithm))

    def evaluate_subject(
        self,
        video_path: Union[str, Path],
        ground_truth_path: Union[str, Path],
        subject_id: str = "subject_1",
    ) -> BenchmarkSubjectResult:
        """
        Evaluate a single video against its ground truth reference file.
        Ground truth text file format:
          Line 0: Reference PPG waveform
          Line 1: Instantaneous Heart Rate in BPM per frame
          Line 2: Timestamps in seconds
        """
        v_path = Path(video_path)
        gt_path = Path(ground_truth_path)

        if not v_path.exists():
            raise FileNotFoundError(f"Video file missing: {v_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file missing: {gt_path}")

        # Parse Ground Truth
        with open(gt_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        if len(lines) >= 2:
            gt_bpm_tokens = [float(x) for x in lines[1].split()]
            if self.max_frames:
                gt_bpm_slice = gt_bpm_tokens[:self.max_frames]
            else:
                gt_bpm_slice = gt_bpm_tokens
            ref_hr = float(np.mean(gt_bpm_slice))
        else:
            raise ValueError(f"Ground truth file malformed: {gt_path}")

        # Run Pipeline
        result: RPPGPipelineResult = self.pipeline.process_video_file(
            v_path, max_frames=self.max_frames
        )

        est_hr = result.heart_rate.bpm if result.heart_rate else None
        err = abs(est_hr - ref_hr) if est_hr is not None else None
        is_val = (
            result.quality.status == QualityStatus.ACCEPTABLE
            and est_hr is not None
        )

        return BenchmarkSubjectResult(
            subject_id=subject_id,
            algorithm=self.algorithm.upper(),
            estimated_hr=round(est_hr, 1) if est_hr is not None else None,
            ground_truth_hr=round(ref_hr, 1),
            error_bpm=round(err, 2) if err is not None else None,
            quality_status=result.quality.status.value,
            snr_db=round(result.quality.snr_db, 2),
            confidence=round(result.heart_rate.confidence, 3) if result.heart_rate else 0.0,
            duration_seconds=round(result.raw_signal.duration_seconds, 2),
            is_valid=is_val,
        )

    def evaluate_dataset_directory(
        self, dataset_dir: Union[str, Path], max_subjects: int = 10
    ) -> Tuple[List[BenchmarkSubjectResult], EvaluationMetrics]:
        """
        Scan directory containing subject subfolders with vid_small.mp4 and ground_truth.txt.
        Returns list of results and aggregate metrics.
        """
        root = Path(dataset_dir)
        subject_dirs = sorted([d for d in root.iterdir() if d.is_dir() and (d / "ground_truth.txt").exists()])

        results: List[BenchmarkSubjectResult] = []
        estimates: List[float] = []
        references: List[float] = []

        for sdir in subject_dirs[:max_subjects]:
            v_file = sdir / "vid_small.mp4"
            gt_file = sdir / "ground_truth.txt"
            if not v_file.exists():
                continue

            try:
                sub_res = self.evaluate_subject(v_file, gt_file, subject_id=sdir.name)
                results.append(sub_res)
                if sub_res.estimated_hr is not None:
                    estimates.append(sub_res.estimated_hr)
                    references.append(sub_res.ground_truth_hr)
            except Exception:
                continue

        metrics = compute_evaluation_metrics(estimates, references)
        return results, metrics
