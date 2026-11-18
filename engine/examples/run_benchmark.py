#!/usr/bin/env python3
"""
SOMYTHRA Research — Benchmark Execution Script.

Evaluates SOMYTHRA rPPG (POS/CHROM) algorithms against synchronized ground truth reference data.
Usage:
    python3 run_benchmark.py --dataset ../data/benchmarks/DATASET_2 --subjects 5 --algorithm pos
"""

import argparse
import sys
from pathlib import Path

# Add engine root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from somythra.evaluation.benchmark import RPPGBenchmarkSuite


def main():
    parser = argparse.ArgumentParser(description="SOMYTHRA rPPG Ground Truth Benchmark Suite")
    parser.add_argument("--dataset", type=str, default="data/benchmarks/DATASET_2", help="Path to benchmark dataset folder")
    parser.add_argument("--algorithm", type=str, default="pos", choices=["pos", "chrom"], help="rPPG algorithm to benchmark")
    parser.add_argument("--subjects", type=int, default=5, help="Number of subjects to evaluate")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames per video to process (e.g. 300 for ~10s)")
    args = parser.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        print(f"Error: Dataset directory not found: {ds_path}")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f" SOMYTHRA rPPG GROUND TRUTH BENCHMARK")
    print(f" Algorithm: {args.algorithm.upper()} | Subjects: {args.subjects} | Max Frames: {args.max_frames}")
    print(f"=======================================================\n")

    suite = RPPGBenchmarkSuite(algorithm=args.algorithm, max_frames_per_video=args.max_frames)
    results, metrics = suite.evaluate_dataset_directory(ds_path, max_subjects=args.subjects)

    print(f"{'Subject':<12} | {'Estimated HR':<14} | {'Ref HR':<10} | {'Error (BPM)':<12} | {'SNR (dB)':<10} | {'Status'}")
    print("-" * 75)
    for r in results:
        est = f"{r.estimated_hr:.1f} BPM" if r.estimated_hr else "REJECTED"
        err = f"{r.error_bpm:.2f}" if r.error_bpm is not None else "N/A"
        print(f"{r.subject_id:<12} | {est:<14} | {r.ground_truth_hr:<10.1f} | {err:<12} | {r.snr_db:<10.2f} | {r.quality_status}")

    print("\n---------------- AGGREGATE EVALUATION METRICS ----------------")
    print(f"Total Subjects Evaluated:  {metrics.total_samples}")
    print(f"Mean Absolute Error (MAE): {metrics.mae:.2f} BPM")
    print(f"Root Mean Sq Error (RMSE): {metrics.rmse:.2f} BPM")
    print(f"Pearson Correlation (r):   {metrics.pearson_r:.4f}")
    print(f"Bland-Altman Mean Bias:    {metrics.mean_bias:.2f} BPM")
    print(f"95% LoA (Lower, Upper):    [{metrics.lower_loa:.2f}, {metrics.upper_loa:.2f}] BPM")
    print(f"Within 5 BPM Accuracy:     {metrics.percentage_within_5bpm:.1f}%")
    print(f"Within 10 BPM Accuracy:    {metrics.percentage_within_10bpm:.1f}%")
    print("===============================================================\n")


if __name__ == "__main__":
    main()
