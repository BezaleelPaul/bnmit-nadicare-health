#!/usr/bin/env python3
"""
SOMYTHRA Examples — Executable Camera / Video rPPG Research Tool.

Usage:
  # Analyze prerecorded video:
  python run_rppg_analysis.py --video sample_face_video.mp4 --output analysis.png

  # Test synthetic signal without any files:
  python run_rppg_analysis.py --synthetic --bpm 75.0 --output synthetic_analysis.png

  # Stream from live webcam (if connected):
  python run_rppg_analysis.py --camera 0 --duration 10.0 --output camera_analysis.png
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np

# Ensure engine package is importable
ENGINE_ROOT = Path(__file__).resolve().parents[1]
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from somythra.pipeline import RPPGPipeline, PipelineConfig
from somythra.acquisition.camera import CameraCapture, CameraConfig
from somythra.signals.synthetic import generate_synthetic_rgb_signal
from somythra.models.schemas import QualityStatus, RPPGPipelineResult


def plot_results(result: RPPGPipelineResult, output_path: Optional_Path = None, show: bool = False):
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), constrained_layout=True)
    fig.patch.set_facecolor("#0F141A")

    for ax in axes:
        ax.set_facecolor("#161F28")
        ax.tick_params(colors="#94A3B8")
        ax.grid(True, color="#263545", linestyle="--", alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color("#2A3E54")

    # 1. Raw RGB traces
    raw = result.raw_signal
    axes[0].plot(raw.timestamps, raw.r, label="Red", color="#EF4444", linewidth=1.2)
    axes[0].plot(raw.timestamps, raw.g, label="Green", color="#10B981", linewidth=1.2)
    axes[0].plot(raw.timestamps, raw.b, label="Blue", color="#0EA5E9", linewidth=1.2)
    axes[0].set_title(
        f"1. Raw Multichannel Skin Color Signals ({raw.roi_name})",
        color="#F8FAFC",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    axes[0].set_ylabel("Intensity (DN)", color="#94A3B8")
    axes[0].legend(loc="upper right", facecolor="#1A2634", edgecolor="#26394D", labelcolor="#F8FAFC")

    # 2. Preprocessed & Detrended Signals
    prep = result.preprocessed_signal
    axes[1].plot(prep.timestamps, prep.r, color="#EF4444", alpha=0.6, label="Norm R")
    axes[1].plot(prep.timestamps, prep.g, color="#10B981", alpha=0.9, label="Norm G (Dominant Pulse)")
    axes[1].plot(prep.timestamps, prep.b, color="#0EA5E9", alpha=0.6, label="Norm B")
    axes[1].set_title(
        "2. Detrended & Temporally Normalized AC Signals",
        color="#F8FAFC",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    axes[1].set_ylabel("AC Modulation", color="#94A3B8")
    axes[1].legend(loc="upper right", facecolor="#1A2634", edgecolor="#26394D", labelcolor="#F8FAFC")

    # 3. Extracted Pulse Waveform
    pulse = result.pulse_waveform
    axes[2].plot(pulse.timestamps, pulse.signal, color="#F43F5E", linewidth=1.8, label=f"{pulse.algorithm_name} Pulse")
    axes[2].set_title(
        f"3. Extracted Blood Volume Pulse Waveform ({pulse.algorithm_name})",
        color="#F8FAFC",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    axes[2].set_ylabel("Pulse Amplitude", color="#94A3B8")
    axes[2].legend(loc="upper right", facecolor="#1A2634", edgecolor="#26394D", labelcolor="#F8FAFC")

    # 4. Power Spectral Density
    hr = result.heart_rate
    fs = pulse.sampling_rate_hz
    n = len(pulse.signal)
    fft_vals = np.abs(np.fft.rfft(pulse.signal - np.mean(pulse.signal), n=4096)) ** 2
    freqs = np.fft.rfftfreq(4096, d=1.0 / fs)
    mask = (freqs >= 0.5) & (freqs <= 4.0)

    axes[3].plot(freqs[mask] * 60.0, fft_vals[mask], color="#38BDF8", linewidth=1.6, label="Pulse PSD")

    # Highlight cardiac band [42, 210] BPM
    axes[3].axvspan(42, 210, color="#0EA5E9", alpha=0.08, label="Physiological Band (42-210 BPM)")

    if hr.is_reliable and hr.bpm is not None:
        axes[3].axvline(
            hr.bpm,
            color="#22C55E",
            linestyle="--",
            linewidth=2.0,
            label=f"Detected HR: {hr.bpm:.1f} BPM (SNR: {result.quality.snr_db:.1f} dB)",
        )
        status_text = f"HR: {hr.bpm:.1f} BPM (Quality: {result.quality.status.value})"
        status_color = "#22C55E"
    else:
        status_text = f"REJECTED: {hr.rejection_reason or 'Poor Quality'}"
        status_color = "#F59E0B"

    axes[3].set_title(
        f"4. Power Spectral Density — {status_text}",
        color=status_color,
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    axes[3].set_xlabel("Frequency (Beats Per Minute / BPM)", color="#94A3B8")
    axes[3].set_ylabel("Power", color="#94A3B8")
    axes[3].legend(loc="upper right", facecolor="#1A2634", edgecolor="#26394D", labelcolor="#F8FAFC")

    if output_path:
        out = Path(output_path)
        plt.savefig(out, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"Plot saved to: {out.resolve()}")

    if show:
        plt.show()

    plt.close(fig)


Optional_Path = Optional[Union[str, Path]]


def main():
    parser = argparse.ArgumentParser(description="SOMYTHRA R001 — Camera/Video rPPG Pulse Extraction Foundation")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--video", type=str, help="Path to video file (.mp4, .avi, .mov)")
    group.add_argument("--camera", type=int, default=None, help="Live camera device index (e.g. 0)")
    group.add_argument("--synthetic", action="store_true", help="Run on deterministic synthetic signal")

    parser.add_argument("--bpm", type=float, default=74.0, help="Heart rate for synthetic test (default: 74.0)")
    parser.add_argument("--algorithm", type=str, default="POS", choices=["POS", "CHROM"], help="rPPG algorithm")
    parser.add_argument("--duration", type=float, default=8.0, help="Camera capture duration in seconds")
    parser.add_argument("--output", type=str, default="rppg_analysis.png", help="Path to save output plot")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate offline plot file")
    args = parser.parse_args()

    pipeline = RPPGPipeline(PipelineConfig(algorithm=args.algorithm))

    print("=" * 70)
    print("  SOMYTHRA — R001: Camera Physiological Signal Foundation")
    print("  Algorithm:", args.algorithm)
    print("=" * 70)

    if args.video:
        video_file = Path(args.video)
        if not video_file.exists():
            print(f"Error: Video file '{video_file}' not found.")
            sys.exit(1)
        print(f"Processing video file: {video_file}...")
        result = pipeline.process_video_file(video_file)

    elif args.camera is not None:
        print(f"Opening camera index {args.camera} for {args.duration}s...")
        cam = CameraCapture(CameraConfig(device_index=args.camera))
        if not cam.open():
            print(f"Error: Could not open camera device {args.camera}.")
            print("Tip: Run with '--synthetic' or generate sample video with 'generate_sample_video.py'.")
            sys.exit(1)

        frames = []
        start = time.perf_counter()
        with cam:
            while (time.perf_counter() - start) < args.duration:
                ret, frame, ts = cam.read()
                if ret and frame is not None:
                    frames.append((frame, ts))

        print(f"Captured {len(frames)} frames. Average FPS: {cam.average_fps:.1f}")
        result = pipeline.process_frame_stream(iter(frames), measured_fps=cam.average_fps)

    else:
        # Default to synthetic physiological mode
        print(f"Generating deterministic synthetic signal ({args.duration}s @ 30 Hz, {args.bpm} BPM)...")
        sig, ground_truth = generate_synthetic_rgb_signal(
            duration_seconds=args.duration,
            sampling_rate_hz=30.0,
            heart_rate_bpm=args.bpm,
            snr_factor=30.0,
        )
        result = pipeline.process_rgb_signal(sig)
        print(f"Ground Truth Reference Heart Rate: {ground_truth:.1f} BPM")

    print("\n" + "-" * 70)
    print("  RESULTS & SIGNAL INTEGRITY SUMMARY")
    print("-" * 70)
    print(f"  Frames Processed:    {result.total_frames_processed} (Lost/Dropped: {result.dropped_frames})")
    print(f"  Measured Framerate:  {result.fps_measured:.2f} FPS")
    print(f"  Signal Duration:     {result.raw_signal.duration_seconds:.2f} seconds")
    print(f"  Quality Status:      {result.quality.status.value}")
    print(f"  Quality SNR:         {result.quality.snr_db:.2f} dB")
    print(f"  Confidence:          {result.quality.confidence * 100.0:.1f}%")
    print(f"  Timing Jitter Std:   {result.quality.jitter_std_ms:.2f} ms")

    hr = result.heart_rate
    if hr.is_reliable and hr.bpm is not None:
        print(f"  ESTIMATED HEART RATE: {hr.bpm:.1f} BPM  ✓ (Verified by Quality Gate)")
    else:
        print(f"  ESTIMATED HEART RATE: [REJECTED — UNRELIABLE SIGNAL]")
        print(f"  Reason:               {hr.rejection_reason}")

    print("\n  Disclaimer: " + hr.disclaimer)
    print("-" * 70)

    if not args.no_plot:
        plot_results(result, output_path=args.output)


if __name__ == "__main__":
    main()
