#!/usr/bin/env python3
"""
SOMYTHRA Examples — Generate Sample Facial Video for Offline Testing.

Renders a synthetic facial video with an anatomically realistic face and
subtle plethysmographic color modulation corresponding to a known heart rate.
Enables zero-hardware, zero-camera end-to-end verification.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from somythra.signals.synthetic import generate_synthetic_rgb_signal, generate_synthetic_face_frame


def main():
    parser = argparse.ArgumentParser(description="Generate sample physiological face video.")
    parser.add_argument("--output", type=str, default="sample_face_video.mp4", help="Output MP4 file path")
    parser.add_argument("--duration", type=float, default=8.0, help="Duration in seconds (default: 8.0s)")
    parser.add_argument("--fps", type=float, default=30.0, help="Framerate (default: 30.0)")
    parser.add_argument("--bpm", type=float, default=74.0, help="Target heart rate in BPM (default: 74.0)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    args = parser.parse_args()

    out_path = Path(args.output)
    n_frames = int(args.duration * args.fps)
    print(f"Generating {args.duration}s video ({n_frames} frames @ {args.fps} FPS) with {args.bpm} BPM modulation...")

    sig, _ = generate_synthetic_rgb_signal(
        duration_seconds=args.duration,
        sampling_rate_hz=args.fps,
        heart_rate_bpm=args.bpm,
        jitter_std_sec=0.0,
        snr_factor=30.0,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    # Face centered in 640x480 frame
    face_coords = (
        int(args.width * 0.22),
        int(args.height * 0.12),
        int(args.width * 0.56),
        int(args.height * 0.72),
    )

    for i in range(n_frames):
        frame = generate_synthetic_face_frame(
            width=args.width,
            height=args.height,
            r_val=sig.r[i],
            g_val=sig.g[i],
            b_val=sig.b[i],
            face_box_coords=face_coords,
        )
        writer.write(frame)

    writer.release()
    print(f"Sample video successfully written to: {out_path.resolve()} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
