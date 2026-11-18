"""
SOMYTHRA Signals — Deterministic Synthetic Signal & Video Generator.

Generates ground-truth physiological RGB signals and synthetic face videos
with configurable heart rates, respiration drift, noise levels, and optical modulation.
Used for zero-hardware automated verification and mathematical validation.
"""

from typing import Tuple
import cv2
import numpy as np
from ..models.schemas import TemporalRGBSignal


def generate_synthetic_rgb_signal(
    duration_seconds: float = 10.0,
    sampling_rate_hz: float = 30.0,
    heart_rate_bpm: float = 72.0,
    respiration_rate_cpm: float = 15.0,
    snr_factor: float = 15.0,  # Higher = cleaner
    dc_r: float = 180.0,
    dc_g: float = 130.0,
    dc_b: float = 110.0,
    jitter_std_sec: float = 0.001,
    random_seed: int = 42,
) -> Tuple[TemporalRGBSignal, float]:
    """
    Generate a deterministic multichannel RGB signal with physiological optical pulsatility.
    Hemoglobin pulsatile absorption modulates the green channel predominantly.
    Returns:
        (TemporalRGBSignal, ground_truth_bpm)
    """
    rng = np.random.RandomState(random_seed)
    n_samples = int(duration_seconds * sampling_rate_hz)

    # Time vector with slight optional jitter
    dt = 1.0 / sampling_rate_hz
    ideal_ts = np.arange(n_samples) * dt
    if jitter_std_sec > 0:
        jitter = rng.normal(0, jitter_std_sec, n_samples)
        # Ensure monotonic timestamps
        ts = np.maximum.accumulate(ideal_ts + jitter)
    else:
        ts = ideal_ts

    pulse_freq_hz = heart_rate_bpm / 60.0
    resp_freq_hz = respiration_rate_cpm / 60.0

    # Cardiac pulsatile waveform: fundamental + subtle 2nd harmonic
    cardiac = np.sin(2.0 * np.pi * pulse_freq_hz * ts) + 0.25 * np.sin(4.0 * np.pi * pulse_freq_hz * ts)
    # Respiration baseline wander
    respiration = 0.5 * np.sin(2.0 * np.pi * resp_freq_hz * ts)

    # Physical hemoglobin absorption modulation vector across RGB channels:
    # Systolic blood influx attenuates reflectance predominantly in Green, followed by Blue and Red.
    mod_r = 0.010
    mod_g = 0.028
    mod_b = 0.018

    noise_amp = 1.0 / max(1.0, snr_factor)
    noise_r = rng.normal(0, noise_amp * mod_r, n_samples)
    noise_g = rng.normal(0, noise_amp * mod_g, n_samples)
    noise_b = rng.normal(0, noise_amp * mod_b, n_samples)

    r = dc_r * (1.0 - mod_r * cardiac + 0.004 * respiration + noise_r)
    g = dc_g * (1.0 - mod_g * cardiac + 0.006 * respiration + noise_g)
    b = dc_b * (1.0 - mod_b * cardiac + 0.005 * respiration + noise_b)

    sig = TemporalRGBSignal(
        timestamps=ts,
        r=r,
        g=g,
        b=b,
        roi_name="synthetic_face",
        sampling_rate_hz=sampling_rate_hz,
    )
    return sig, heart_rate_bpm


def generate_synthetic_face_frame(
    width: int,
    height: int,
    r_val: float,
    g_val: float,
    b_val: float,
    face_box_coords: Tuple[int, int, int, int] = (160, 100, 320, 320),
) -> np.ndarray:
    """
    Renders an image frame containing a stylized face with specified skin RGB values.
    OpenCV expects BGR format.
    """
    frame = np.full((height, width, 3), 40, dtype=np.uint8)  # Dark background
    fx, fy, fw, fh = face_box_coords

    # Draw skin region (oval / rounded rectangle)
    center = (fx + fw // 2, fy + fh // 2)
    axes = (fw // 2 - 10, fh // 2)

    skin_bgr = (int(np.clip(b_val, 0, 255)), int(np.clip(g_val, 0, 255)), int(np.clip(r_val, 0, 255)))
    cv2.ellipse(frame, center, axes, 0, 0, 360, skin_bgr, -1)

    # Draw facial features (eyes, nose, mouth) in dark tones
    eye_y = fy + int(0.38 * fh)
    eye_w = int(0.12 * fw)
    eye_h = int(0.06 * fh)
    left_eye_x = fx + int(0.28 * fw)
    right_eye_x = fx + int(0.72 * fw)

    cv2.ellipse(frame, (left_eye_x, eye_y), (eye_w, eye_h), 0, 0, 360, (20, 20, 20), -1)
    cv2.ellipse(frame, (right_eye_x, eye_y), (eye_w, eye_h), 0, 0, 360, (20, 20, 20), -1)

    # Mouth
    mouth_y = fy + int(0.78 * fh)
    cv2.ellipse(frame, (center[0], mouth_y), (int(0.20 * fw), int(0.05 * fh)), 0, 0, 360, (30, 30, 80), -1)

    return frame
