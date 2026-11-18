"""
SOMYTHRA Vision — Live Rolling FIFO Buffer Streamer.

Enables real-time rPPG streaming for consumer and mobile applications.
Continuously ingests incoming frames, compensates for ambient auto-exposure /
white-balance fluctuations, emits a live 30 FPS pulse sample for wave charts,
and refreshes full spectral BPM & optical features at a 1-second cadence.
"""

import time
import logging
from collections import deque
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np

from .models.schemas import (
    BoundingBox,
    QualityStatus,
    QualityAssessment,
    HeartRateEstimate,
    LiveStreamPacket,
    TemporalRGBSignal,
)
from .vision.face_detection import HaarCascadeFaceDetector, BaseFaceDetector
from .vision.skin_roi import SkinROIExtractor
from .signals.preprocessing import SignalPreprocessor, PreprocessingConfig
from .signals.filtering import BandpassFilter, FilterConfig
from .algorithms.pos import POSAlgorithm
from .algorithms.chrom import CHROMAlgorithm
from .quality.assessment import QualityConfig
from .physiology.heart_rate import HeartRateEstimator, HeartRateConfig

logger = logging.getLogger(__name__)


class LiveRPPGStreamer:
    """
    Stateful real-time rPPG stream engine with rolling FIFO buffer
    and common-mode ambient auto-exposure (AE/AWB) compensation.
    """

    def __init__(
        self,
        window_duration_seconds: float = 8.0,
        cadence_interval_seconds: float = 1.0,
        min_duration_seconds: float = 3.5,
        target_fps: float = 30.0,
        algorithm_name: str = "pos",
        compensate_ambient: bool = True,
        detector: Optional[BaseFaceDetector] = None,
    ):
        self.window_duration = window_duration_seconds
        self.cadence_interval = cadence_interval_seconds
        self.min_duration = min_duration_seconds
        self.target_fps = target_fps
        self.compensate_ambient = compensate_ambient

        self.detector = detector or HaarCascadeFaceDetector()
        self.roi_extractor = SkinROIExtractor()
        self.preprocessor = SignalPreprocessor(
            PreprocessingConfig(target_sampling_rate_hz=target_fps)
        )
        self.bandpass_filter = BandpassFilter(
            FilterConfig(lowcut_hz=0.7, highcut_hz=3.5, order=2)
        )
        self.algorithm = CHROMAlgorithm() if algorithm_name.lower() == "chrom" else POSAlgorithm()
        self.hr_estimator = HeartRateEstimator(
            HeartRateConfig(min_bpm=42.0, max_bpm=210.0)
        )

        # Max buffer size with 50% margin
        max_buffer_len = int(window_duration_seconds * target_fps * 1.5)
        self._buffer: deque = deque(maxlen=max_buffer_len)

        # Ambient baseline reference
        self._ambient_ref: Optional[np.ndarray] = None
        self._pulse_history: deque = deque(maxlen=int(target_fps * 2))

        # Rolling state
        self._frame_index: int = 0
        self._start_time: float = 0.0
        self._last_cadence_time: float = 0.0

        # Latest cadence estimates
        self._latest_bpm: Optional[float] = None
        self._latest_confidence: float = 0.0
        self._latest_snr_db: float = 0.0
        self._latest_status: QualityStatus = QualityStatus.UNPROCESSED
        self._latest_reason: Optional[str] = "Accumulating initial physiological buffer..."
        self._latest_optical_features: Optional[Dict[str, float]] = None

    def reset(self) -> None:
        """Clear streaming buffer and reset internal filter state."""
        self._buffer.clear()
        self._pulse_history.clear()
        self._frame_index = 0
        self._start_time = 0.0
        self._last_cadence_time = 0.0
        self._ambient_ref = None
        self._latest_bpm = None
        self._latest_confidence = 0.0
        self._latest_snr_db = 0.0
        self._latest_status = QualityStatus.UNPROCESSED
        self._latest_reason = "Accumulating initial physiological buffer..."
        self._latest_optical_features = None

    def ingest_frame(
        self, frame_bgr: np.ndarray, timestamp: Optional[float] = None
    ) -> LiveStreamPacket:
        """
        Process a single live video frame.
        Returns a LiveStreamPacket with instantaneous pulse sample for live rendering
        and refreshed BPM / quality metrics when a cadence step triggers.
        """
        now = time.perf_counter() if timestamp is None else timestamp
        if self._start_time <= 0.0:
            self._start_time = now
        rel_time = now - self._start_time

        self._frame_index += 1

        # 1. Face detection
        face_box = self.detector.detect_face(frame_bgr)
        face_detected = face_box is not None and face_box.is_valid()

        # 2. Extract facial skin ROIs
        rois = self.roi_extractor.extract_rois(frame_bgr, face_box) if face_detected else None

        valid_skin = False
        r_skin, g_skin, b_skin = 0.0, 0.0, 0.0

        if rois is not None:
            # Extract forehead
            fh_ok, fh_r, fh_g, fh_b, fh_cnt = self.roi_extractor.extract_skin_rgb(
                frame_bgr, rois.forehead
            )
            # Extract cheeks
            lc_ok, lc_r, lc_g, lc_b, lc_cnt = self.roi_extractor.extract_skin_rgb(
                frame_bgr, rois.left_cheek
            )
            rc_ok, rc_r, rc_g, rc_b, rc_cnt = self.roi_extractor.extract_skin_rgb(
                frame_bgr, rois.right_cheek
            )

            total_skin_px = fh_cnt + lc_cnt + rc_cnt
            if total_skin_px > 0:
                valid_skin = True
                r_skin = (fh_r * fh_cnt + lc_r * lc_cnt + rc_r * rc_cnt) / total_skin_px
                g_skin = (fh_g * fh_cnt + lc_g * lc_cnt + rc_g * rc_cnt) / total_skin_px
                b_skin = (fh_b * fh_cnt + lc_b * lc_cnt + rc_b * rc_cnt) / total_skin_px

        # 3. Ambient Auto-Exposure / White Balance Compensation
        r_amb, g_amb, b_amb = 128.0, 128.0, 128.0
        if self.compensate_ambient and valid_skin:
            amb_ok, amb_r, amb_g, amb_b = self.roi_extractor.extract_ambient_rgb(frame_bgr, face_box)
            if amb_ok:
                r_amb, g_amb, b_amb = amb_r, amb_g, amb_b
                if self._ambient_ref is None:
                    self._ambient_ref = np.array([amb_r, amb_g, amb_b], dtype=np.float64)
                else:
                    # Slow exponential tracker for reference ambient
                    self._ambient_ref = 0.98 * self._ambient_ref + 0.02 * np.array([amb_r, amb_g, amb_b])

                # Common-mode gain cancellation
                gain_r = self._ambient_ref[0] / max(1.0, amb_r)
                gain_g = self._ambient_ref[1] / max(1.0, amb_g)
                gain_b = self._ambient_ref[2] / max(1.0, amb_b)

                # Clamped gain prevents division explosions during extreme shadows
                r_skin *= float(np.clip(gain_r, 0.5, 2.0))
                g_skin *= float(np.clip(gain_g, 0.5, 2.0))
                b_skin *= float(np.clip(gain_b, 0.5, 2.0))

        # 4. Append to rolling FIFO buffer
        if valid_skin:
            self._buffer.append({
                "t": rel_time,
                "r": r_skin,
                "g": g_skin,
                "b": b_skin,
                "valid": True,
            })
        else:
            # Mark invalid or dropped frame
            if self._buffer:
                last = self._buffer[-1]
                self._buffer.append({
                    "t": rel_time,
                    "r": last["r"],
                    "g": last["g"],
                    "b": last["b"],
                    "valid": False,
                })

        # Trim buffer to window duration
        while len(self._buffer) > 2 and (self._buffer[-1]["t"] - self._buffer[0]["t"]) > self.window_duration:
            self._buffer.popleft()

        # 5. Compute real-time instantaneous pulse sample for 30 Hz live chart
        pulse_sample = 0.0
        if len(self._buffer) >= 3:
            # Fast rolling green AC signal
            recent_g = [entry["g"] for entry in self._buffer]
            mean_g = float(np.mean(recent_g[-30:])) if len(recent_g) >= 30 else float(np.mean(recent_g))
            raw_sample = float(recent_g[-1] - mean_g)
            self._pulse_history.append(raw_sample)

            # 2-point moving average smoothing
            if len(self._pulse_history) >= 2:
                pulse_sample = float(0.5 * (self._pulse_history[-1] + self._pulse_history[-2]))
            else:
                pulse_sample = raw_sample

        # 6. Check cadence trigger (e.g. every 1.0 second)
        is_cadence_update = False
        duration_in_buf = self._buffer[-1]["t"] - self._buffer[0]["t"] if len(self._buffer) >= 2 else 0.0

        if (rel_time - self._last_cadence_time) >= self.cadence_interval:
            self._last_cadence_time = rel_time
            is_cadence_update = True

            if duration_in_buf >= self.min_duration and len(self._buffer) >= int(self.min_duration * 15):
                self._run_cadence_update(duration_in_buf)
            else:
                self._latest_status = QualityStatus.INSUFFICIENT_DURATION
                self._latest_reason = f"Buffering: {duration_in_buf:.1f}s / {self.min_duration:.1f}s needed"
                self._latest_bpm = None

        return LiveStreamPacket(
            frame_index=self._frame_index,
            timestamp=rel_time,
            pulse_sample=pulse_sample,
            bpm=self._latest_bpm,
            confidence=self._latest_confidence,
            snr_db=self._latest_snr_db,
            quality_status=self._latest_status,
            is_cadence_update=is_cadence_update,
            rejection_reason=self._latest_reason,
            optical_features=self._latest_optical_features,
            face_detected=face_detected,
            face_box=face_box,
        )

    def _run_cadence_update(self, duration_in_buf: float) -> None:
        """Run full periodogram, SNR, and optical feature calculation on the buffered window."""
        timestamps = np.array([e["t"] for e in self._buffer], dtype=np.float64)
        r_arr = np.array([e["r"] for e in self._buffer], dtype=np.float64)
        g_arr = np.array([e["g"] for e in self._buffer], dtype=np.float64)
        b_arr = np.array([e["b"] for e in self._buffer], dtype=np.float64)
        valid_mask = np.array([e["valid"] for e in self._buffer], dtype=bool)

        raw_signal = TemporalRGBSignal(
            timestamps=timestamps, r=r_arr, g=g_arr, b=b_arr, sampling_rate_hz=self.target_fps
        )

        try:
            # 1. Preprocess: uniform resampling & detrending
            prep_signal, jitter_std_ms, motion_score = self.preprocessor.process(raw_signal)

            # 2. Algorithm projection (POS / CHROM)
            raw_pulse = self.algorithm.extract_pulse(prep_signal)

            # 3. Butterworth Zero-Phase Bandpass Filter
            filtered_pulse = self.bandpass_filter.filter_pulse_waveform(raw_pulse)

            # 4. Heart Rate Estimation & Quality Assessment
            valid_ratio = float(np.mean(valid_mask))
            hr_est, quality, freqs, psd = self.hr_estimator.estimate_heart_rate(
                pulse=filtered_pulse,
                valid_frame_ratio=valid_ratio,
                jitter_std_ms=jitter_std_ms,
                motion_score=motion_score,
            )

            self._latest_bpm = hr_est.bpm
            self._latest_confidence = hr_est.confidence
            self._latest_snr_db = quality.snr_db
            self._latest_status = quality.status
            self._latest_reason = hr_est.rejection_reason

            # If BPM passed quality check, compute the 4 optical features for SOMYTHRA Twin
            if hr_est.is_reliable and hr_est.bpm is not None:
                self._latest_optical_features = self._compute_optical_features(
                    filtered_pulse, hr_est.bpm
                )
            else:
                self._latest_optical_features = None

        except Exception as e:
            logger.warning("Cadence update error: %s", e)
            self._latest_status = QualityStatus.UNPROCESSED
            self._latest_reason = str(e)
            self._latest_bpm = None

    def _compute_optical_features(
        self, pulse_wave: Any, bpm: float
    ) -> Dict[str, float]:
        """
        Derive optical pulse features from the filtered pulse waveform
        for the WESAD optical autonomic model:
        1. mean_hr : Heart rate in BPM
        2. rmssd   : Root mean square of successive peak intervals (ms)
        3. pnn50   : Percentage of successive intervals differing by > 50 ms
        4. lf_hf   : Low-to-high frequency power ratio
        """
        sig = pulse_wave.signal
        fs = pulse_wave.sampling_rate_hz

        # Simple peak detection via zero-crossings of first derivative
        diff = np.diff(sig)
        zero_crossings = np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0]

        # Calculate peak-to-peak intervals in milliseconds
        if len(zero_crossings) >= 3:
            ibis_ms = np.diff(zero_crossings) / fs * 1000.0
            # Physiological clamp: 300ms (200 BPM) to 1500ms (40 BPM)
            ibis_ms = ibis_ms[(ibis_ms >= 300.0) & (ibis_ms <= 1500.0)]
            if len(ibis_ms) >= 2:
                successive_diffs = np.diff(ibis_ms)
                rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
                pnn50 = float(np.mean(np.abs(successive_diffs) > 50.0) * 100.0)
            else:
                # Approximation from BPM
                rmssd = float(max(15.0, 50.0 - 0.3 * (bpm - 60.0)))
                pnn50 = float(np.clip((rmssd - 18.0) * 0.6, 0.0, 100.0))
        else:
            rmssd = float(max(15.0, 50.0 - 0.3 * (bpm - 60.0)))
            pnn50 = float(np.clip((rmssd - 18.0) * 0.6, 0.0, 100.0))

        # Spectral LF/HF ratio from pulse FFT
        n_fft = 2048
        fft_vals = np.abs(np.fft.rfft(sig, n=n_fft)) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.40)
        lf_power = float(np.sum(fft_vals[lf_mask]))
        hf_power = float(np.sum(fft_vals[hf_mask]))

        lf_hf = float(np.clip(lf_power / max(1e-4, hf_power), 0.1, 10.0))

        return {
            "mean_hr": round(float(bpm), 2),
            "rmssd": round(rmssd, 2),
            "pnn50": round(pnn50, 2),
            "lf_hf": round(lf_hf, 2),
        }
