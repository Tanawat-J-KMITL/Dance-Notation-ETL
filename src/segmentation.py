"""segmentation.py — Gait segmentation via subsequence LiveDTW.

Overview
--------
Segments a motion stream into labelled gait events by comparing live frames
against one or more reference recordings using online (streaming) DTW.

Detection pipeline
------------------
    raw subseq_distance  →  EMA smoothing  →  Otsu threshold  →  detect

For each live frame the raw ``LiveDTW.subseq_distance`` is fed through a
per-gait exponential moving average (EMA).  During calibration:

1. Raw ``subseq_distance`` is collected after a warmup of
   ``max(reference_N) + WARMUP_FRAMES`` frames to let the DP settle.
2. The EMA alpha that maximises Otsu's between-class variance on the
   smoothed signal is auto-detected.
3. Otsu's threshold is computed on that EMA-smoothed calibration signal.

At detection time the same EMA (fresh start, same alpha) is applied
frame-by-frame and the smoothed value is compared to the threshold.
The EMA state is reset whenever a detection segment ends and the DTW
probes are reset, so every segment starts with a clean slate.

Typical usage
-------------
::

    import segmentation as seg
    from dtw import LiveDTW
    from motion import MotionStream

    ref    = MotionStream("reference.bvh")
    stream = MotionStream("recording.bvh")

    detector = seg.GaitDetector([(LiveDTW(ref), "Pick up")],
                                detect_high=True)
    detector.calibrate(stream)
    segments = detector.detect(stream)

    for begin, end, label in segments:
        print(f"[{begin}:{end}] {label}")

Constants
---------
WARMUP_FRAMES : int
    Extra frames added on top of the reference length during warmup.
    Default: 30.
"""

import numpy as np

from dtw import LiveDTW
from motion import MotionStream


# ── Module-level defaults ────────────────────────────────────────────────────

WARMUP_FRAMES = 30

_Gaits = list[tuple[LiveDTW, str]]


# ── GaitDetector ─────────────────────────────────────────────────────────────

class GaitDetector:
    """Online gait segmentation against a fixed set of reference recordings.

    Pipeline: raw subseq_distance → EMA smoothing → Otsu threshold → detect.

    The warmup defaults to ``max(reference_N) + WARMUP_FRAMES`` so the DP
    is fully settled before any distance value is used.  The EMA alpha is
    auto-detected during calibration by maximising Otsu's between-class
    variance over a sweep of candidate values.

    State machine
    -------------
    Two states: IDLE and ACTIVE.  At each frame the EMA-smoothed distance
    for the winning gait is compared to its threshold.  With
    ``detect_high=False`` (default) score ≤ 1.0 triggers ACTIVE; with
    ``detect_high=True`` score ≥ 1.0 triggers ACTIVE.  All DTW objects and
    the EMA state are reset on every state exit so the same gait can fire
    again cleanly.

    Parameters
    ----------
    gaits:
        List of ``(LiveDTW, label)`` pairs — one per reference recording.
    warmup_frames:
        Frames to skip before distances are used.  ``None`` (default)
        auto-computes ``max(ref_N) + WARMUP_FRAMES`` across all gaits.
    detect_high:
        When ``True``, fire when EMA distance is *above* threshold.
        Use when high distance indicates the target motion.

    Attributes
    ----------
    gaits : list[tuple[LiveDTW, str]]
    thresholds : list[float] or None
        Per-gait Otsu thresholds on the EMA-smoothed signal.
    ema_alphas : list[float] or None
        Per-gait auto-detected EMA smoothing factors.
    warmup_frames : int

    Raises
    ------
    RuntimeError
        ``detect()`` raises if called before ``calibrate()``.
    """

    def __init__(
        self,
        gaits: _Gaits,
        warmup_frames: int | None = None,
        detect_high: bool = False,
    ) -> None:
        if warmup_frames is None:
            warmup_frames = max(g._N for g, _ in gaits) + WARMUP_FRAMES
        self.gaits         = gaits
        self.warmup_frames = warmup_frames
        self.detect_high   = detect_high
        self.thresholds: list[float] | None = None
        self.ema_alphas:  list[float] | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    def calibrate(self, stream: MotionStream) -> "GaitDetector":
        """Calibrate per-gait EMA alphas and Otsu thresholds.

        Pipeline: collect raw subseq_distance → auto-detect EMA alpha →
        apply EMA → Otsu threshold.

        Returns *self* for method chaining.
        """
        raw_dists: list[list[float]] = [[] for _ in self.gaits]
        for _, vel in stream:
            if vel is None:
                continue
            for i, (gait, _) in enumerate(self.gaits):
                gait.update(vel)
                if gait.frames_seen >= self.warmup_frames:
                    raw_dists[i].append(gait.subseq_distance)
        self._reset()

        # Auto-detect EMA alpha: maximise Otsu between-class variance
        self.ema_alphas = []
        for raw in raw_dists:
            if len(raw) < 10:
                self.ema_alphas.append(0.1)
                continue
            best_alpha, best_var = 0.1, -1.0
            for a in np.linspace(0.01, 0.5, 40):
                smoothed = np.array(self._apply_ema(raw, a))
                v = self._otsu_between_class_var(smoothed)
                if v > best_var:
                    best_var, best_alpha = v, a
            self.ema_alphas.append(best_alpha)

        # Otsu threshold on EMA-smoothed calibration distribution
        self.thresholds = [
            self._otsu_threshold(np.array(self._apply_ema(raw, a)))
            if len(raw) > 1 else np.inf
            for raw, a in zip(raw_dists, self.ema_alphas)
        ]
        return self

    def detect(self, stream: MotionStream) -> list[tuple[int, int, str]]:
        """Detect labelled gait segments in *stream*.

        Applies per-gait EMA to ``subseq_distance`` in real time and runs
        the two-state (IDLE / ACTIVE) machine against the calibrated
        thresholds.  EMA state and DTW probes are both reset at segment
        boundaries for a clean slate.

        Raises ``RuntimeError`` if called before ``calibrate()``.
        """
        if self.thresholds is None:
            raise RuntimeError("call calibrate() before detect()")

        thresholds = self.thresholds
        ema_states = [float("nan")] * len(self.gaits)

        detected: list[tuple[int, int, str]] = []
        last_label: str | None = None
        detection_begin = 0
        last_match_frame = 0

        for frame_index, (_, vel) in enumerate(stream):
            if vel is None:
                continue
            for gait, _ in self.gaits:
                gait.update(vel)

            scores: list[float] = []
            for i, (g, _) in enumerate(self.gaits):
                if g.frames_seen >= self.warmup_frames:
                    raw = g.subseq_distance
                    a   = self.ema_alphas[i]
                    ema_states[i] = (
                        raw if np.isnan(ema_states[i])
                        else a * raw + (1 - a) * ema_states[i]
                    )
                    scores.append(ema_states[i] / thresholds[i])
                else:
                    scores.append(0.0 if self.detect_high else np.inf)

            min_idx = int(np.argmin(scores))
            _, label = self.gaits[min_idx]

            triggered = (
                scores[min_idx] >= 1.0 if self.detect_high
                else scores[min_idx] <= 1.0
            )
            if triggered:
                last_match_frame = frame_index
                if label != last_label:
                    if last_label is not None:
                        detected.append(
                            (detection_begin, frame_index - 1, last_label)
                        )
                        self._reset_all(ema_states)
                    detection_begin = frame_index
                    last_label = label
            else:
                if last_label is not None:
                    detected.append(
                        (detection_begin, last_match_frame, last_label)
                    )
                    last_label = None
                    self._reset_all(ema_states)

        if last_label is not None:
            detected.append((detection_begin, last_match_frame, last_label))
        self._reset()
        return detected

    # ── Private helpers ──────────────────────────────────────────────────────

    def _reset(self) -> None:
        for gait, _ in self.gaits:
            gait.reset()

    def _reset_all(self, ema_states: list) -> None:
        """Reset DTW probes and clear EMA state for a clean segment start."""
        self._reset()
        for i in range(len(ema_states)):
            ema_states[i] = float("nan")

    @staticmethod
    def _apply_ema(values: list[float], alpha: float) -> list[float]:
        out, state = [], float("nan")
        for v in values:
            state = v if np.isnan(state) else alpha * v + (1 - alpha) * state
            out.append(state)
        return out

    @staticmethod
    def _otsu_between_class_var(values: np.ndarray) -> float:
        hist, edges = np.histogram(values, bins=256, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        best = 0.0
        for i in range(1, len(hist)):
            w0, w1 = hist[:i].sum(), hist[i:].sum()
            if w0 == 0 or w1 == 0:
                continue
            m0 = np.average(centers[:i], weights=hist[:i])
            m1 = np.average(centers[i:], weights=hist[i:])
            best = max(best, w0 * w1 * (m0 - m1) ** 2)
        return best

    @staticmethod
    def _otsu_threshold(values: np.ndarray) -> float:
        hist, edges = np.histogram(values, bins=256, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        best_t, best_var = edges[0], 0.0
        for i in range(1, len(hist)):
            w0, w1 = hist[:i].sum(), hist[i:].sum()
            if w0 == 0 or w1 == 0:
                continue
            m0 = np.average(centers[:i], weights=hist[:i])
            m1 = np.average(centers[i:], weights=hist[i:])
            var = w0 * w1 * (m0 - m1) ** 2
            if var > best_var:
                best_var, best_t = var, edges[i]
        return float(best_t)
