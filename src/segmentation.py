"""segmentation.py — Gait segmentation via LiveDTW with EMA smoothing.

Overview
--------
Segments a motion stream into labelled gait events by comparing live frames
against one or more reference recordings using online (streaming) DTW.

The core signal is the **DTW increment**: the marginal change in cumulative
cost when a new live frame arrives — ``D[N, j] - D[N, j-1]``, clipped to
``[0, ∞)``.  During the reference motion the increment is near zero; during
idle motion it is large.  An exponential moving average (EMA) smooths the
signal before comparison.

Each reference gait gets its own detection threshold, derived automatically
from a calibration pass over a representative stream using Otsu's method on
the bimodal increment distribution.

Typical usage
-------------
::

    import segmentation as seg
    from dtw import LiveDTW
    from motion import MotionStream

    ref    = MotionStream("reference.bvh")
    stream = MotionStream("recording.bvh")

    detector = seg.GaitDetector([(LiveDTW(ref), "Pick up")])
    detector.calibrate(stream)
    segments = detector.detect(stream)

    for begin, end, label in segments:
        print(f"[{begin}:{end}] {label}")

    # Or as a one-liner with method chaining:
    segments = seg.GaitDetector(gaits).calibrate(stream).detect(stream)

Constants
---------
EMA_ALPHA : float
    Default smoothing factor for the EMA applied to the raw DTW increment.
    Lower values produce a smoother (but more lagged) signal.  Default: 0.04.

WARMUP_FRAMES : int
    Default number of frames to skip after ``frames_seen`` first reaches
    ``N`` (the reference length).  At ``j == N`` the DTW always yields a
    zero increment, which would be a false detection.  Default: 30.
"""

import numpy as np

from dtw import LiveDTW
from motion import MotionStream


# ── Module-level defaults ────────────────────────────────────────────────────

EMA_ALPHA = 0.04
WARMUP_FRAMES = 30

_Gaits = list[tuple[LiveDTW, str]]


# ── GaitDetector ─────────────────────────────────────────────────────────────

class GaitDetector:
    """Online gait segmentation against a fixed set of reference recordings.

    Uses Dynamic Time Warping (DTW) increments smoothed with an exponential
    moving average (EMA) to identify when a live motion stream matches a
    reference gait.  Each reference gets its own detection threshold, derived
    automatically by Otsu's method during calibration.

    State machine
    -------------
    Two states: IDLE and ACTIVE.  At each frame every gait's EMA-smoothed
    increment is divided by its threshold.  The gait with the lowest
    normalised score wins.  Score ≤ 1.0 triggers ACTIVE; rising back above
    1.0 returns to IDLE.  All DTW objects are reset on every state *exit*,
    allowing the same gait to fire again (recurring detection).

    Parameters
    ----------
    gaits:
        List of ``(LiveDTW, label)`` pairs — one per reference recording.
    ema_alpha:
        EMA smoothing factor for the DTW increment signal.
        Lower = smoother but more lag.  Defaults to ``EMA_ALPHA`` (0.04).
    warmup_frames:
        Frames to ignore after the DTW warmup period ends.  Guards against
        the artificial zero increment that occurs at ``j == N``.
        Defaults to ``WARMUP_FRAMES`` (30).

    Attributes
    ----------
    gaits : list[tuple[LiveDTW, str]]
        The ``(LiveDTW, label)`` pairs supplied at construction.
    thresholds : list[float] or None
        Per-gait Otsu thresholds; ``None`` until ``calibrate()`` is called.
    ema_alpha : float
        EMA smoothing factor in use.
    warmup_frames : int
        Warmup guard length in use.

    Raises
    ------
    RuntimeError
        ``detect()`` raises if called before ``calibrate()``.
    """

    def __init__(
        self,
        gaits: _Gaits,
        ema_alpha: float = EMA_ALPHA,
        warmup_frames: int = WARMUP_FRAMES,
    ) -> None:
        self.gaits = gaits
        self.ema_alpha = ema_alpha
        self.warmup_frames = warmup_frames
        self.thresholds: list[float] | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    def calibrate(self, stream: MotionStream) -> "GaitDetector":
        """Run a forward pass over *stream* to compute per-gait thresholds.

        Collects EMA-smoothed DTW increments for every gait over the full
        stream, then applies Otsu's method independently to each gait's
        distribution to find the threshold that best separates the motion
        and idle modes.

        All gaits are reset after the pass so both *stream* and the detector
        can be reused immediately for detection.

        Parameters
        ----------
        stream:
            A representative ``MotionStream`` to calibrate against.  It will
            be iterated from its current position.

        Returns
        -------
        GaitDetector
            Returns *self* to allow method chaining.
        """
        gait_increments: list[list[float]] = [[] for _ in self.gaits]
        ema = [np.nan] * len(self.gaits)
        for _, vel in stream:
            if vel is None:
                continue
            for i, (gait, _) in enumerate(self.gaits):
                gait.update(vel)
                if gait.frames_seen >= gait._N + self.warmup_frames:
                    raw = gait.increment
                    if np.isnan(ema[i]):
                        ema[i] = raw
                    else:
                        ema[i] = (
                            self.ema_alpha * raw
                            + (1 - self.ema_alpha) * ema[i]
                        )
                    gait_increments[i].append(ema[i])
        self._reset()
        self.thresholds = [
            self._otsu_threshold(np.array(incs))
            if len(incs) > 1 else np.inf
            for incs in gait_increments
        ]
        return self

    def detect(
        self, stream: MotionStream
    ) -> list[tuple[int, int, str]]:
        """Detect labelled gait segments in *stream*.

        Iterates over *stream* frame by frame, maintaining per-gait EMA
        state and a two-state (IDLE / ACTIVE) machine.

        Gaits are reset on every state *exit* so the same label can fire
        again once the motion resumes (recurring detection).  There is an
        inherent dead zone of ``N + warmup_frames`` frames between
        consecutive detections of the same gait.

        Parameters
        ----------
        stream:
            The ``MotionStream`` to segment.  Iterated from its current
            position.

        Returns
        -------
        list[tuple[int, int, str]]
            ``(begin, end, label)`` tuples with **inclusive** frame indices,
            in chronological order.

        Raises
        ------
        RuntimeError
            If ``calibrate()`` has not been called yet.
        """
        if self.thresholds is None:
            raise RuntimeError("call calibrate() before detect()")
        thresholds = self.thresholds

        detected: list[tuple[int, int, str]] = []
        last_label: str | None = None
        detection_begin = 0
        last_match_frame = 0
        ema = [np.nan] * len(self.gaits)

        def _clear() -> None:
            nonlocal ema
            self._reset()
            ema = [np.nan] * len(self.gaits)

        for frame_index, (_, vel) in enumerate(stream):
            if vel is None:
                continue
            for gait, _ in self.gaits:
                gait.update(vel)

            for i, (g, _) in enumerate(self.gaits):
                if g.frames_seen >= g._N + self.warmup_frames:
                    raw = g.increment
                    if np.isnan(ema[i]):
                        ema[i] = raw
                    else:
                        ema[i] = (
                            self.ema_alpha * raw
                            + (1 - self.ema_alpha) * ema[i]
                        )

            scores = [
                ema[i] / thresholds[i]
                if (
                    g.frames_seen >= g._N + self.warmup_frames
                    and not np.isnan(ema[i])
                )
                else np.inf
                for i, (g, _) in enumerate(self.gaits)
            ]
            min_idx = int(np.argmin(scores))
            _, label = self.gaits[min_idx]

            if scores[min_idx] <= 1.0:
                last_match_frame = frame_index
                if label != last_label:
                    if last_label is not None:
                        detected.append(
                            (detection_begin, frame_index - 1, last_label)
                        )
                        _clear()
                    detection_begin = frame_index
                    last_label = label
            else:
                if last_label is not None:
                    detected.append(
                        (detection_begin, last_match_frame, last_label)
                    )
                    last_label = None
                    _clear()

        if last_label is not None:
            detected.append(
                (detection_begin, last_match_frame, last_label)
            )

        self._reset()
        return detected

    # ── Private helpers ──────────────────────────────────────────────────────

    def _reset(self) -> None:
        """Reset all LiveDTW objects to their initial state."""
        for gait, _ in self.gaits:
            gait.reset()

    @staticmethod
    def _otsu_threshold(values: np.ndarray) -> float:
        """Return Otsu's optimal threshold for a 1-D array.

        Finds the split point that maximises between-class variance, assuming
        a bimodal distribution of near-zero (motion) vs large (idle) values.

        Parameters
        ----------
        values:
            1-D array of non-negative floats (EMA-smoothed DTW increments).

        Returns
        -------
        float
            Threshold separating the two distribution modes.
        """
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
