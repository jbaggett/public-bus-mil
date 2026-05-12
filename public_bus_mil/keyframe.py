"""BUV keyframe selection algorithm — Laplacian variance + min_interval + top-K.

Reproduces the algorithm used to produce `BUV_Extracted/manifest.json`
from the MICCAI-BUV source distribution. See DATASET_SPEC.md §4.

Input: iterable of (frame_idx, frame_array) pairs for one clip.
Output: ordered list of selected (frame_idx, sharpness_score) tuples.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from scipy.ndimage import laplace


def laplacian_variance(frame: np.ndarray) -> float:
    """Variance of the Laplacian — standard blur metric.

    Higher = sharper. Pure scipy implementation (no OpenCV dependency).

    Parameters
    ----------
    frame : np.ndarray
        2-D (H, W) grayscale array, or (H, W, 3) RGB (converted internally).

    Returns
    -------
    float
        Variance of the Laplacian response. Typical range for clinical US:
        20–400. Severely blurry frames score < 10.
    """
    if frame.ndim == 3:
        # ITU-R BT.601 luma
        frame = (
            0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
        )
    frame = frame.astype(np.float32)
    lap = laplace(frame)
    return float(lap.var())


def select_keyframes(
    sharpness: List[Tuple[int, float]],
    *,
    frames_per_video: int = 8,
    min_interval: int = 10,
    quality_threshold: float = 0.7,
) -> List[Tuple[int, float]]:
    """Select keyframes per the BUV algorithm with uniform fallback.

    Parameters
    ----------
    sharpness : list of (frame_idx, score)
        Per-frame sharpness scores for the clip, ordered by frame_idx.
    frames_per_video : int
        Target number of frames to return. The algorithm always produces
        this many frames if the clip has at least `frames_per_video`
        distinct frames; if the clip is shorter, all frames are returned.
    min_interval : int
        Minimum gap (in original-video frame indices) between any two
        frames selected by the quality-based phase. The uniform-fallback
        phase relaxes this to 1.
    quality_threshold : float
        Fraction of the clip's max sharpness below which frames are
        dropped in the quality-based phase. 0.7 means: drop frames with
        sharpness < 0.7 × max.

    Returns
    -------
    list of (frame_idx, score)
        Selected frames, ordered by frame_idx ascending. Length is
        min(frames_per_video, len(sharpness)).

    Algorithm
    ---------
    Phase 1 (quality-based, BUV original):
      1. Drop frames below `quality_threshold × max`.
      2. Sort surviving frames by sharpness descending.
      3. Greedy selection respecting `min_interval` gap until
         `frames_per_video` selected or candidates exhausted.

    Phase 2 (uniform fallback, added 2026-05-12 after discovering
    WHBUS clips often have outlier max sharpness making 0.7×max
    unreachable for all but 1–2 frames):
      If fewer than `frames_per_video` selected, fill from the full
      frame set with uniformly-spaced indices, skipping frames already
      selected. The fallback ensures every clip with at least
      `frames_per_video` frames yields exactly `frames_per_video`
      keyframes; BUV's existing manifest is unaffected because that
      manifest was produced before this fallback existed.
    """
    if not sharpness:
        return []

    # Phase 1: quality-based greedy selection with min_interval
    max_score = max(s for _, s in sharpness)
    threshold = quality_threshold * max_score
    eligible = [(idx, s) for idx, s in sharpness if s >= threshold]
    eligible.sort(key=lambda t: -t[1])  # by score descending

    selected: List[Tuple[int, float]] = []
    for idx, s in eligible:
        if len(selected) >= frames_per_video:
            break
        if all(abs(idx - sel_idx) >= min_interval for sel_idx, _ in selected):
            selected.append((idx, s))

    # Phase 2: uniform-fallback fill to frames_per_video
    target = min(frames_per_video, len(sharpness))
    if len(selected) < target:
        selected_idxs = {idx for idx, _ in selected}
        all_sorted = sorted(sharpness, key=lambda t: t[0])  # by frame_idx
        remaining = [t for t in all_sorted if t[0] not in selected_idxs]
        n_needed = target - len(selected)
        if remaining and n_needed > 0:
            # Uniformly-spaced indices into `remaining`
            step = len(remaining) / n_needed
            for k in range(n_needed):
                pos = int(round(step * (k + 0.5)))
                pos = min(pos, len(remaining) - 1)
                pick = remaining[pos]
                if pick[0] in selected_idxs:
                    # Linear scan for the nearest unselected frame
                    for offset in range(1, len(remaining)):
                        for direction in (-1, +1):
                            j = pos + direction * offset
                            if 0 <= j < len(remaining) and remaining[j][0] not in selected_idxs:
                                pick = remaining[j]
                                break
                        if pick[0] not in selected_idxs:
                            break
                if pick[0] not in selected_idxs:
                    selected.append(pick)
                    selected_idxs.add(pick[0])

    selected.sort(key=lambda t: t[0])  # by frame_idx ascending
    return selected
