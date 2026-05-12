"""Unit tests for the keyframe-selection algorithm.

Run: pytest tests/
"""
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from public_bus_mil.keyframe import laplacian_variance, select_keyframes
from public_bus_mil.frame_trim import trim_dark_margins
from PIL import Image


def test_laplacian_variance_uniform():
    """Uniform image has zero Laplacian variance."""
    img = np.full((100, 100), 128.0, dtype=np.float32)
    assert laplacian_variance(img) == 0.0


def test_laplacian_variance_noise():
    """Noisy image has high Laplacian variance."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, size=(100, 100), dtype=np.uint8).astype(np.float32)
    assert laplacian_variance(img) > 1000


def test_select_keyframes_min_interval():
    """No two selected frames are within min_interval of each other."""
    sharpness = [(i, 100.0) for i in range(50)]
    sel = select_keyframes(sharpness, frames_per_video=5, min_interval=10, quality_threshold=0.7)
    assert len(sel) == 5
    indices = sorted([s[0] for s in sel])
    for i in range(len(indices) - 1):
        assert indices[i + 1] - indices[i] >= 10


def test_select_keyframes_quality_threshold():
    """Frames below threshold × max are preferred; uniform fallback fills to K."""
    # max sharpness = 100; threshold 0.7 → keep only frames with sharpness >= 70
    sharpness = [(i, float(s)) for i, s in enumerate([100, 90, 50, 80, 30, 70, 20, 95])]

    # When K (5) <= quality-eligible count (5), no fallback, exact same behavior as before
    sel = select_keyframes(sharpness, frames_per_video=5, min_interval=1, quality_threshold=0.7)
    selected_scores = sorted([s[1] for s in sel])
    assert selected_scores == [70.0, 80.0, 90.0, 95.0, 100.0]

    # When K (10) > total frames (8), fallback returns all 8 frames
    sel_all = select_keyframes(sharpness, frames_per_video=10, min_interval=1, quality_threshold=0.7)
    assert len(sel_all) == 8

    # When K (7) is between (quality-eligible = 5) and (total = 8),
    # fallback picks 2 more uniformly from the below-threshold frames
    sel_mid = select_keyframes(sharpness, frames_per_video=7, min_interval=1, quality_threshold=0.7)
    assert len(sel_mid) == 7


def test_select_keyframes_empty():
    """Empty input returns empty list."""
    assert select_keyframes([]) == []


def test_trim_no_op_on_clean_image():
    """Image with no dark margins is returned unchanged."""
    arr = np.full((100, 100, 3), 128, dtype=np.uint8)
    img = Image.fromarray(arr)
    out = trim_dark_margins(img)
    assert out.size == img.size


def test_trim_removes_bottom_margin():
    """Image with a black bottom band has it trimmed."""
    arr = np.full((100, 100, 3), 128, dtype=np.uint8)
    arr[80:, :, :] = 0  # 20-pixel black band on bottom
    img = Image.fromarray(arr)
    out = trim_dark_margins(img, thr=8)
    assert out.size == (100, 80)
