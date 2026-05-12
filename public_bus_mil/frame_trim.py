"""Dark-margin trim for ultrasound video frames.

Some WHBUS clips carry a thin black border (up to ~43 pixels on the bottom
in the worst case observed). BUV frames have no such border. The trim is
a no-op when no margin is present.

The algorithm:
  1. Convert to grayscale if needed.
  2. Compute mean intensity per row and per column.
  3. Find the first row/column from each edge with mean > THR (default 8).
  4. Crop to the bounding rectangle.

THR=8 is chosen well below typical tissue intensity (≥20) and well above
JPEG/PNG compression noise on a true-black region.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def trim_dark_margins(img: Image.Image, thr: float = 8.0) -> Image.Image:
    """Trim near-black edges where row/col mean intensity is below `thr`.

    Parameters
    ----------
    img : PIL.Image
        Input frame (any mode; converted to grayscale internally for the
        intensity profile, but the original-mode image is what gets
        cropped and returned).
    thr : float
        Intensity threshold below which a row/column is considered "dark"
        and trimmed. Default 8 (out of 255).

    Returns
    -------
    PIL.Image
        Cropped image. If no margins are below `thr` the input is returned
        unchanged.
    """
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    H, W = gray.shape

    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)

    top = _first_above(row_means, thr)
    bot = _last_above(row_means, thr)
    lft = _first_above(col_means, thr)
    rgt = _last_above(col_means, thr)

    if (top, bot, lft, rgt) == (0, H - 1, 0, W - 1):
        return img  # no trim needed

    # PIL crop box is (left, upper, right, lower); right/lower are exclusive.
    return img.crop((lft, top, rgt + 1, bot + 1))


def _first_above(profile: np.ndarray, thr: float) -> int:
    """Return the first index where profile > thr; 0 if all are below."""
    above = np.where(profile > thr)[0]
    return int(above[0]) if above.size else 0


def _last_above(profile: np.ndarray, thr: float) -> int:
    """Return the last index where profile > thr; len-1 if all are below."""
    above = np.where(profile > thr)[0]
    return int(above[-1]) if above.size else len(profile) - 1
