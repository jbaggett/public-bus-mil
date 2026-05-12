"""Output-manifest schema helpers.

The benchmark emits two CSVs:
  - manifest.csv : bag-level (one row per MIL bag)
  - frames.csv   : frame-level (one row per (bag, frame) pair)

Schemas are stable across versions; adding columns is allowed,
renaming or removing columns is a breaking change requiring a v2.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import csv

BAG_COLUMNS = [
    "bag_id",        # globally unique, e.g. "buv_0001"
    "dataset",       # "buv" or "whbus"
    "source_id",     # native id (video hash for BUV, clip folder name for WHBUS)
    "label",         # 0 = benign, 1 = malignant
    "n_frames",      # number of frames in this bag
    "patient_id",    # inferred where available; "unknown" otherwise
    "has_bboxes",    # True if any frame in this bag has a bbox annotation
]

FRAME_COLUMNS = [
    "bag_id",
    "frame_idx_in_bag",     # 0..n_frames-1
    "src_path",             # relative path under output dir, e.g. images/buv_0001_000.png
    "original_video_frame_idx",  # frame index in the original video
    "sharpness_score",      # Laplacian variance at selection time
    "has_bbox",
    "n_bboxes",             # number of bbox annotations on this frame (0+; we emit the largest if >1)
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",  # in native (post-trim) pixel coords; blank if no bbox
]


def write_manifest(
    bag_rows: List[Dict[str, Any]],
    frame_rows: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    """Write manifest.csv and frames.csv to `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BAG_COLUMNS)
        w.writeheader()
        for row in bag_rows:
            w.writerow({c: row.get(c, "") for c in BAG_COLUMNS})

    with open(out_dir / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FRAME_COLUMNS)
        w.writeheader()
        for row in frame_rows:
            w.writerow({c: row.get(c, "") for c in FRAME_COLUMNS})
