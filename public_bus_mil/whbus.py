"""WHBUS loader and bag builder.

Consumes:
  - WHBUS/buvimgs/{benign,malignant}/<clip_folder>/<frame>.png

WHBUS clip folders are named like `benign_9161001_10`:
  benign|malignant _ patient_id _ clip_idx

Patient_id is recorded as audit metadata only — bag construction is
per-clip (see DATASET_SPEC.md §3).

Frame selection runs the BUV keyframe algorithm (Laplacian variance +
min_interval + top-K) per clip. See DATASET_SPEC.md §4.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from .frame_trim import trim_dark_margins
from .keyframe import laplacian_variance, select_keyframes

CLIP_NAME_RX = re.compile(r"^(benign|malignant)_(\d+)_(\d+)$")


def build_whbus_bags(
    whbus_dir: str | Path,
    out_dir: Path,
    bag_id_start: int = 1,
    frames_per_video: int = 8,
    min_interval: int = 15,
    quality_threshold: float = 0.7,
) -> Tuple[List[dict], List[dict]]:
    """Build WHBUS MIL bags.

    Returns (bag_rows, frame_rows) suitable for write_manifest().
    """
    whbus_dir = Path(whbus_dir)
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    bag_rows: List[dict] = []
    frame_rows: List[dict] = []
    next_bag_id = bag_id_start

    # Discover all clips, deterministically ordered for reproducibility
    clip_paths = []
    for label_str in ("benign", "malignant"):
        sub = whbus_dir / label_str
        if not sub.exists():
            continue
        for clip_dir in sorted(sub.iterdir()):
            if clip_dir.is_dir():
                clip_paths.append((label_str, clip_dir))

    for label_str, clip_dir in clip_paths:
        m = CLIP_NAME_RX.match(clip_dir.name)
        patient_id = f"whbus_{m.group(2)}" if m else "whbus_unknown"
        label = 1 if label_str == "malignant" else 0

        bag_id = f"whbus_{next_bag_id:04d}"
        next_bag_id += 1

        # Compute sharpness for every frame in the clip
        frame_files = sorted(clip_dir.glob("*.png"))
        sharpness = []
        for fp in frame_files:
            # Frame index is the integer in the filename (000000, 000001, ...)
            try:
                fidx = int(fp.stem)
            except ValueError:
                continue
            arr = np.asarray(Image.open(fp).convert("L"), dtype=np.float32)
            sharpness.append((fidx, laplacian_variance(arr)))
        if not sharpness:
            continue

        selected = select_keyframes(
            sharpness,
            frames_per_video=frames_per_video,
            min_interval=min_interval,
            quality_threshold=quality_threshold,
        )

        for i, (original_fidx, score) in enumerate(selected):
            src = clip_dir / f"{original_fidx:06d}.png"
            if not src.exists():
                continue
            img = trim_dark_margins(Image.open(src))
            dst_name = f"{bag_id}_{i:03d}.png"
            dst = out_images / dst_name
            img.save(dst)

            frame_rows.append({
                "bag_id": bag_id,
                "frame_idx_in_bag": i,
                "src_path": f"images/{dst_name}",
                "original_video_frame_idx": original_fidx,
                "sharpness_score": score,
                "has_bbox": False,
                "n_bboxes": 0,
                "bbox_x": "", "bbox_y": "", "bbox_w": "", "bbox_h": "",
            })

        bag_rows.append({
            "bag_id": bag_id,
            "dataset": "whbus",
            "source_id": clip_dir.name,
            "label": label,
            "n_frames": len(selected),
            "patient_id": patient_id,
            "has_bboxes": False,
        })

    return bag_rows, frame_rows
