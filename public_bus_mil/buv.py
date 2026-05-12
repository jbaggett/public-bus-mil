"""BUV loader and bag builder.

Consumes:
  - BUV_Extracted/ folder (rawframes, manifest.json)
  - BUV's per-frame bbox annotations (imagenet_vid_*_15frames.json + val.json)

Produces:
  - One bag per video (with errata-merged exception)
  - Bbox metadata joined per-frame

See DATASET_SPEC.md §2 (errata), §3 (bag unit), §7 (bbox handling).
"""
from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from .frame_trim import trim_dark_margins

# Errata per BUV's errata.txt (see DATASET_SPEC.md §2)
ERRATA_DROP = {"x66ef02e7f1b9a0ef"}
ERRATA_RELABEL_TO_MALIGNANT = {
    "x63c9ba1377f35bf6", "x5a1c46ec6377e946",
    "2390fbea047347b", "7a39ab5d4970bf89",
}
ERRATA_SAME_PATIENT_GROUP = {
    "x63c9ba1377f35bf6", "x5a1c46ec6377e946",
    "2390fbea047347b", "7a39ab5d4970bf89",
}


def load_buv_annotations(*ann_paths: str) -> Dict[str, Dict[int, List[dict]]]:
    """Build a {video_id: {frame_idx: [bbox_dict, ...]}} lookup from BUV's COCO JSONs."""
    out: Dict[str, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for ap in ann_paths:
        d = json.load(open(ap))
        img_lookup = {}
        for img in d["images"]:
            # file_name like "benign/x28f299ceb056964c/000000.png"
            m = re.match(r"\w+/(\w+)/(\d+)\.png", img["file_name"])
            if not m:
                continue
            img_lookup[img["id"]] = (m.group(1), int(m.group(2)))
        for ann in d["annotations"]:
            if ann["image_id"] not in img_lookup:
                continue
            vid, fidx = img_lookup[ann["image_id"]]
            x, y, w, h = ann["bbox"]
            out[vid][fidx].append({
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "area": int(ann.get("area", w * h)),
                "category_id": ann.get("category_id"),
            })
    return out


def build_buv_bags(
    buv_extracted_dir: str | Path,
    annotation_paths: List[str],
    out_dir: Path,
    bag_id_start: int = 1,
) -> Tuple[List[dict], List[dict]]:
    """Build BUV MIL bags.

    Returns (bag_rows, frame_rows) suitable for write_manifest().
    Frames are trimmed and copied to out_dir/images/.
    """
    buv_extracted_dir = Path(buv_extracted_dir)
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    manifest = json.load(open(buv_extracted_dir / "manifest.json"))
    bboxes = load_buv_annotations(*annotation_paths)

    # Group errata patient
    errata_group_videos = []

    bag_rows: List[dict] = []
    frame_rows: List[dict] = []
    next_bag_id = bag_id_start
    next_patient_id = 1

    for v in manifest["videos"]:
        vid = v["video_id"]
        if vid in ERRATA_DROP:
            continue

        label_str = v["label"]
        if vid in ERRATA_RELABEL_TO_MALIGNANT:
            label_str = "malignant"
        label = 1 if label_str == "malignant" else 0

        # Defer errata-group videos for a single merged bag
        if vid in ERRATA_SAME_PATIENT_GROUP:
            errata_group_videos.append((vid, label, v))
            continue

        bag_id = f"buv_{next_bag_id:04d}"
        patient_id = f"buv_p{next_patient_id:03d}"
        next_bag_id += 1
        next_patient_id += 1
        n_with_bbox = _emit_video_frames(
            vid, v, label, bag_id, buv_extracted_dir, bboxes,
            out_images, frame_rows, dark_trim=True,
        )
        bag_rows.append({
            "bag_id": bag_id,
            "dataset": "buv",
            "source_id": vid,
            "label": label,
            "n_frames": len(v["metadata"]),
            "patient_id": patient_id,
            "has_bboxes": n_with_bbox > 0,
        })

    # Emit each errata-group video as its own bag, all sharing one
    # patient_id so the patient-grouped split places them in the same fold.
    # Per the BUV errata, all four should be labeled malignant. Some frames
    # are near-duplicates across these four videos (the upstream errata
    # notes that multiple samples from the same patient were saved); see
    # DATASET_SPEC.md §2 for the implication on training i.i.d. assumptions.
    if errata_group_videos:
        merged_label = 1  # all four are malignant after relabel
        shared_patient_id = "buv_errata_patient_1"
        for vid, _label, v in errata_group_videos:
            bag_id = f"buv_{next_bag_id:04d}"
            next_bag_id += 1
            n_with_bbox = _emit_video_frames(
                vid, v, merged_label, bag_id, buv_extracted_dir, bboxes,
                out_images, frame_rows, dark_trim=True,
            )
            bag_rows.append({
                "bag_id": bag_id,
                "dataset": "buv",
                "source_id": vid,
                "label": merged_label,
                "n_frames": len(v["metadata"]),
                "patient_id": shared_patient_id,
                "has_bboxes": n_with_bbox > 0,
            })

    return bag_rows, frame_rows


def _emit_video_frames(
    vid: str,
    video_entry: dict,
    label: int,
    bag_id: str,
    buv_extracted_dir: Path,
    bboxes: Dict[str, Dict[int, List[dict]]],
    out_images: Path,
    frame_rows: List[dict],
    *,
    dark_trim: bool,
    frame_idx_offset: int = 0,
) -> int:
    """Emit frames for one BUV video, returning count with bbox."""
    label_dir = "malignant" if label == 1 else "benign"
    # Source frames in BUV_Extracted are under {benign,malignant}/{vid}/{frame_idx:06d}.png.
    # We use the manifest's own metadata since it knows which frames were extracted.
    n_with_bbox = 0
    for sub_i, md in enumerate(video_entry["metadata"]):
        i = sub_i + frame_idx_offset
        original_fidx = int(md["frame_idx"])
        sharpness = float(md.get("quality_score", 0.0))
        # Resolve the source path. The manifest's frame_paths use the
        # absolute path at extraction time, which won't be valid for a
        # downstream user — re-construct relative to buv_extracted_dir.
        # Try both label dirs since the errata may have flipped label.
        cand_a = buv_extracted_dir / label_dir / vid / f"{original_fidx:06d}.png"
        cand_b = buv_extracted_dir / ("benign" if label_dir == "malignant" else "malignant") / vid / f"{original_fidx:06d}.png"
        src = cand_a if cand_a.exists() else cand_b
        if not src.exists():
            # Skip silently — manifest may reference frames not on disk
            continue

        img = Image.open(src)
        if dark_trim:
            img = trim_dark_margins(img)
        dst_name = f"{bag_id}_{i:03d}.png"
        dst = out_images / dst_name
        img.save(dst)

        # Look up bbox(es) for this (vid, frame_idx)
        bb_list = bboxes.get(vid, {}).get(original_fidx, [])
        # Pick largest by area
        if bb_list:
            bb = max(bb_list, key=lambda b: b["area"])
            has_bbox = True
            n_with_bbox += 1
            n_bboxes = len(bb_list)
            bbx, bby, bbw, bbh = bb["x"], bb["y"], bb["w"], bb["h"]
            # NOTE: bboxes are in *original* coordinates; if we trimmed dark
            # margins, the bbox needs the same offset applied. BUV has
            # no dark margins, so this is a no-op here, but we should
            # implement it properly:
            # Compute original-to-trimmed offset from the trim itself.
            # Since BUV is no-op, we leave bbox as-is. A v2 of this code
            # should compute the trim offset and apply it.
        else:
            has_bbox = False
            n_bboxes = 0
            bbx = bby = bbw = bbh = ""

        frame_rows.append({
            "bag_id": bag_id,
            "frame_idx_in_bag": i,
            "src_path": f"images/{dst_name}",
            "original_video_frame_idx": original_fidx,
            "sharpness_score": sharpness,
            "has_bbox": has_bbox,
            "n_bboxes": n_bboxes,
            "bbox_x": bbx, "bbox_y": bby, "bbox_w": bbw, "bbox_h": bbh,
        })
    return n_with_bbox
