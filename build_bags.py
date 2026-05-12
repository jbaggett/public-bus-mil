#!/usr/bin/env python3
"""Build the public BUS MIL benchmark from one or both source datasets.

Single deterministic CLI. See DATASET_SPEC.md for the protocol.

Usage:
    # Both datasets
    python build_bags.py \\
        --dataset both \\
        --buv-extracted /path/to/BUV_Extracted \\
        --buv-annotations-train /path/to/imagenet_vid_train_15frames.json \\
        --buv-annotations-val   /path/to/imagenet_vid_val.json \\
        --whbus /path/to/WHBUS/buvimgs \\
        --out ./external_test_v1/

    # BUV only
    python build_bags.py --dataset buv \\
        --buv-extracted ... --buv-annotations-train ... --buv-annotations-val ... \\
        --out ./out/

    # WHBUS only
    python build_bags.py --dataset whbus --whbus ... --out ./out/
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

from public_bus_mil import write_manifest, __version__
from public_bus_mil.buv import build_buv_bags
from public_bus_mil.whbus import build_whbus_bags


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the public BUS MIL benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        choices=["buv", "whbus", "both"],
        required=True,
        help="Which source dataset(s) to include.",
    )
    p.add_argument(
        "--buv-extracted",
        type=Path,
        help="Path to BUV_Extracted/ directory (required if --dataset includes buv).",
    )
    p.add_argument(
        "--buv-annotations-train",
        type=Path,
        help="Path to imagenet_vid_train_15frames.json (required if --dataset includes buv).",
    )
    p.add_argument(
        "--buv-annotations-val",
        type=Path,
        help="Path to imagenet_vid_val.json (required if --dataset includes buv).",
    )
    p.add_argument(
        "--whbus",
        type=Path,
        help="Path to WHBUS/buvimgs/ directory (required if --dataset includes whbus).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (will be created).",
    )
    # Algorithm parameters (defaults match DATASET_SPEC §4)
    p.add_argument("--whbus-frames-per-clip", type=int, default=8)
    p.add_argument("--whbus-min-interval", type=int, default=15)
    p.add_argument("--whbus-quality-threshold", type=float, default=0.7)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    bag_rows: list = []
    frame_rows: list = []

    if args.dataset in ("buv", "both"):
        if not all([args.buv_extracted, args.buv_annotations_train, args.buv_annotations_val]):
            print("ERROR: --buv-extracted, --buv-annotations-train, and --buv-annotations-val required for buv", file=sys.stderr)
            return 2
        print(f"=== Building BUV bags from {args.buv_extracted} ===")
        b_bags, b_frames = build_buv_bags(
            buv_extracted_dir=args.buv_extracted,
            annotation_paths=[str(args.buv_annotations_train), str(args.buv_annotations_val)],
            out_dir=out,
            bag_id_start=len(bag_rows) + 1,
        )
        bag_rows.extend(b_bags)
        frame_rows.extend(b_frames)
        print(f"  → {len(b_bags)} bags, {len(b_frames)} frames")

    if args.dataset in ("whbus", "both"):
        if not args.whbus:
            print("ERROR: --whbus required for whbus", file=sys.stderr)
            return 2
        print(f"=== Building WHBUS bags from {args.whbus} ===")
        w_bags, w_frames = build_whbus_bags(
            whbus_dir=args.whbus,
            out_dir=out,
            bag_id_start=1,  # whbus bags have their own bag_id namespace via prefix
            frames_per_video=args.whbus_frames_per_clip,
            min_interval=args.whbus_min_interval,
            quality_threshold=args.whbus_quality_threshold,
        )
        bag_rows.extend(w_bags)
        frame_rows.extend(w_frames)
        print(f"  → {len(w_bags)} bags, {len(w_frames)} frames")

    print(f"\n=== Writing manifests to {out} ===")
    write_manifest(bag_rows, frame_rows, out)

    # Write BUILD_LOG.json for provenance
    log = {
        "build_timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "tool_version": __version__,
        "cli_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "bag_count": len(bag_rows),
        "frame_count": len(frame_rows),
        "bag_count_by_dataset": _count_by(bag_rows, "dataset"),
        "bag_count_by_label": _count_by(bag_rows, "label"),
    }
    (out / "BUILD_LOG.json").write_text(json.dumps(log, indent=2))
    print(f"  BUILD_LOG.json written")

    print(f"\nDone:")
    print(f"  Total bags  : {len(bag_rows)}")
    print(f"  Total frames: {len(frame_rows)}")
    print(f"  Output dir  : {out}/")
    return 0


def _count_by(rows: list, key: str) -> dict:
    out: dict = {}
    for r in rows:
        v = r.get(key)
        out[str(v)] = out.get(str(v), 0) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
