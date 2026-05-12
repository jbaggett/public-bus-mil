# Dataset construction spec

This document defines the exact protocol used by `build_bags.py` to
convert the two source video datasets into MIL bags. Anyone reproducing
the benchmark from scratch should be able to read this document, run the
script, and obtain bit-identical output.

---

## 1. Sources

| Dataset | Source paper | Original distribution |
|---|---|---|
| MICCAI-BUV | Lin et al., MICCAI 2022 ([arXiv 2207.00141](https://arxiv.org/abs/2207.00141)) | https://github.com/jhl-Det/CVA-Net |
| WHBUS | Zhang et al., ISBI 2024 (doi 10.1109/ISBI56570.2024.10635722) | https://github.com/imzhangyd/SAG-Net |

For BUV we additionally consume the per-frame bounding-box annotations
distributed with the source archive as `imagenet_vid_train_15frames.json`
and `imagenet_vid_val.json`. These are COCO-style annotations
(`{images, annotations, categories, videos}`) keyed by video_id and
frame_idx.

## 2. Errata applied

Per `BUV_Extracted/errata.txt`:

- **Drop**: `rawframes/benign/x66ef02e7f1b9a0ef` (identical to
  `rawframes/malignant/x3b88488853e8b7d1`; the benign copy was mislabeled).
- **Relabel benign → malignant** for these four videos (same patient,
  multiple acquisitions):
  - `benign/x63c9ba1377f35bf6`
  - `benign/x5a1c46ec6377e946`
  - `malignant/2390fbea047347b` (already malignant; relabel is no-op)
  - `malignant/7a39ab5d4970bf89` (already malignant; relabel is no-op)
- **Merge** the four-video group above into a single patient bag
  (`patient_id = "buv_errata_patient_1"`).

## 3. Bag unit

| Dataset | Bag unit | Bags per dataset |
|---|---|---|
| BUV | One bag = one video (with errata-merge exception) | ~183 after errata |
| WHBUS | One bag = one clip | 184 |

Rationale: both source-paper authors report metrics at video / clip
granularity, never patient-aggregated. WHBUS does not document whether
multi-clip patients = same breast vs different breasts, so clip-level is
the only verifiable unit. `patient_id` is recorded as audit metadata
(`whbus_<patient_id>` parsed from clip names like
`benign_9161001_10` → patient `9161001`) for downstream sensitivity
analyses but is not used during bag construction.

## 4. Frame selection algorithm (BUV keyframe algorithm)

Both datasets use the same per-clip frame-selection algorithm, applied
to **every** clip prior to dark-margin trim and bag assembly:

1. **Sharpness score** — for each frame, compute the variance of the
   Laplacian after converting to grayscale. This is the standard
   blur-detection metric (high variance = sharp, low variance = blurry).
2. **Quality threshold** — drop any frame whose sharpness score is below
   `0.7 × max(sharpness over the clip)` (i.e., relative threshold). This
   removes severely blurry frames while keeping the threshold scale-free
   across clips with different absolute brightness.
3. **Temporal spacing** — when picking frames, enforce a minimum
   `min_interval` frame-index gap between selected frames (no two
   selected frames within `min_interval` of each other in the original
   video).
4. **Top-K** — among frames that pass (2) and (3), select the top
   `frames_per_video` by sharpness score.

**Per-dataset parameters** (locked):

| Dataset | `frames_per_video` (K) | `min_interval` | `quality_threshold` |
|---|---|---|---|
| BUV | 8 | 10 | 0.7 (relative) |
| WHBUS | 8 | 15 | 0.7 (relative) |

**BUV optimization.** For BUV we do not re-run the algorithm from the
source 4 GB 7z archive — the existing `BUV_Extracted/manifest.json`
(distributed with the BUV repo) was produced by the same algorithm and
is bit-identical to what we would get. We consume it directly.

**Where the algorithm came from.** The same algorithm is documented in
`Public_BUS_Dataset_Manager_Spec.md` (internal) and was used to produce
the `BUV_Extracted` manifest distributed with the BUV source.

## 5. Dark-margin trim

Some WHBUS frames contain a thin black border (up to ~43 pixels of pure
black at the bottom in the worst case observed). Before consumer-side
preprocessing, every frame is trimmed:

1. Convert to grayscale.
2. Compute mean intensity per row and per column.
3. Find the first row / column from each edge with mean intensity above
   `THR = 8` (essentially "any non-black pixel").
4. Crop to the rectangle bounded by those edges.

On BUV frames, all edges already exceed the threshold, so the trim is a
no-op.

## 6. Output preprocessing

The benchmark emits frames at **their post-trim native resolution** as
PNG. Consumer-side preprocessing (resize, letterbox, normalization) is
intentionally left to the user, so the benchmark works with any model's
input convention.

A reference preprocessing wrapper is provided in
`examples/load_torch_dataset.py` that mimics our internal pipeline (256+
letterbox with gray-128 fill + ImageNet normalization).

## 7. Bbox handling (BUV only)

The BUV annotations contain per-frame bounding boxes for the lesion.
**Verified 2026-05-11**: 100 % of the frames selected by the keyframe
algorithm have corresponding bbox annotations in
`imagenet_vid_train_15frames.json` and `imagenet_vid_val.json`. The
benchmark joins via (video_id, frame_idx) and emits the bbox alongside
each frame row in `frames.csv` as `bbox_x, bbox_y, bbox_w, bbox_h` in
native-resolution pixel coordinates **after dark-margin trim**.

When multiple bbox annotations exist for the same frame (multi-lesion
case), we emit the largest by area and set `n_bboxes > 1` so consumers
can filter.

WHBUS frames have `has_bbox = False` and empty bbox columns.

## 8. Counts (locked expectations)

| | BUV | WHBUS | Combined |
|---|---|---|---|
| Bags | ~183 | 184 | ~367 |
| Benign bags | ~74 | 101 | ~175 |
| Malignant bags | ~109 | 83 | ~192 |
| Frames total | ~1,314 | 1,472 | ~2,786 |
| Frames with bbox | ~1,314 | 0 | ~1,314 |
| Median bag size | 7 | 8 | 8 |

(Final numbers will be written to `BUILD_LOG.json` after the script runs.)

## 9. Determinism and provenance

The build is deterministic. `BUILD_LOG.json` records:
- Exact CLI arguments
- Algorithm parameters (per §4–§5)
- SHA-256 hash of every output PNG
- SHA-256 hash of every input source file consumed
- Build timestamp + tool version

A consumer who runs the same command against the same source archives
must get bit-identical output.

## 10. Versioning

The benchmark is versioned. Version 1 (`v1`) corresponds to:
- BUV `Miccai 2022 BUV Dataset.7z` SHA-256 = (to be recorded on first build)
- WHBUS Google Drive snapshot accessed 2026-05-11
- Code release tag `v1.0.0`

Subsequent versions (`v2`, `v3`, …) will be tagged in this repo and
should not silently overwrite v1 output.
