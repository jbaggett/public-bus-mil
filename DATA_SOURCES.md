# Source dataset acquisition

## MICCAI-BUV

- **Paper**: Lin et al., "A New Dataset and A Baseline Model for Breast Lesion Detection in Ultrasound Videos." MICCAI 2022.
- **arXiv**: https://arxiv.org/abs/2207.00141
- **Source code + dataset**: https://github.com/jhl-Det/CVA-Net
- **License**: see the CVA-Net repository (research use; check before
  redistribution).

The dataset is distributed as a 7-Zip archive (`Miccai 2022 BUV Dataset.7z`,
~4 GB) containing:
- `rawframes/{benign,malignant}/<video_hash>/<frame_idx>.png` — full-resolution frames
- `imagenet_vid_train_15frames.json` — per-frame bbox annotations for training videos
- `imagenet_vid_val.json` — per-frame bbox annotations for validation videos

For this benchmark, we consume the *already-extracted-and-quality-filtered*
form (`BUV_Extracted/`) that ships with the source distribution. If you only
have the 7z archive, extract `imagenet_vid_*.json` (small) and follow the
CVA-Net repo's frame-extraction instructions to produce `BUV_Extracted/`.

## WHBUS

- **Paper**: Zhang et al., "Using Segment-Level Attention to Guide Breast Ultrasound Video Classification." ISBI 2024.
- **DOI**: https://doi.org/10.1109/ISBI56570.2024.10635722
- **GitHub**: https://github.com/imzhangyd/SAG-Net
- **Dataset download**: https://drive.google.com/drive/folders/13oBWsRzKooeZBD6eRMHs6Mpf7ZeiPQHo
- **License**: not explicitly stated in the SAG-Net README — treat as
  research-use; confirm with the authors before any redistribution.

The dataset extracts to a `WHBUS/buvimgs/{benign,malignant}/<clip_folder>/`
layout with `<frame_idx>.png` files inside each clip folder. Clip folder
names encode `{label}_{patient_id}_{clip_index}` (e.g., `benign_9161001_10`).

## Layout expected by `build_bags.py`

```
<some_root>/
├── BUV_Extracted/                 # Used if --dataset includes buv
│   ├── manifest.json
│   ├── errata.txt
│   ├── benign/<video_hash>/<frame_idx>.png
│   └── malignant/<video_hash>/<frame_idx>.png
├── imagenet_vid_train_15frames.json   # BUV annotation file (passed via --buv-annotations-train)
├── imagenet_vid_val.json              # BUV annotation file (passed via --buv-annotations-val)
└── WHBUS/                         # Used if --dataset includes whbus
    └── buvimgs/
        ├── benign/<clip_folder>/<frame_idx>.png
        └── malignant/<clip_folder>/<frame_idx>.png
```
