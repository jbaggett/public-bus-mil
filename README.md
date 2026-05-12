# public-bus-mil

A reproducible **exam-level Multiple-Instance Learning (MIL) benchmark**
for breast ultrasound, built from two publicly-available breast-ultrasound
video datasets:

- **MICCAI-BUV** (Lin et al., MICCAI 2022) — 186 videos with per-frame
  lesion bounding-box annotations.
- **WHBUS** (Zhang et al., ISBI 2024) — 184 clips from 131 patients,
  clip-level malignancy labels.

The benchmark constructs uniform exam-level bags of frames using a
deterministic keyframe algorithm (Laplacian-variance + minimum-interval
temporal spacing), trims dark scanner margins, and emits a manifest
compatible with most attention-based MIL pipelines.

Designed for use as an **external validation cohort** for breast-ultrasound
classification and saliency models trained on private clinical datasets.

---

## Where to get the source datasets

This package converts the two source datasets into a standardized
exam-level MIL form; **you must first download the source data from the
original publishers**. Both are publicly available for research use:

| Dataset | Paper | Source repository | Direct download |
|---|---|---|---|
| **MICCAI-BUV** | Lin et al., MICCAI 2022 ([arXiv](https://arxiv.org/abs/2207.00141)) | <https://github.com/jhl-Det/CVA-Net> | 7-Zip archive linked from the CVA-Net README (~4 GB) |
| **WHBUS** | Zhang et al., ISBI 2024 ([DOI](https://doi.org/10.1109/ISBI56570.2024.10635722)) | <https://github.com/imzhangyd/SAG-Net> | [Google Drive folder](https://drive.google.com/drive/folders/13oBWsRzKooeZBD6eRMHs6Mpf7ZeiPQHo) (linked from the SAG-Net README) |

See `DATA_SOURCES.md` for the full unpacking layout, expected directory
structure, and licensing notes from each source repository.

---

## Why this exists

Most public BUS datasets (BUSI, UDIAT, BUS-UCLM, BUS-BRA) are image-level —
one labeled image per case. Models trained on exam-level MIL bags (multiple
frames per patient breast) have no matched-shape public benchmark for
external evaluation.

This package converts two public **video** datasets into exam-level bags
following a documented protocol, so that:

1. The construction is fully reproducible from the original public
   downloads + this code,
2. Different research groups can report results on the same bags,
3. Both classification AUC and (on BUV) frame-level saliency pointing
   accuracy can be evaluated against the same model.

---

## Quick start

```bash
# 1. Install (no GPU required for the benchmark itself)
pip install -e .

# 2. Download the source datasets (see DATA_SOURCES.md):
#    - MICCAI-BUV from https://github.com/jhl-Det/CVA-Net
#    - WHBUS from https://github.com/imzhangyd/SAG-Net

# 3. Build the benchmark
python build_bags.py \
    --dataset both \
    --buv-extracted /path/to/BUV_Extracted \
    --buv-annotations-train /path/to/imagenet_vid_train_15frames.json \
    --buv-annotations-val   /path/to/imagenet_vid_val.json \
    --whbus /path/to/WHBUS/buvimgs \
    --out ./external_test_v1/

# Output:
#   external_test_v1/
#   ├── manifest.csv       # bag-level
#   ├── frames.csv         # frame-level
#   ├── images/            # native-resolution PNGs after dark-margin trim
#   ├── README.md          # build provenance
#   └── BUILD_LOG.json     # exact parameters + counts + file hashes
```

## What the manifest looks like

`manifest.csv` (bag-level):

| bag_id | dataset | source_id | label | n_frames | patient_id | has_bboxes |
|---|---|---|---|---|---|---|
| buv_0001 | buv | 263b86b85a58f270 | 0 | 8 | buv_p001 | True |
| buv_0002 | buv | xf7f83b2a576016b | 1 | 8 | buv_p002 | True |
| whbus_0001 | whbus | benign_9161001_10 | 0 | 8 | whbus_9161001 | False |
| ... | | | | | | |

`frames.csv` (frame-level):

| bag_id | frame_idx_in_bag | src_path | original_video_frame_idx | has_bbox | bbox_x | bbox_y | bbox_w | bbox_h |
|---|---|---|---|---|---|---|---|---|
| buv_0001 | 0 | images/buv_0001_000.png | 0 | True | 52 | 90 | 124 | 133 |
| ... | | | | | | | | |

## Loading into PyTorch

See `examples/load_torch_dataset.py` for a 30-line PyTorch dataset that
reads the manifest and yields `(bag_tensor, label)` pairs.

---

## What gets evaluated

| Metric | Available on | Notes |
|---|---|---|
| Bag-level AUROC, AUPRC | All bags | Standard classification |
| Frame-level pointing accuracy | BUV bags only (~1,314 frames) | Saliency-peak inside / within-Nmm of bbox |
| Frame-level Hit@2mm / Hit@4mm | BUV bags only | Needs pixel-spacing — not provided per-frame; pixel-based variants are reported instead |

WHBUS provides classification labels only; pointing accuracy is not
computable there because the source dataset has no bounding boxes.

---

## Reproducibility guarantees

- Every PNG in the output is bit-identical to a deterministic function of
  the inputs (no randomness in the build).
- `BUILD_LOG.json` records the exact parameters (K, min_interval,
  trim threshold) and SHA-256 hashes of every output frame.
- A consumer who downloads the source datasets and runs the same command
  will get the identical manifest.

---

## Citation

If you use this benchmark, please cite the two underlying dataset papers:

```bibtex
@inproceedings{lin2022buv,
  title     = {A New Dataset and A Baseline Model for Breast Lesion Detection in Ultrasound Videos},
  author    = {Lin, Zhi and Lin, Junhao and Zhu, Lei and Fu, Huazhu and Qin, Jing and Wang, Liansheng},
  booktitle = {MICCAI},
  year      = {2022},
  pages     = {614--623},
  url       = {https://arxiv.org/abs/2207.00141}
}

@inproceedings{zhang2024sagnet,
  title     = {Using Segment-Level Attention to Guide Breast Ultrasound Video Classification},
  author    = {Zhang, Yudong and Kong, Deguang and Li, Juanjuan and Yang, Tao and Yao, Feng and Yang, Ge},
  booktitle = {2024 IEEE International Symposium on Biomedical Imaging (ISBI)},
  pages     = {1--5},
  year      = {2024},
  doi       = {10.1109/ISBI56570.2024.10635722}
}
```

(Optional citation for this benchmark itself once the companion publication
is available.)

## License

MIT — see `LICENSE`. The underlying source datasets are governed by their
own licenses; see `DATA_SOURCES.md`.
