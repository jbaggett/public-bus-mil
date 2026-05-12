"""Example: load the benchmark as a PyTorch MIL dataset.

Each item is (bag_tensor, label, bag_id) where bag_tensor has shape
(n_frames, 3, 256, 256). Uses 256+letterbox preprocessing with gray-128
fill and ImageNet normalization, matching the internal CADBUSI pipeline.
Adjust to your model's input convention.

Usage:
    python examples/load_torch_dataset.py /path/to/built_benchmark/

    or in code:

    from public_bus_mil.examples.load_torch_dataset import PublicBusMilDataset
    ds = PublicBusMilDataset("./external_test_v1/")
    print(f"N bags: {len(ds)}")
    bag, label, bag_id = ds[0]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    print("This example requires torch. Install with: pip install torch", file=sys.stderr)
    raise


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox_256(img: Image.Image, fill: int = 128) -> np.ndarray:
    """Aspect-preserving resize to 256 with gray-fill padding."""
    img = img.convert("RGB")
    w, h = img.size
    scale = 256 / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    out = Image.new("RGB", (256, 256), (fill, fill, fill))
    out.paste(img, ((256 - new_w) // 2, (256 - new_h) // 2))
    return np.asarray(out, dtype=np.float32) / 255.0


def normalize(arr: np.ndarray) -> np.ndarray:
    """ImageNet normalization. Input shape (H,W,3), output same shape."""
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


class PublicBusMilDataset(Dataset):
    """Reads a built benchmark folder and yields (bag, label, bag_id) tuples."""

    def __init__(self, root: str | Path, dataset_filter: str | None = None):
        """
        Parameters
        ----------
        root : path to benchmark folder (with manifest.csv, frames.csv, images/)
        dataset_filter : "buv", "whbus", or None (no filter)
        """
        self.root = Path(root)
        self.manifest = pd.read_csv(self.root / "manifest.csv")
        if dataset_filter:
            self.manifest = self.manifest[self.manifest["dataset"] == dataset_filter].reset_index(drop=True)
        frames = pd.read_csv(self.root / "frames.csv")
        self._frames_by_bag = {bag_id: g for bag_id, g in frames.groupby("bag_id")}

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.manifest.iloc[idx]
        bag_id = row["bag_id"]
        label = int(row["label"])
        frame_rows = self._frames_by_bag[bag_id].sort_values("frame_idx_in_bag")
        frames = []
        for _, fr in frame_rows.iterrows():
            img = Image.open(self.root / fr["src_path"])
            arr = normalize(letterbox_256(img))
            # to (C, H, W)
            frames.append(np.transpose(arr, (2, 0, 1)))
        bag = torch.from_numpy(np.stack(frames, axis=0)).float()
        return bag, label, bag_id


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python examples/load_torch_dataset.py <benchmark_root>")
        sys.exit(1)
    ds = PublicBusMilDataset(sys.argv[1])
    print(f"N bags: {len(ds)}")
    print(f"First bag:")
    bag, label, bag_id = ds[0]
    print(f"  bag_id : {bag_id}")
    print(f"  label  : {label}")
    print(f"  shape  : {tuple(bag.shape)}  (n_frames, 3, 256, 256)")
    print(f"  dtype  : {bag.dtype}")
