"""Patient-grouped, source-and-label-stratified 5-fold splits.

The standard splits for this benchmark are 5 folds assigned at the
**patient** level (no patient appears in more than one fold) and
stratified by ``(dataset, label)`` so each fold has comparable
BUV/WHBUS × benign/malignant composition.

By convention, **fold 5 is the default held-out test set**. Groups
training on this small benchmark should treat folds 1–4 as the
training pool (with their own train/val split inside that, e.g.,
fold 4 as validation) and report numbers on fold 5.

Splits are deterministic given the input manifest order plus the
``random_state`` (default 42). Two builds of the benchmark on the
same source downloads will produce the same bag IDs and the same
fold assignments.
"""
from __future__ import annotations

from typing import Dict, List, Any

import numpy as np


DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_FOLD = 5  # by convention, fold 5 is the standard held-out test


def assign_folds(
    bag_rows: List[Dict[str, Any]],
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> List[int]:
    """Assign each bag a fold number in 1..n_splits.

    Grouping: patient_id (so all bags from one patient land in the same
    fold). Stratification: f"{dataset}|{label}" (so each fold has
    proportional BUV/WHBUS × benign/malignant counts).

    Returns a list of fold integers (1-indexed) parallel to ``bag_rows``.
    """
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as e:
        raise ImportError(
            "splits.assign_folds requires scikit-learn; install with "
            "`pip install scikit-learn`."
        ) from e

    n = len(bag_rows)
    if n == 0:
        return []

    groups = np.array([r["patient_id"] for r in bag_rows])
    stratum = np.array([f"{r['dataset']}|{r['label']}" for r in bag_rows])
    X = np.zeros((n, 1))  # placeholder, not used by SGKFold

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    fold_of = np.zeros(n, dtype=int)
    for fold_idx, (_train_idx, test_idx) in enumerate(
        sgkf.split(X, y=stratum, groups=groups), start=1
    ):
        fold_of[test_idx] = fold_idx

    # Sanity: every bag has a fold; no patient straddles folds.
    assert (fold_of > 0).all(), "some bag did not get a fold assigned"
    patient_to_folds: Dict[str, set] = {}
    for r, f in zip(bag_rows, fold_of):
        patient_to_folds.setdefault(r["patient_id"], set()).add(int(f))
    bad = {p: fs for p, fs in patient_to_folds.items() if len(fs) > 1}
    assert not bad, f"patient(s) straddle folds: {bad}"

    return fold_of.tolist()


def summarize_folds(
    bag_rows: List[Dict[str, Any]],
    folds: List[int],
    n_splits: int = DEFAULT_N_SPLITS,
) -> str:
    """Return a human-readable per-fold composition summary."""
    lines = [
        f"Fold | n_bags | BUV | WHBUS | benign | malignant | n_patients",
        f"-----|--------|-----|-------|--------|-----------|-----------",
    ]
    for f in range(1, n_splits + 1):
        rows = [r for r, fi in zip(bag_rows, folds) if fi == f]
        n_buv = sum(1 for r in rows if r["dataset"] == "buv")
        n_whbus = sum(1 for r in rows if r["dataset"] == "whbus")
        n_ben = sum(1 for r in rows if r["label"] == 0)
        n_mal = sum(1 for r in rows if r["label"] == 1)
        n_pat = len({r["patient_id"] for r in rows})
        marker = "  (default test)" if f == DEFAULT_TEST_FOLD else ""
        lines.append(
            f" {f}   | {len(rows):>5}  | {n_buv:>3} | {n_whbus:>5} | "
            f"{n_ben:>6} | {n_mal:>9} | {n_pat:>10}{marker}"
        )
    return "\n".join(lines)
