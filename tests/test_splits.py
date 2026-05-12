"""Sanity tests for public_bus_mil.splits.

Verifies:
  - all bags receive a fold in 1..5
  - no patient straddles two folds
  - splits are deterministic given the random_state
  - per-stratum proportions are roughly balanced across folds
"""
from __future__ import annotations

import pytest

pytest.importorskip("sklearn")

from public_bus_mil.splits import assign_folds, DEFAULT_TEST_FOLD


def _synth_bags(n_buv_pat=120, n_whbus_pat=100, whbus_clips_per_pat=2,
                buv_mal_frac=0.55, whbus_mal_frac=0.45):
    """Build a synthetic bag list shaped like the real benchmark:
    BUV is 1 bag per patient; WHBUS is multi-bag per patient."""
    rows = []
    bag_id = 1
    for i in range(1, n_buv_pat + 1):
        label = 1 if i / n_buv_pat <= buv_mal_frac else 0
        rows.append({
            "bag_id": f"buv_{bag_id:04d}", "dataset": "buv",
            "label": label, "patient_id": f"buv_p{i:03d}",
        })
        bag_id += 1
    for i in range(1, n_whbus_pat + 1):
        label = 1 if i / n_whbus_pat <= whbus_mal_frac else 0
        for c in range(whbus_clips_per_pat):
            rows.append({
                "bag_id": f"whbus_{bag_id:04d}", "dataset": "whbus",
                "label": label, "patient_id": f"whbus_pat{i:03d}",
            })
            bag_id += 1
    return rows


def test_all_bags_assigned():
    rows = _synth_bags()
    folds = assign_folds(rows)
    assert len(folds) == len(rows)
    assert all(1 <= f <= 5 for f in folds)


def test_no_patient_straddles_folds():
    rows = _synth_bags()
    folds = assign_folds(rows)
    pat_folds = {}
    for r, f in zip(rows, folds):
        pat_folds.setdefault(r["patient_id"], set()).add(f)
    leaks = {p: fs for p, fs in pat_folds.items() if len(fs) > 1}
    assert not leaks, f"{len(leaks)} patients straddle folds"


def test_deterministic_under_seed():
    rows = _synth_bags()
    a = assign_folds(rows, random_state=42)
    b = assign_folds(rows, random_state=42)
    assert a == b


def test_changing_seed_changes_assignment():
    rows = _synth_bags()
    a = assign_folds(rows, random_state=42)
    b = assign_folds(rows, random_state=7)
    assert a != b, "different random_state should yield different folds"


def test_stratum_balance():
    """Each fold should hold roughly 1/5 of each (dataset,label) stratum."""
    rows = _synth_bags()
    folds = assign_folds(rows)
    from collections import Counter
    strata = Counter(f"{r['dataset']}|{r['label']}" for r in rows)
    for stratum, total in strata.items():
        per_fold = Counter()
        for r, f in zip(rows, folds):
            if f"{r['dataset']}|{r['label']}" == stratum:
                per_fold[f] += 1
        # No fold should have more than 2x the proportional share for any stratum.
        expected = total / 5
        worst = max(per_fold.values())
        assert worst <= 2 * expected, (
            f"stratum {stratum}: worst fold has {worst} (>2× expected {expected:.1f})"
        )


def test_default_test_fold_is_5():
    assert DEFAULT_TEST_FOLD == 5
