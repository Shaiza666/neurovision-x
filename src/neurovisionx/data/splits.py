"""Patient-level train/val/test splits.

Slices from the same patient never appear in more than one split. This is the
primary leakage control for NeuroVision-X.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PatientSplit:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    seed: int

    def all_ids(self) -> tuple[str, ...]:
        return self.train + self.val + self.test


def split_patients(
    patient_ids: list[str] | tuple[str, ...],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 2026,
) -> PatientSplit:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    unique = sorted(set(patient_ids))
    if len(unique) != len(patient_ids):
        raise ValueError("Duplicate patient ids are not allowed")
    if len(unique) < 3:
        raise ValueError("Need at least three patients to form train/val/test")

    rng = random.Random(seed)
    shuffled = unique[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(n_train, n - 2)
    n_val = min(n_val, n - n_train - 1)
    n_train = max(n_train, 1)
    n_val = max(n_val, 1)
    train = tuple(shuffled[:n_train])
    val = tuple(shuffled[n_train : n_train + n_val])
    test = tuple(shuffled[n_train + n_val :])
    if not test:
        raise ValueError("Test split is empty; add more patients or reduce train/val ratios")
    split = PatientSplit(train=train, val=val, test=test, seed=seed)
    assert_no_leakage(split)
    return split


def assert_no_leakage(split: PatientSplit) -> None:
    groups = (set(split.train), set(split.val), set(split.test))
    if any(len(group) != len(ids) for group, ids in zip(groups, (split.train, split.val, split.test))):
        raise ValueError("Duplicate ids inside a split")
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Patient leakage across train/val/test")


def save_split(split: PatientSplit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(split), indent=2), encoding="utf-8")


def make_split(
    train: list[str] | tuple[str, ...],
    val: list[str] | tuple[str, ...],
    test: list[str] | tuple[str, ...],
    seed: int,
) -> PatientSplit:
    split = PatientSplit(train=tuple(train), val=tuple(val), test=tuple(test), seed=seed)
    if not split.train or not split.val or not split.test:
        raise ValueError("train, val, and test must each contain at least one patient")
    assert_no_leakage(split)
    return split


def split_from_cases(
    cases: list,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 2026,
    honor_official_validation: bool = True,
) -> PatientSplit:
    """Split complete labeled cases. Official val/test subfolders are held out when present."""
    complete = [case for case in cases if getattr(case, "is_complete", lambda: False)()]
    if not complete:
        raise ValueError("No complete labeled patients to split")
    ids = [case.patient_id for case in complete]
    if not honor_official_validation:
        return split_patients(ids, train_ratio, val_ratio, test_ratio, seed)

    val_names = {"validation", "val", "valid"}
    test_names = {"test", "testing"}
    forced_val = [
        case.patient_id
        for case in complete
        if (getattr(case, "source_subdir", None) or "").lower() in val_names
    ]
    forced_test = [
        case.patient_id
        for case in complete
        if (getattr(case, "source_subdir", None) or "").lower() in test_names
    ]
    if not forced_val and not forced_test:
        return split_patients(ids, train_ratio, val_ratio, test_ratio, seed)

    pool = [patient_id for patient_id in ids if patient_id not in set(forced_val) | set(forced_test)]
    rng = random.Random(seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    need_val = not forced_val
    need_test = not forced_test
    n_val = max(1, int(round(len(shuffled) * val_ratio))) if need_val else 0
    n_test = max(1, int(round(len(shuffled) * test_ratio))) if need_test else 0
    if len(shuffled) - n_val - n_test < 1:
        return split_patients(ids, train_ratio, val_ratio, test_ratio, seed)
    extra_val = shuffled[:n_val]
    extra_test = shuffled[n_val : n_val + n_test]
    train = shuffled[n_val + n_test :]
    return make_split(train, list(forced_val) + extra_val, list(forced_test) + extra_test, seed)


def load_split(path: Path) -> PatientSplit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = PatientSplit(
        train=tuple(payload["train"]),
        val=tuple(payload["val"]),
        test=tuple(payload["test"]),
        seed=int(payload["seed"]),
    )
    assert_no_leakage(split)
    return split
