"""Download MSD Task01 and write BraTS-style patient folders.

Does not touch data/raw/brats. Volumes are gitignored. The patient-id split
JSON under data/splits/ is small and is meant to be committed.

Official BraTS / Synapse is not contacted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.data.msd import (
    MSD_TASK01_MONAI_MD5,
    MSD_TASK01_URL,
    assert_task01_labels,
    load_dataset_json,
    modality_index,
    write_brats_style_case,
)
from neurovisionx.data.splits import save_split, split_patients


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Archive already present: {destination} ({destination.stat().st_size} bytes)")
        return
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "8",
        "--retry-delay",
        "5",
        "-C",
        "-",
        "-o",
        str(partial),
        url,
    ]
    print("Downloading", url)
    subprocess.check_call(cmd)
    partial.replace(destination)


def extract_archive(archive: Path, extract_root: Path) -> Path:
    marker = extract_root / ".extract_complete"
    if marker.is_file():
        return extract_root
    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} (this is a multi-gigabyte tar)")
    if tarfile.is_tarfile(archive):
        subprocess.check_call(["tar", "-xf", str(archive), "-C", str(extract_root)])
    else:
        raise SystemExit(f"{archive} is not a tar file")
    marker.write_text("ok\n", encoding="utf-8")
    return extract_root


def find_dataset_json(extract_root: Path) -> Path:
    matches = sorted(extract_root.rglob("dataset.json"))
    if not matches:
        raise SystemExit(f"No dataset.json under {extract_root}")
    # Prefer the Task01 file if several exist.
    for path in matches:
        try:
            payload = load_dataset_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        name = str(payload.get("name", "")).lower()
        if "brat" in name or payload.get("numTraining"):
            return path
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert MSD Task01_BrainTumour.")
    parser.add_argument("--url", default=MSD_TASK01_URL)
    parser.add_argument("--dest", type=Path, default=ROOT / "data" / "raw" / "msd_task01")
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Convert at most N labeled cases (debug).")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--split-out", type=Path, default=ROOT / "data" / "splits" / "msd_task01_patients.json")
    parser.add_argument("--summary-out", type=Path, default=ROOT / "docs" / "msd_task01_conversion_summary.json")
    args = parser.parse_args()

    dest = args.dest
    archive = args.archive or (dest / "archives" / "Task01_BrainTumour.tar")
    if not args.skip_download:
        download_archive(args.url, archive)
    if not archive.is_file():
        raise SystemExit(f"Archive missing: {archive}")

    print("Computing MD5 (the S3 object is multipart, so the ETag is not the MD5)...")
    digest = _md5(archive)
    print(f"md5={digest}")
    if digest != MSD_TASK01_MONAI_MD5:
        print(
            "WARNING: MD5 does not match the MONAI 1.3 published digest "
            f"{MSD_TASK01_MONAI_MD5}. Conversion still checks dataset.json."
        )

    extract_root = dest / "extracted"
    extract_archive(archive, extract_root)
    dataset_path = find_dataset_json(extract_root)
    dataset = load_dataset_json(dataset_path)
    assert_task01_labels(dataset.get("labels") or {})
    index = modality_index(dataset.get("modality") or {})
    print(f"dataset.json={dataset_path}")
    print(f"modality index (repo key → channel)={index}")
    print(f"labels={dataset.get('labels')}")

    base = dataset_path.parent
    training = dataset.get("training") or []
    if args.limit:
        training = training[: args.limit]
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = []
    for row in training:
        image_rel = row["image"] if isinstance(row, dict) else row[0]
        label_rel = row["label"] if isinstance(row, dict) else row[1]
        jobs.append((base / image_rel, base / label_rel))

    results = []
    failures = []
    workers = max(1, int(args.workers))
    print(f"Converting {len(jobs)} labeled cases with {workers} workers")
    if workers == 1:
        iterator = (write_brats_style_case(image, label, dest, index) for image, label in jobs)
        for i, outcome in enumerate(iterator, start=1):
            results.append(outcome)
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(write_brats_style_case, image, label, dest, index): image
                for image, label in jobs
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                image = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001 — record and continue
                    failures.append({"image": str(image), "error": str(exc)})
                    print(f"FAILED {image}: {exc}")
                if done % 25 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)}")

    ok_ids = sorted({row["patient_id"] for row in results if "patient_id" in row})
    if len(ok_ids) < 3:
        raise SystemExit(f"Only {len(ok_ids)} cases converted. Refusing to write a split.")
    split = split_patients(ok_ids, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20, seed=2026)
    save_split(split, args.split_out)
    meta = {
        "source_url": args.url,
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "md5": digest,
        "monai_published_md5": MSD_TASK01_MONAI_MD5,
        "md5_matches_monai_1_3": digest == MSD_TASK01_MONAI_MD5,
        "dataset_json": str(dataset_path),
        "dataset_name": dataset.get("name"),
        "licence_field": dataset.get("licence") or dataset.get("license"),
        "reference": dataset.get("reference"),
        "modality_raw": dataset.get("modality"),
        "modality_index_repo_key_to_channel": index,
        "labels_raw": dataset.get("labels"),
        "label_remap_msd_to_brats": {"0": 0, "1": 2, "2": 1, "3": 3},
        "n_labeled_in_json": len(dataset.get("training") or []),
        "n_converted": len(ok_ids),
        "n_failed": len(failures),
        "failures": failures,
        "n_train": len(split.train),
        "n_val": len(split.val),
        "n_test": len(split.test),
        "seed": split.seed,
        "split": str(args.split_out),
        "limit": args.limit,
        "clinical_disclaimer": "Research prototype. Not a clinically validated diagnostic tool.",
        "not_official_brats": True,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    sidecar = args.split_out.with_name(args.split_out.stem + "_meta.json")
    sidecar.write_text(json.dumps({k: meta[k] for k in (
        "md5", "md5_matches_monai_1_3", "n_converted", "n_failed", "n_train", "n_val", "n_test",
        "seed", "label_remap_msd_to_brats", "modality_index_repo_key_to_channel",
        "licence_field", "not_official_brats", "clinical_disclaimer", "limit",
    )}, indent=2), encoding="utf-8")
    print(
        f"Converted {len(ok_ids)} cases, failed {len(failures)}. "
        f"Split train/val/test {len(split.train)}/{len(split.val)}/{len(split.test)} → {args.split_out}"
    )
    # Drop the bulky extracted 4D copies once patient folders exist. The tar remains.
    extracted_images = list(extract_root.rglob("imagesTr"))
    if ok_ids and not failures and not args.limit:
        for folder in extracted_images:
            print(f"Removing extracted images to save disk: {folder}")
            shutil.rmtree(folder, ignore_errors=True)
        for folder in extract_root.rglob("labelsTr"):
            shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    main()
