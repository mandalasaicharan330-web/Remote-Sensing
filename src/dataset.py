"""
dataset.py
==========
Manages curation, storage, and loading of the Phase 2 dataset.
Enforces:
  - Strict preservation of official BigEarthNet v2.0 split assignments (train, val, test).
  - No data leakage: verifies zero overlap between train, validation, and test IDs.
  - No reassignment of test patches: original Phase 1 inspection patches stay strictly in test.
  - Storage in compressed NumPy archive (.npz) with patch IDs, multi-hot targets, and metadata.
"""

import os
import sys
import subprocess
import tarfile
import io
import re
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.label_mapping import (
    BROAD_CLASSES,
    NUM_CLASSES,
    CORINE_TO_BROAD,
    map_corine_labels,
    labels_to_multihot
)
from src.preprocessing import (
    SELECTED_BANDS,
    TARGET_SHAPE,
    load_and_preprocess_patch
)

ZENODO_S2_ARCHIVE_URL = "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1"


class ZstdProcStream(io.RawIOBase):
    """
    Streams and decompresses Zstandard-compressed archive data on the fly.
    """
    def __init__(self, pipe):
        import zstandard as zstd
        self.pipe = pipe
        self.decompressor = zstd.ZstdDecompressor().decompressobj()
        self.buffer = bytearray()
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while len(self.buffer) < len(b):
            chunk = self.pipe.read(65536)
            if not chunk:
                break
            self.bytes_read += len(chunk)
            d = self.decompressor.decompress(chunk)
            if d:
                self.buffer.extend(d)
        if not self.buffer:
            return 0
        n = min(len(b), len(self.buffer))
        b[:n] = self.buffer[:n]
        del self.buffer[:n]
        return n


def curate_genuine_dataset(
    metadata_path: str = "data/metadata.parquet",
    patches_dir: str = "data/curated_patches",
    output_npz: str = "data/curated_dataset_phase2.npz",
    target_counts: Dict[str, int] = {"train": 60, "validation": 20, "test": 20},
    bands: List[str] = SELECTED_BANDS,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Extracts genuine patches directly from the official Zenodo archive to fulfill
    the required target split counts (e.g. 60 train, 20 val, 20 test).

    Strict Guarantees:
      1. Every patch retains its genuine official split from metadata.parquet.
      2. Train, validation, and test sets are strictly disjoint (0 overlap).
      3. Original source files are preserved under patches_dir.
      4. Curated data is packaged into a compressed .npz archive.
    """
    os.makedirs(patches_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_npz)), exist_ok=True)

    # Load metadata lookup
    table = pq.read_table(metadata_path)
    meta_pids = table["patch_id"].to_pylist()
    meta_splits = table["split"].to_pylist()
    meta_labels = table["labels"].to_pylist()
    meta_countries = table["country"].to_pylist()

    pid_to_meta = {}
    for p, s, l, c in zip(meta_pids, meta_splits, meta_labels, meta_countries):
        pid_to_meta[p] = {"split": s, "labels": l, "country": c}

    collected: Dict[str, Dict[str, Set[str]]] = {
        "train": {},
        "validation": {},
        "test": {}
    }

    # First, check if any patches are already present locally in patches_dir
    local_folders = [d for d in os.listdir(patches_dir) if os.path.isdir(os.path.join(patches_dir, d))]
    for pid in local_folders:
        meta = pid_to_meta.get(pid)
        if not meta:
            continue
        sp = meta["split"]
        # Check if all bands exist
        all_exist = all(os.path.exists(os.path.join(patches_dir, pid, f"{pid}_{b}.tif")) for b in bands)
        if all_exist and sp in collected and len(collected[sp]) < target_counts[sp]:
            collected[sp][pid] = set(bands)

    # Check if we need to stream additional patches
    needed = {sp: target_counts[sp] - len(collected[sp]) for sp in target_counts}
    total_needed = sum(needed.values())

    if total_needed > 0:
        if verbose:
            print(f"[dataset] Streaming genuine patches from Zenodo to satisfy splits: {needed}")

        curl_cmd = ["curl.exe" if sys.platform == "win32" else "curl", "-s", "-L", ZENODO_S2_ARCHIVE_URL]
        proc = subprocess.Popen(curl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stream = ZstdProcStream(proc.stdout)

        in_progress_bands: Dict[str, Set[str]] = {}

        try:
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    parts = member.name.split("/")
                    if len(parts) < 4:
                        continue
                    pid = parts[-2]
                    filename = parts[-1]
                    m = re.search(r"_(B[0-9A-Za-z]+)\.tif$", filename)
                    if not m:
                        continue
                    b = m.group(1)
                    if b not in bands:
                        continue

                    meta = pid_to_meta.get(pid)
                    if not meta:
                        continue
                    sp = meta["split"]

                    # Only collect if this split still needs samples
                    if len(collected[sp]) >= target_counts[sp] and pid not in collected[sp]:
                        continue

                    # Extract file
                    p_folder = os.path.join(patches_dir, pid)
                    os.makedirs(p_folder, exist_ok=True)
                    out_tif = os.path.join(p_folder, filename)

                    if not os.path.exists(out_tif):
                        f_in = tar.extractfile(member)
                        if f_in:
                            with open(out_tif, "wb") as f_out:
                                f_out.write(f_in.read())

                    in_progress_bands.setdefault(pid, set()).add(b)
                    if len(in_progress_bands[pid]) == len(bands):
                        collected[sp][pid] = set(bands)
                        rem = sum(max(0, target_counts[s] - len(collected[s])) for s in target_counts)
                        if verbose and (len(collected[sp]) % 10 == 0 or rem == 0):
                            mb = stream.bytes_read / (1024 * 1024)
                            counts_str = ", ".join(f"{s}={len(collected[s])}/{target_counts[s]}" for s in target_counts)
                            print(f"[dataset] Collected [{counts_str}] ({mb:.2f} MB streamed)")
                        if rem == 0:
                            break
        finally:
            proc.kill()

    # --- Preprocess and assemble into .npz ---
    images_list = []
    labels_list = []
    patch_ids_list = []
    splits_list = []
    countries_list = []
    original_labels_list = []

    for sp in ["train", "validation", "test"]:
        for pid in sorted(collected[sp].keys()):
            meta = pid_to_meta[pid]
            # Preprocess tensor (resamples B11/B12, scales reflectance to [0, 1])
            tensor, _ = load_and_preprocess_patch(patches_dir, pid, bands=bands)
            mhot = labels_to_multihot(meta["labels"])

            images_list.append(tensor)
            labels_list.append(mhot)
            patch_ids_list.append(pid)
            splits_list.append(sp)
            countries_list.append(meta["country"])
            original_labels_list.append("; ".join(meta["labels"]))

    X_all = np.stack(images_list, axis=0)  # (N, 6, 120, 120)
    Y_all = np.stack(labels_list, axis=0)  # (N, 7)
    patch_ids_arr = np.array(patch_ids_list)
    splits_arr = np.array(splits_list)
    countries_arr = np.array(countries_list)
    orig_labels_arr = np.array(original_labels_list)

    # Verification checks
    train_ids = set(patch_ids_arr[splits_arr == "train"])
    val_ids = set(patch_ids_arr[splits_arr == "validation"])
    test_ids = set(patch_ids_arr[splits_arr == "test"])

    assert len(patch_ids_arr) == len(set(patch_ids_arr)), "Duplicate patch IDs detected!"
    assert len(train_ids & val_ids) == 0, "Data leakage: train and validation sets overlap!"
    assert len(train_ids & test_ids) == 0, "Data leakage: train and test sets overlap!"
    assert len(val_ids & test_ids) == 0, "Data leakage: validation and test sets overlap!"

    np.savez_compressed(
        output_npz,
        images=X_all,
        labels=Y_all,
        patch_ids=patch_ids_arr,
        splits=splits_arr,
        countries=countries_arr,
        original_labels=orig_labels_arr,
        class_names=np.array(BROAD_CLASSES),
        band_names=np.array(bands)
    )

    if verbose:
        print(f"[dataset] Curated dataset successfully saved to: {output_npz}")
        print(f" - Total patches: {len(X_all)}")
        print(f" - Image array shape: {X_all.shape} ({X_all.dtype})")
        print(f" - Labels array shape: {Y_all.shape} ({Y_all.dtype})")
        print(f" - Split counts: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    return {
        "npz_path": output_npz,
        "total_patches": len(X_all),
        "split_counts": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)},
        "patch_ids": patch_ids_arr,
        "splits": splits_arr,
        "images": X_all,
        "labels": Y_all
    }


def load_curated_dataset(npz_path: str = "data/curated_dataset_phase2.npz") -> Dict[str, any]:
    """
    Loads curated dataset from the compressed .npz archive.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Curated dataset not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    return {
        "images": data["images"],
        "labels": data["labels"],
        "patch_ids": data["patch_ids"],
        "splits": data["splits"],
        "countries": data["countries"],
        "original_labels": data["original_labels"],
        "class_names": data["class_names"],
        "band_names": data["band_names"]
    }


def curate_phase2_1_dataset(
    metadata_path: str = "data/metadata.parquet",
    patches_dir: str = "data/curated_patches",
    output_npz: str = "data/curated_dataset_phase2_1.npz",
    bands: List[str] = SELECTED_BANDS,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Assembles, validates, and packages the corrected Phase 2.1 dataset (500 genuine patches)
    into a compressed .npz archive with zero data leakage and strict split preservation.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_npz)), exist_ok=True)

    # Load metadata lookup
    table = pq.read_table(metadata_path)
    meta_pids = table["patch_id"].to_pylist()
    meta_splits = table["split"].to_pylist()
    meta_labels = table["labels"].to_pylist()
    meta_countries = table["country"].to_pylist()

    pid_to_meta = {
        p: {"split": s, "labels": l, "country": c}
        for p, s, l, c in zip(meta_pids, meta_splits, meta_labels, meta_countries)
    }

    # Identify all patches in patches_dir
    local_folders = sorted([d for d in os.listdir(patches_dir) if os.path.isdir(os.path.join(patches_dir, d))])
    assert len(local_folders) == 500, f"Expected 500 genuine patches in {patches_dir}, found {len(local_folders)}"

    images_list = []
    labels_list = []
    patch_ids_list = []
    splits_list = []
    countries_list = []
    original_labels_list = []

    # Sort deterministically by official split then patch_id
    split_order = {"train": 0, "validation": 1, "test": 2}
    sorted_pids = sorted(local_folders, key=lambda p: (split_order[pid_to_meta[p]["split"]], p))

    for pid in sorted_pids:
        meta = pid_to_meta[pid]
        tensor, _ = load_and_preprocess_patch(patches_dir, pid, bands=bands, scale_reflectance=True)
        mhot = labels_to_multihot(meta["labels"])

        images_list.append(tensor)
        labels_list.append(mhot)
        patch_ids_list.append(pid)
        splits_list.append(meta["split"])
        countries_list.append(meta["country"])
        original_labels_list.append("; ".join(meta["labels"]))

    X_all = np.stack(images_list, axis=0)  # (500, 6, 120, 120)
    Y_all = np.stack(labels_list, axis=0)  # (500, 7)
    patch_ids_arr = np.array(patch_ids_list)
    splits_arr = np.array(splits_list)
    countries_arr = np.array(countries_list)
    orig_labels_arr = np.array(original_labels_list)

    # Verification checks
    train_ids = set(patch_ids_arr[splits_arr == "train"])
    val_ids = set(patch_ids_arr[splits_arr == "validation"])
    test_ids = set(patch_ids_arr[splits_arr == "test"])

    assert len(patch_ids_arr) == 500, f"Expected 500 patches, got {len(patch_ids_arr)}"
    assert len(patch_ids_arr) == len(set(patch_ids_arr)), "Duplicate patch IDs detected!"
    assert len(train_ids) == 300, f"Expected 300 train patches, got {len(train_ids)}"
    assert len(val_ids) == 100, f"Expected 100 val patches, got {len(val_ids)}"
    assert len(test_ids) == 100, f"Expected 100 test patches, got {len(test_ids)}"
    assert len(train_ids & val_ids) == 0, "Data leakage: train and validation sets overlap!"
    assert len(train_ids & test_ids) == 0, "Data leakage: train and test sets overlap!"
    assert len(val_ids & test_ids) == 0, "Data leakage: validation and test sets overlap!"

    # Verify all 5 Phase 1 test patches are in test split
    p1_patches = {
        "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57",
        "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_27_55",
        "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_27_56",
        "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_27_57",
        "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_27_58"
    }
    assert p1_patches.issubset(test_ids), "Phase 1 inspection patches are missing from the test split!"

    # Numerical integrity
    assert not np.isnan(X_all).any(), "NaN values found in preprocessed image cubes!"
    assert not np.isinf(X_all).any(), "Inf values found in preprocessed image cubes!"
    assert (X_all >= 0.0).all() and (X_all <= 1.0).all(), "Values outside [0, 1] range!"

    np.savez_compressed(
        output_npz,
        images=X_all,
        labels=Y_all,
        patch_ids=patch_ids_arr,
        splits=splits_arr,
        countries=countries_arr,
        original_labels=orig_labels_arr,
        class_names=np.array(BROAD_CLASSES),
        band_names=np.array(bands)
    )

    if verbose:
        print(f"[dataset] Corrected Phase 2.1 dataset saved to: {output_npz}")
        print(f" - Image array: {X_all.shape} ({X_all.dtype})")
        print(f" - Labels array: {Y_all.shape} ({Y_all.dtype})")
        print(f" - Splits: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    return {
        "npz_path": output_npz,
        "total_patches": len(X_all),
        "split_counts": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)},
        "patch_ids": patch_ids_arr,
        "splits": splits_arr,
        "images": X_all,
        "labels": Y_all
    }

