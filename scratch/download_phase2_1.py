import os
import sys
import subprocess
import tarfile
import io
import re
from typing import Dict, List, Set
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, r"c:\Users\manda\Downloads\Remote Sensing")
from src.label_mapping import (
    BROAD_CLASSES,
    map_corine_labels,
    labels_to_multihot
)
from src.preprocessing import SELECTED_BANDS, load_and_preprocess_patch

ZENODO_S2_ARCHIVE_URL = "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1"


class ZstdProcStream(io.RawIOBase):
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


def get_phase2_1_selected_patches(
    metadata_path: str = "data/metadata.parquet",
    patches_dir: str = "data/curated_patches",
    seed: int = 42
) -> Dict[str, List[str]]:
    table = pq.read_table(metadata_path)
    pids = table["patch_id"].to_pylist()
    splits = table["split"].to_pylist()
    labels = table["labels"].to_pylist()

    t33 = sorted(
        [(p, s, map_corine_labels(l), l) for p, s, l in zip(pids, splits, labels) if "_T33UUP_" in p],
        key=lambda x: x[0]
    )

    local_pids = set(os.listdir(patches_dir)) if os.path.exists(patches_dir) else set()

    train_pool = [x for x in t33 if x[1] == "train"]
    val_pool = [x for x in t33 if x[1] == "validation"]
    test_pool = [x for x in t33 if x[1] == "test"]

    rng = np.random.RandomState(seed)

    def select(pool, target_n):
        chosen = []
        # Priority 1: Already downloaded local patches
        for x in pool:
            if x[0] in local_pids:
                chosen.append(x)
        # Priority 2: Rare classes (Wetland, Shrub)
        for x in pool:
            if ("Wetland" in x[2] or "Shrub/Transitional Vegetation" in x[2]) and x not in chosen:
                chosen.append(x)
        # Priority 3: Deterministic shuffle of remaining patches
        rem = [x for x in pool if x not in chosen]
        rng.shuffle(rem)
        chosen.extend(rem[:max(0, target_n - len(chosen))])
        return [x[0] for x in chosen]

    selected = {
        "train": select(train_pool, 300),
        "validation": select(val_pool, 100),
        "test": select(test_pool, 100)
    }
    return selected


if __name__ == "__main__":
    patches_dir = "data/curated_patches"
    os.makedirs(patches_dir, exist_ok=True)
    selected = get_phase2_1_selected_patches()

    all_target_pids = set()
    for sp, p_list in selected.items():
        print(f"Selected {len(p_list)} for {sp}")
        all_target_pids.update(p_list)

    print(f"Total target patches: {len(all_target_pids)}")

    # Check which patches already have all 6 bands locally
    complete_local = set()
    for pid in all_target_pids:
        p_folder = os.path.join(patches_dir, pid)
        if os.path.isdir(p_folder):
            if all(os.path.exists(os.path.join(p_folder, f"{pid}_{b}.tif")) for b in SELECTED_BANDS):
                complete_local.add(pid)

    needed_pids = all_target_pids - complete_local
    print(f"Already complete on disk: {len(complete_local)}")
    print(f"Patches needed from Zenodo: {len(needed_pids)}")

    if len(needed_pids) > 0:
        print("[download] Initiating streaming connection to Zenodo archive...")
        curl_cmd = ["curl.exe" if sys.platform == "win32" else "curl", "-s", "-L", ZENODO_S2_ARCHIVE_URL]
        proc = subprocess.Popen(curl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stream = ZstdProcStream(proc.stdout)

        in_progress_bands: Dict[str, Set[str]] = {}
        downloaded_count = 0

        try:
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    parts = member.name.split("/")
                    if len(parts) < 4:
                        continue
                    pid = parts[-2]
                    if pid not in needed_pids:
                        continue

                    filename = parts[-1]
                    m = re.search(r"_(B[0-9A-Za-z]+)\.tif$", filename)
                    if not m:
                        continue
                    b = m.group(1)
                    if b not in SELECTED_BANDS:
                        continue

                    p_folder = os.path.join(patches_dir, pid)
                    os.makedirs(p_folder, exist_ok=True)
                    out_tif = os.path.join(p_folder, filename)

                    if not os.path.exists(out_tif):
                        f_in = tar.extractfile(member)
                        if f_in:
                            with open(out_tif, "wb") as f_out:
                                f_out.write(f_in.read())

                    in_progress_bands.setdefault(pid, set()).add(b)
                    if len(in_progress_bands[pid]) == len(SELECTED_BANDS):
                        downloaded_count += 1
                        remaining = len(needed_pids) - downloaded_count
                        if downloaded_count % 25 == 0 or remaining == 0:
                            mb = stream.bytes_read / (1024 * 1024)
                            print(f"[download] Downloaded {downloaded_count}/{len(needed_pids)} ({remaining} remaining) | Streamed {mb:.2f} MB")
                        if remaining == 0:
                            print("[download] All needed patches successfully retrieved!")
                            break
        finally:
            proc.kill()
            print("[download] Stream closed.")

    # Final check
    still_missing = [p for p in all_target_pids if not all(os.path.exists(os.path.join(patches_dir, p, f"{p}_{b}.tif")) for b in SELECTED_BANDS)]
    print(f"Final check: missing patches count = {len(still_missing)}")
