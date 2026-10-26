"""
data_access.py
==============
Handles data access and extraction for BigEarthNet-S2 v2.0 (reBEN).
Enables selective on-the-fly streaming extraction from the official Zenodo
archive without downloading the full ~59 GiB archive.
"""

import os
import sys
import subprocess
import tarfile
import io
import re
from typing import Set, List, Optional, Dict
import pyarrow.parquet as pq

ZENODO_METADATA_URL = "https://zenodo.org/records/10891137/files/metadata.parquet?download=1"
ZENODO_S2_ARCHIVE_URL = "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1"

DEFAULT_BANDS = {"B02", "B03", "B04", "B08", "B11", "B12"}


class ZstdProcStream(io.RawIOBase):
    """
    Wraps an active subprocess stdout stream and decompresses incoming
    Zstandard compressed chunks on the fly.
    """
    def __init__(self, stdout_pipe):
        import zstandard as zstd
        self.pipe = stdout_pipe
        self.dctx = zstd.ZstdDecompressor()
        self.decompressor = self.dctx.decompressobj()
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
            decompressed = self.decompressor.decompress(chunk)
            if decompressed:
                self.buffer.extend(decompressed)

        if not self.buffer:
            return 0

        n = min(len(b), len(self.buffer))
        b[:n] = self.buffer[:n]
        del self.buffer[:n]
        return n


def download_metadata(output_path: str = "data/metadata.parquet") -> str:
    """
    Downloads official BigEarthNet v2.0 metadata.parquet (3.44 MB) from Zenodo.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000000:
        print(f"[data_access] Metadata file already exists: {output_path} ({os.path.getsize(output_path):,} bytes)")
        return output_path

    print(f"[data_access] Downloading official metadata.parquet from Zenodo (~3.44 MB)...")
    cmd = ["curl.exe" if sys.platform == "win32" else "curl", "-s", "-L", ZENODO_METADATA_URL, "-o", output_path]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to download metadata.parquet from {ZENODO_METADATA_URL}")
    print(f"[data_access] Download complete: {output_path} ({os.path.getsize(output_path):,} bytes)")
    return output_path


def load_metadata_dict(parquet_path: str = "data/metadata.parquet") -> Dict[str, dict]:
    """
    Loads metadata into an in-memory dictionary keyed by patch_id.
    """
    if not os.path.exists(parquet_path):
        download_metadata(parquet_path)

    table = pq.read_table(parquet_path)
    patch_ids = table['patch_id'].to_pylist()
    labels = table['labels'].to_pylist()
    splits = table['split'].to_pylist()
    countries = table['country'].to_pylist()

    meta = {}
    for pid, lbl, sp, cnt in zip(patch_ids, labels, splits, countries):
        meta[pid] = {
            'labels': lbl,
            'split': sp,
            'country': cnt
        }
    return meta


def stream_extract_patches(
    output_dir: str = "data/patches",
    max_patches: int = 100,
    target_bands: Set[str] = DEFAULT_BANDS,
    patch_ids_filter: Optional[Set[str]] = None,
    verbose: bool = True
) -> Dict[str, Set[str]]:
    """
    Streams and extracts individual Sentinel-2 patches directly from the official
    Zenodo archive on the fly without downloading the full 59 GiB archive.

    Parameters:
        output_dir: Directory where patch subfolders will be created.
        max_patches: Maximum number of patches to extract before closing connection.
        target_bands: Set of band names to retain (e.g. {'B02', 'B03', 'B04', 'B08', 'B11', 'B12'}).
        patch_ids_filter: Optional set of specific patch IDs to retrieve.
        verbose: If True, prints extraction progress.

    Returns:
        Dict mapping patch_id to the set of extracted bands.
    """
    os.makedirs(output_dir, exist_ok=True)
    curl_bin = "curl.exe" if sys.platform == "win32" else "curl"
    cmd = [curl_bin, "-s", "-L", ZENODO_S2_ARCHIVE_URL]

    if verbose:
        print(f"[data_access] Initiating stream from Zenodo archive...")
        print(f"[data_access] Target patches: {max_patches} | Bands: {sorted(list(target_bands))}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stream = ZstdProcStream(proc.stdout)

    extracted: Dict[str, Set[str]] = {}

    try:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue

                parts = member.name.split("/")
                if len(parts) < 4:
                    continue

                patch_id = parts[-2]
                filename = parts[-1]

                # Match band suffix: e.g. _B02.tif
                m = re.search(r"_(B[0-9A-Za-z]+)\.tif$", filename)
                if not m:
                    continue
                band = m.group(1)

                if band not in target_bands:
                    continue

                if patch_ids_filter and patch_id not in patch_ids_filter:
                    continue

                patch_folder = os.path.join(output_dir, patch_id)
                os.makedirs(patch_folder, exist_ok=True)
                out_path = os.path.join(patch_folder, filename)

                f_in = tar.extractfile(member)
                if f_in:
                    with open(out_path, "wb") as f_out:
                        f_out.write(f_in.read())

                extracted.setdefault(patch_id, set()).add(band)

                if len(extracted[patch_id]) == len(target_bands):
                    if verbose and len(extracted) % 10 == 0:
                        mb = stream.bytes_read / (1024 * 1024)
                        print(f"[data_access] Extracted {len(extracted)}/{max_patches} patches ({mb:.2f} MB downloaded)")
                    if len(extracted) >= max_patches:
                        break
    finally:
        proc.kill()

    total_mb = stream.bytes_read / (1024 * 1024)
    if verbose:
        print(f"[data_access] Completed! Extracted {len(extracted)} patches.")
        print(f"[data_access] Total compressed data streamed: {total_mb:.2f} MB")

    return extracted
