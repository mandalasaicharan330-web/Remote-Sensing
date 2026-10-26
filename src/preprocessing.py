"""
preprocessing.py
================
Sentinel-2 multispectral patch preprocessing and spatial harmonization.
Handles:
  - Band reading and order validation: [B02, B03, B04, B08, B11, B12].
  - Bilinear spatial resampling of 20 m bands (B11, B12) from 60x60 to 120x120.
  - Surface reflectance scaling: DN / 10000.0 clipped to [0.0, 1.0].
  - Data integrity checks: missing values, NaN/Inf, data types, and value bounds.
  - Preservation of native raw GeoTIFF files (read-only).
"""

import os
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image

# Exact required band order and definitions
SELECTED_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

BAND_METADATA = {
    "B02": {"name": "Blue", "wavelength": 490, "native_res": 10, "native_shape": (120, 120)},
    "B03": {"name": "Green", "wavelength": 560, "native_res": 10, "native_shape": (120, 120)},
    "B04": {"name": "Red", "wavelength": 665, "native_res": 10, "native_shape": (120, 120)},
    "B08": {"name": "NIR", "wavelength": 842, "native_res": 10, "native_shape": (120, 120)},
    "B11": {"name": "SWIR-1", "wavelength": 1610, "native_res": 20, "native_shape": (60, 60)},
    "B12": {"name": "SWIR-2", "wavelength": 2190, "native_res": 20, "native_shape": (60, 60)},
}

# Standard target working grid (10 m resolution for 1.2 km patch)
TARGET_SHAPE = (120, 120)

# Physical reflectance quantification scale factor for Sentinel-2 Level-2A
REFLECTANCE_SCALE_FACTOR = 10000.0


def resample_band_bilinear(arr: np.ndarray, target_shape: Tuple[int, int] = TARGET_SHAPE) -> np.ndarray:
    """
    Resamples a 2D band array from its native resolution (e.g. 60x60 for 20 m bands)
    to target shape (120x120 for 10 m working grid) using bilinear interpolation.
    
    Bilinear interpolation is chosen because it produces smooth, continuous
    spatial gradients without introducing artificial step artifacts (unlike nearest neighbor)
    or overshooting values (unlike bicubic splines).
    """
    if arr.shape == target_shape:
        return arr.copy()
    img = Image.fromarray(arr)
    # PIL resize takes (width, height)
    resampled = img.resize((target_shape[1], target_shape[0]), resample=Image.BILINEAR)
    return np.array(resampled, dtype=arr.dtype)


def validate_single_band(
    tif_path: str,
    band_name: str
) -> Dict[str, any]:
    """
    Performs data integrity verification on a raw single-band GeoTIFF file.
    Verifies existence, data type, shape, NaNs, Infs, min, max, and mean.
    """
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"Raw band GeoTIFF not found: {tif_path}")

    with Image.open(tif_path) as img:
        arr = np.array(img)

    expected_shape = BAND_METADATA[band_name]["native_shape"]
    is_shape_valid = (arr.shape == expected_shape)
    is_dtype_valid = (arr.dtype == np.uint16)
    has_nan = np.isnan(arr).any() if np.issubdtype(arr.dtype, np.floating) else False
    has_inf = np.isinf(arr).any() if np.issubdtype(arr.dtype, np.floating) else False
    is_finite = not (has_nan or has_inf)

    return {
        "band": band_name,
        "path": tif_path,
        "native_shape": arr.shape,
        "expected_shape": expected_shape,
        "is_shape_valid": is_shape_valid,
        "dtype": str(arr.dtype),
        "is_dtype_valid": is_dtype_valid,
        "has_nan": bool(has_nan),
        "is_finite": bool(is_finite),
        "min_raw": int(arr.min()),
        "max_raw": int(arr.max()),
        "mean_raw": float(arr.mean())
    }


def load_and_preprocess_patch(
    patch_dir: str,
    patch_id: str,
    bands: List[str] = SELECTED_BANDS,
    target_shape: Tuple[int, int] = TARGET_SHAPE,
    scale_reflectance: bool = True
) -> Tuple[np.ndarray, Dict[str, dict]]:
    """
    Loads raw GeoTIFF bands for a patch, verifies integrity, resamples 20 m bands
    to 120x120, and applies physical reflectance scaling.

    Transformation Rationale:
      rho = clip(DN / 10000.0, 0.0, 1.0)
      - Division by 10,000 converts raw digital numbers to physical Bottom-Of-Atmosphere (BOA)
        surface reflectance as documented in the ESA Sentinel-2 L2A product specification.
      - Clipping to [0.0, 1.0] guarantees valid physical reflectance bounds (0% to 100%),
        suppressing rare negative atmospheric correction noise (<0) or saturated cloud highlights (>10,000).

    Returns:
      tensor: Preprocessed array of shape (len(bands), 120, 120) in float32.
      audit_info: Detailed per-band integrity report.
    """
    patch_folder = os.path.join(patch_dir, patch_id)
    if not os.path.exists(patch_folder):
        raise FileNotFoundError(f"Patch folder does not exist: {patch_folder}")

    band_arrays = []
    audit_info = {}

    for b in bands:
        tif_path = os.path.join(patch_folder, f"{patch_id}_{b}.tif")
        # 1. Audit raw band
        audit = validate_single_band(tif_path, b)
        audit_info[b] = audit

        # 2. Read raw array (preserves original file)
        with Image.open(tif_path) as img:
            arr = np.array(img)

        # 3. Spatial harmonization: resample 20 m bands to 120x120
        if arr.shape != target_shape:
            arr = resample_band_bilinear(arr, target_shape=target_shape)

        # 4. Conversion to float32
        arr_float = arr.astype(np.float32)

        # 5. Radiometric scaling to physical surface reflectance [0, 1]
        if scale_reflectance:
            arr_float = np.clip(arr_float / REFLECTANCE_SCALE_FACTOR, 0.0, 1.0)

        band_arrays.append(arr_float)

    # Stack along channel dimension: shape (6, 120, 120)
    preprocessed_cube = np.stack(band_arrays, axis=0)

    # Confirm final tensor constraints
    assert preprocessed_cube.shape == (len(bands), target_shape[0], target_shape[1]), \
        f"Unexpected output shape: {preprocessed_cube.shape}"
    assert np.isfinite(preprocessed_cube).all(), "Non-finite values found in preprocessed tensor!"
    assert preprocessed_cube.min() >= 0.0 and preprocessed_cube.max() <= 1.0, \
        f"Values outside [0, 1]: min={preprocessed_cube.min()}, max={preprocessed_cube.max()}"

    return preprocessed_cube, audit_info
