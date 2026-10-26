"""
data_preprocessing.py
=====================
Preprocesses Sentinel-2 multispectral patches for multi-label classification.
- Harmonies 10 m and 20 m spatial resolutions to a common 120x120 grid.
- Scales surface reflectance values to [0, 1].
- Maps 19 CORINE Land Cover classes to 7 broad project classes.
- Generates binary multi-hot ground truth vectors.
- Manages geographical train, validation, and test splits without data leakage.
"""

import os
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional

# The 7 simplified broad classes for the B.Tech practical
BROAD_CLASSES = [
    "Forest",
    "Agriculture",
    "Grassland_Pasture",
    "Built_up",
    "Water",
    "Wetland",
    "Shrub_Transitional"
]

NUM_CLASSES = len(BROAD_CLASSES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(BROAD_CLASSES)}

# Explicit documented mapping from BigEarthNet v2.0 19 CORINE classes to 7 broad classes
CORINE_TO_BROAD: Dict[str, str] = {
    # 1. Forest
    "Broad-leaved forest": "Forest",
    "Coniferous forest": "Forest",
    "Mixed forest": "Forest",
    
    # 2. Agriculture
    "Arable land": "Agriculture",
    "Permanent crops": "Agriculture",
    "Complex cultivation patterns": "Agriculture",
    "Land principally occupied by agriculture, with significant areas of natural vegetation": "Agriculture",
    "Agro-forestry areas": "Agriculture",
    
    # 3. Grassland & Pasture
    "Pastures": "Grassland_Pasture",
    "Natural grassland and sparsely vegetated areas": "Grassland_Pasture",
    
    # 4. Built-up / Artificial Surfaces
    "Urban fabric": "Built_up",
    "Industrial or commercial units": "Built_up",
    
    # 5. Water
    "Inland waters": "Water",
    "Marine waters": "Water",
    
    # 6. Wetland
    "Inland wetlands": "Wetland",
    "Coastal wetlands": "Wetland",
    
    # 7. Shrub & Transitional Vegetation
    "Transitional woodland, shrub": "Shrub_Transitional",
    "Moors, heathland and sclerophyllous vegetation": "Shrub_Transitional"
    
    # Note: 'Beaches, dunes, sands' represents < 0.3% of the dataset and is left unmapped
    # as a bare natural surface, preserving ground truth integrity.
}

# The 6 selected Sentinel-2 bands in standard spectral order
SELECTED_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def map_labels_to_broad(corine_labels: List[str]) -> List[str]:
    """
    Maps a list of raw CORINE labels to the set of broad classes.
    """
    broad_set = set()
    for lbl in corine_labels:
        if lbl in CORINE_TO_BROAD:
            broad_set.add(CORINE_TO_BROAD[lbl])
    return sorted(list(broad_set))


def labels_to_multi_hot(labels: List[str]) -> np.ndarray:
    """
    Converts either raw CORINE labels or mapped broad labels into a
    binary multi-hot target vector of length 7.
    """
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    # Check if labels are already broad or need mapping
    for l in labels:
        if l in CLASS_TO_IDX:
            vec[CLASS_TO_IDX[l]] = 1.0
        elif l in CORINE_TO_BROAD:
            broad_name = CORINE_TO_BROAD[l]
            vec[CLASS_TO_IDX[broad_name]] = 1.0
    return vec


def resample_band(arr: np.ndarray, target_shape: Tuple[int, int] = (120, 120)) -> np.ndarray:
    """
    Resamples a 2D band array (e.g. 60x60 for 20 m bands B11, B12)
    to the target 10 m shape (120x120) using bilinear interpolation.
    """
    if arr.shape == target_shape:
        return arr
    img = Image.fromarray(arr)
    resampled = img.resize((target_shape[1], target_shape[0]), resample=Image.BILINEAR)
    return np.array(resampled)


def load_patch_tensor(
    patch_dir: str,
    patch_id: str,
    bands: List[str] = SELECTED_BANDS,
    target_shape: Tuple[int, int] = (120, 120),
    normalize: bool = True
) -> np.ndarray:
    """
    Loads all specified bands for a patch, resamples to a common 120x120 grid,
    and returns a float32 tensor of shape (len(bands), 120, 120).

    Sentinel-2 L2A BOA reflectance values are stored as uint16 scaled by 10,000.
    When normalize=True, values are scaled by 10,000.0 into physical reflectance [0, 1].
    """
    band_arrays = []
    for b in bands:
        tif_path = os.path.join(patch_dir, patch_id, f"{patch_id}_{b}.tif")
        if not os.path.exists(tif_path):
            raise FileNotFoundError(f"Missing band file: {tif_path}")

        with Image.open(tif_path) as img:
            arr = np.array(img, dtype=np.float32)

        if arr.shape != target_shape:
            arr = resample_band(arr, target_shape=target_shape)

        if normalize:
            # BOA reflectance values: 10,000 = 100% reflectance (clipped to [0, 1])
            arr = np.clip(arr / 10000.0, 0.0, 1.0)

        band_arrays.append(arr)

    # Stack along channel dimension: shape (C, H, W)
    return np.stack(band_arrays, axis=0)


def create_curated_npz(
    patches_dir: str,
    metadata_dict: Dict[str, dict],
    output_npz: str = "data/curated_subset.npz",
    bands: List[str] = SELECTED_BANDS
) -> Dict[str, int]:
    """
    Consolidates extracted patch folders and their metadata into a single
    portable .npz file containing imagery, multi-hot labels, splits, and IDs.
    """
    patch_dirs = [d for d in os.listdir(patches_dir) if os.path.isdir(os.path.join(patches_dir, d))]
    images = []
    labels_list = []
    patch_ids = []
    splits = []
    countries = []

    for pid in sorted(patch_dirs):
        # Verify all bands exist
        all_bands_exist = True
        for b in bands:
            if not os.path.exists(os.path.join(patches_dir, pid, f"{pid}_{b}.tif")):
                all_bands_exist = False
                break
        if not all_bands_exist:
            continue

        meta = metadata_dict.get(pid)
        if not meta:
            continue

        tensor = load_patch_tensor(patches_dir, pid, bands=bands)
        multi_hot = labels_to_multi_hot(meta['labels'])

        # Skip patches where all labels were unmapped
        if multi_hot.sum() == 0:
            continue

        images.append(tensor)
        labels_list.append(multi_hot)
        patch_ids.append(pid)
        splits.append(meta['split'])
        countries.append(meta.get('country', 'Unknown'))

    images_arr = np.stack(images, axis=0)
    labels_arr = np.stack(labels_list, axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(output_npz)), exist_ok=True)
    np.savez_compressed(
        output_npz,
        images=images_arr,
        labels=labels_arr,
        patch_ids=np.array(patch_ids),
        splits=np.array(splits),
        countries=np.array(countries),
        class_names=np.array(BROAD_CLASSES),
        band_names=np.array(bands)
    )

    split_counts = {sp: int(np.sum(np.array(splits) == sp)) for sp in set(splits)}
    print(f"[data_preprocessing] Saved curated dataset to {output_npz}")
    print(f" - Total samples: {len(images_arr)} | Shape: {images_arr.shape}")
    print(f" - Splits: {split_counts}")
    return split_counts
