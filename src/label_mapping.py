"""
label_mapping.py
================
Official CORINE Land Cover (19 classes) to 7-Class Broad Scheme Mapping.
Provides multi-hot vector encoding, class lookup tables, and documentation.

The 7 Broad Target Classes:
  0: Forest
  1: Agriculture
  2: Grassland/Pasture
  3: Built-up/Artificial Surfaces
  4: Water
  5: Wetland
  6: Shrub/Transitional Vegetation
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# The 7 canonical broad classes
BROAD_CLASSES = [
    "Forest",
    "Agriculture",
    "Grassland/Pasture",
    "Built-up/Artificial Surfaces",
    "Water",
    "Wetland",
    "Shrub/Transitional Vegetation"
]

NUM_CLASSES = len(BROAD_CLASSES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(BROAD_CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(BROAD_CLASSES)}

# Explicit documented mapping table for all 19 BigEarthNet v2.0 CORINE classes
# Format: CORINE_LABEL -> (BROAD_CLASS, RATIONALE)
CORINE_MAPPING_SPEC: Dict[str, Tuple[Optional[str], str]] = {
    # 1. Forest
    "Broad-leaved forest": (
        "Forest",
        "Deciduous broad-leaved tree canopy cover > 30%"
    ),
    "Coniferous forest": (
        "Forest",
        "Evergreen needle-leaved coniferous tree canopy cover > 30%"
    ),
    "Mixed forest": (
        "Forest",
        "Neither broad-leaved nor coniferous species dominate canopy"
    ),

    # 2. Agriculture
    "Arable land": (
        "Agriculture",
        "Regularly cultivated agricultural parcels for annual crops"
    ),
    "Permanent crops": (
        "Agriculture",
        "Vineyards, orchards, olive groves, and berry plantations"
    ),
    "Complex cultivation patterns": (
        "Agriculture",
        "Mosaic of small cultivated parcels with scattered rural dwellings"
    ),
    "Land principally occupied by agriculture, with significant areas of natural vegetation": (
        "Agriculture",
        "Cultivated parcels interspersed with patches of natural vegetation"
    ),
    "Agro-forestry areas": (
        "Agriculture",
        "Annual or permanent crops combined with forest tree canopy"
    ),

    # 3. Grassland / Pasture
    "Pastures": (
        "Grassland/Pasture",
        "Dense, permanently managed herbaceous turf grazed by livestock"
    ),
    "Natural grassland and sparsely vegetated areas": (
        "Grassland/Pasture",
        "Low-productivity unmanaged wild grasslands and open steppe"
    ),

    # 4. Built-up / Artificial Surfaces
    "Urban fabric": (
        "Built-up/Artificial Surfaces",
        "Continuous and discontinuous residential, commercial and urban areas"
    ),
    "Industrial or commercial units": (
        "Built-up/Artificial Surfaces",
        "Factories, logistics facilities, commercial zones, and infrastructure"
    ),

    # 5. Water
    "Inland waters": (
        "Water",
        "Rivers, streams, canals, freshwater lakes, and reservoirs"
    ),
    "Marine waters": (
        "Water",
        "Ocean, sea, estuaries, and coastal lagoons"
    ),

    # 6. Wetland
    "Inland wetlands": (
        "Wetland",
        "Inland marshes, peatbogs, and seasonally waterlogged terrain"
    ),
    "Coastal wetlands": (
        "Wetland",
        "Salt marshes, salines, and tidal flats"
    ),

    # 7. Shrub / Transitional Vegetation
    "Transitional woodland, shrub": (
        "Shrub/Transitional Vegetation",
        "Bushy vegetation, forest clear-cuts, young plantation regrowth"
    ),
    "Moors, heathland and sclerophyllous vegetation": (
        "Shrub/Transitional Vegetation",
        "Low shrubby vegetation, heathland, gorse, and maquis"
    ),

    # Explicitly Documented Unmapped Class:
    "Beaches, dunes, sands": (
        None,
        "Coastal bare mineral/sand surfaces without vegetation (only 0.27% of dataset). "
        "Explicitly excluded from the 7 target classes to prevent arbitrary label contamination."
    ),
}

# Simplified direct dictionary: corine_label -> broad_class (or None)
CORINE_TO_BROAD: Dict[str, Optional[str]] = {
    k: v[0] for k, v in CORINE_MAPPING_SPEC.items()
}


def get_mapping_table() -> pd.DataFrame:
    """
    Returns the comprehensive mapping table as a Pandas DataFrame.
    """
    rows = []
    for corine, (broad, rationale) in sorted(CORINE_MAPPING_SPEC.items()):
        rows.append({
            "CORINE Label": corine,
            "Mapped Broad Class": broad if broad else "[Unmapped - Bare Natural]",
            "Status": "Mapped" if broad else "Excluded (Documented)",
            "Ecological Rationale": rationale
        })
    return pd.DataFrame(rows)


def map_corine_labels(corine_labels: List[str]) -> List[str]:
    """
    Maps a list of raw CORINE labels to the corresponding set of broad classes.
    """
    mapped_set = set()
    for lbl in corine_labels:
        broad = CORINE_TO_BROAD.get(lbl)
        if broad is not None:
            mapped_set.add(broad)
    return sorted(list(mapped_set))


def labels_to_multihot(labels: List[str]) -> np.ndarray:
    """
    Converts either CORINE labels or broad class labels into a 7-element
    binary multi-hot target vector: y in {0, 1}^7.
    """
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for lbl in labels:
        if lbl in CLASS_TO_IDX:
            vec[CLASS_TO_IDX[lbl]] = 1.0
        elif lbl in CORINE_TO_BROAD:
            broad = CORINE_TO_BROAD[lbl]
            if broad is not None:
                vec[CLASS_TO_IDX[broad]] = 1.0
    return vec


def multihot_to_labels(vec: np.ndarray) -> List[str]:
    """
    Converts a binary multi-hot vector back to a list of class name strings.
    """
    return [BROAD_CLASSES[i] for i in range(len(vec)) if vec[i] > 0.5]
