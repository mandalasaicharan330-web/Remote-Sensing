# Phase 2.1 — Dataset Audit Corrections and Class Coverage Expansion Report

**Project:** Multi-Label Land-Cover Recognition Using Sentinel-2 Multispectral Classification  
**Project Directory:** `C:\Users\manda\Downloads\Remote Sensing`  
**Dataset:** BigEarthNet v2.0 (Sentinel-2 L2A BOA Multispectral Imagery)  
**Author:** Antigravity (Pair Programming with Undergraduate Student in Applied Geophysics)  
**Verification Date:** October 26, 2026  
**Verification Script:** `verify_phase2_1.py` (Status: Passed with Exit Code 0)  

---

## Executive Summary

Phase 2 constructed and verified an initial set of 100 genuine Sentinel-2 patches. While geometric resampling, reflectance scaling, and pipeline integrity passed, auditing the class frequencies revealed a critical flaw: **Wetland and Shrub/Transitional Vegetation had zero positive training examples in the training split**. Evaluating or training a 7-class multi-label neural network on that dataset would cause severe gradient collapse or catastrophic zero-recall on those minority classes.

In Phase 2.1, we performed an exhaustive audit and deterministic expansion to produce a corrected, balanced dataset:
1. **Audited Phase 2 Dataset:** Confirmed 0 duplicate patch IDs, strict preservation of official splits (60 train, 20 val, 20 test), and identified the 0-frequency defect for Wetland and Shrub in training.
2. **Verified Seven-Class Mapping Scheme:** Validated all 19 official CORINE labels against the 7 target broad classes in the mandatory canonical order. The unmapped class (*Beaches, dunes, sands*) was verified to account for 0.27% (1,316 patches) of the full BigEarthNet dataset and exactly 0 patches in Austrian alpine tile `T33UUP`.
3. **Audited Radiometric Scaling & Bilinear Resampling:** Examined all 6 bands across millions of raw pixels. Established that values exceeding 10,000 DN (surface reflectance > 1.0) constitute only **0.0008% to 0.0042%** of all pixels, arising from physical specular glint (solar reflection on water, roofs, or greenhouse glass). Confirmed that dividing by 10,000.0 and clipping to $[0.0, 1.0]$ is physically and numerically sound, and that 20 m bands (B11, B12) are smoothly harmonized to $120 \times 120$ via bilinear interpolation without artifacts. Raw GeoTIFF files remain completely intact.
4. **Deterministic Class Coverage Expansion:** Expanded the dataset to **500 genuine Sentinel-2 patches** (300 Train, 100 Validation, 100 Test) using a deterministic selection procedure (`seed=42`). 100% of the original 100 patches (including all 5 Phase 1 test patches) were preserved without redownload. All available Wetland and Shrub patches in tile `T33UUP` were captured.
5. **Addressed Class Limitations:** Every single broad class now has positive examples in training (Wetland: 8, Shrub: 59). In the test set, Wetland (27) and Shrub (42) exceed the $\ge 20$ target. In the validation set, Wetland (16) and Shrub (13) reach 100% of all available instances in the entire tile.
6. **Zero Data Leakage:** Verified that Train, Validation, and Test patch sets are strictly disjoint ($Train \cap Val = 0$, $Train \cap Test = 0$, $Val \cap Test = 0$) and that 100% of patches retain their official BigEarthNet split assignments.
7. **Created Corrected Dataset & Artifacts:** Built `data/curated_dataset_phase2_1.npz` (76.66 MB), updated all metric inventories, and generated high-resolution true-colour and false-colour NIR composite figures.

---

## Step 1 — Audit of the Phase 2 Curated Dataset

We independently inspected `data/curated_dataset_phase2.npz`, `data/metadata.parquet`, `data/sample_patches/`, `src/label_mapping.py`, `src/preprocessing.py`, `src/dataset.py`, `verify_phase2.py`, and existing metrics.

### Audit Findings Summary
* **Total Patches:** 100 genuine BigEarthNet patches.
* **Unique Patch IDs:** 100 (0 duplicates detected).
* **Actual Split Counts:**
  * Train: 60 patches (60.0%)
  * Validation: 20 patches (20.0%)
  * Test: 20 patches (20.0%)
* **Array Dimensions & Numerical Health:**
  * Image tensor shape: `(100, 6, 120, 120)`, dtype `float32`.
  * Label tensor shape: `(100, 7)`, dtype `float32`.
  * Non-finite values: Exactly 0 NaNs and 0 Infs across all 8.64 million array elements.
  * Reflectance values: Strictly bounded within $[0.0001, 1.0000]$.
* **Multi-Label Complexity:**
  * Minimum labels per patch: 1.0
  * Maximum labels per patch: 5.0
  * Mean labels per patch: 3.46
* **Phase 1 Inspection Patch Integrity:**
  * All 5 original Phase 1 inspection patches (`T33UUP_26_57`, `T33UUP_27_55`, `T33UUP_27_56`, `T33UUP_27_57`, `T33UUP_27_58`) were confirmed present exactly once.
  * All 5 resided strictly in the `test` split, perfectly matching their official assignment in `metadata.parquet`.

### Phase 2 Class Frequency Audit Table

| Class Index | Target Broad Class | Train Count (N=60) | Val Count (N=20) | Test Count (N=20) | Total Positive Patches | Audit Finding / Defect |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 0 | Forest | 46 (76.7%) | 20 (100.0%) | 20 (100.0%) | 86 | Well represented |
| 1 | Agriculture | 34 (56.7%) | 15 (75.0%) | 15 (75.0%) | 64 | Well represented |
| 2 | Grassland/Pasture | 51 (85.0%) | 19 (95.0%) | 12 (60.0%) | 82 | Well represented |
| 3 | Built-up/Artificial | 47 (78.3%) | 15 (75.0%) | 7 (35.0%) | 69 | Well represented |
| 4 | Water | 20 (33.3%) | 10 (50.0%) | 13 (65.0%) | 43 | Well represented |
| 5 | **Wetland** | **0 (0.0%)** | 2 (10.0%) | **0 (0.0%)** | 2 | **CRITICAL DEFECT: 0 in Train, 0 in Test** |
| 6 | **Shrub/Transitional** | **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** | 0 | **CRITICAL DEFECT: 0 across all splits** |

**Conclusion of Step 1:** Phase 2 was technically valid in terms of file structure and data types, but statistically incapable of supporting multi-label classification training for Wetland and Shrub.

---

## Step 2 — Verification of the Seven-Class Mapping Scheme

The target land-cover classes and canonical indices are strictly defined as follows:

```
0: Forest
1: Agriculture
2: Grassland/Pasture
3: Built-up/Artificial Surfaces
4: Water
5: Wetland
6: Shrub/Transitional Vegetation
```

### Official CORINE Land Cover (CLC 19) Mapping Table

| CORINE Land Cover Label | Mapped Target Broad Class | Ecological & Spectroradiometric Rationale | Status |
|:---|:---|:---|:---:|
| Broad-leaved forest | **Forest** | Deciduous broad-leaved tree canopy cover > 30% | Mapped |
| Coniferous forest | **Forest** | Evergreen needle-leaved coniferous canopy cover > 30% | Mapped |
| Mixed forest | **Forest** | Mosaic/blend where neither broad-leaved nor needle-leaved dominates | Mapped |
| Arable land | **Agriculture** | Regularly cultivated parcels for annual agricultural crops | Mapped |
| Permanent crops | **Agriculture** | Vineyards, orchards, fruit trees, and berry plantations | Mapped |
| Complex cultivation patterns | **Agriculture** | Small parcel mosaics with interspersed rural infrastructure | Mapped |
| Land principally occupied by agriculture, with natural veg. | **Agriculture** | Farmland with significant patches of natural woodland/hedgerows | Mapped |
| Agro-forestry areas | **Agriculture** | Annual crops or pasture combined with forest trees | Mapped |
| Pastures | **Grassland/Pasture** | Dense, managed perennial herbaceous cover grazed by livestock | Mapped |
| Natural grassland and sparsely vegetated areas | **Grassland/Pasture** | Low-productivity alpine turf, open steppes, and rocky grasslands | Mapped |
| Urban fabric | **Built-up/Artificial** | Continuous and discontinuous residential, civic, and urban structures | Mapped |
| Industrial or commercial units | **Built-up/Artificial** | Factories, transport facilities, commercial logistics, and docks | Mapped |
| Inland waters | **Water** | Freshwater lakes, reservoirs, canals, and river channels | Mapped |
| Marine waters | **Water** | Coastal bays, seas, fjords, and open ocean waters | Mapped |
| Inland wetlands | **Wetland** | Freshwater marshes, peatbogs, and seasonally waterlogged soils | Mapped |
| Coastal wetlands | **Wetland** | Salt marshes, salines, and tidal flats subject to marine tides | Mapped |
| Transitional woodland, shrub | **Shrub/Transitional** | Bushy regrowth, young saplings, clear-cut regeneration, maquis | Mapped |
| Moors, heathland and sclerophyllous vegetation | **Shrub/Transitional** | Ericaceous shrubs, dwarf gorse, alpine scrub, and dry heaths | Mapped |
| **Beaches, dunes, sands** | **[None - Unmapped]** | Bare coastal sand/shingle; lacks vegetation; excluded by design | **Explicitly Excluded** |

### Audit of Unmapped Labels
* **Rule Verification:** The code does not silently reassign unmapped labels to arbitrary classes. Unmapped labels evaluate to `None` and contribute zero 1s to target multi-hot vectors.
* **Archive-Wide Census:** Across all 480,038 patches in `metadata.parquet`, *Beaches, dunes, sands* occurs in only **1,316 patches (0.27%)**.
* **Tile `T33UUP` Census:** In the Austrian tile `T33UUP` (3,161 total patches), there are **0 occurrences** of *Beaches, dunes, sands*. All 18 present land-cover classes map without information loss.

---

## Step 3 — Radiometric Scaling and Resampling Audit

Sentinel-2 Level-2A products provide Bottom-Of-Atmosphere (BOA) surface reflectance processed through ESA's Sen2Cor algorithm.

### Scaling Convention & Specular Reflection Audit
The ESA Sentinel-2 L2A quantification convention is:
$$\text{DN} = 10,000 \times \rho$$
where $\rho$ represents surface reflectance. Theoretically, surface reflectance for natural surfaces lies in $[0.0, 1.0]$ ($\text{DN} \in [0, 10000]$).

We audited all **500 genuine patches** across all 6 spectral bands (7,200,000 pixels for 10 m bands, 1,800,000 pixels for 20 m bands):

| Band | Central $\lambda$ | Native Res. | Native Shape | Resampled Shape | Interpolation | Raw Dtype | Raw Min | Raw Max | Pixels DN > 10,000 | Pct DN > 10,000 | Non-Finite |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B02** (Blue) | 490 nm | 10 m | $120 \times 120$ | $120 \times 120$ | None (Native) | uint16 | 1 | 17,760 | 57 / 7.2M | 0.00079% | 0 |
| **B03** (Green) | 560 nm | 10 m | $120 \times 120$ | $120 \times 120$ | None (Native) | uint16 | 1 | 16,816 | 58 / 7.2M | 0.00081% | 0 |
| **B04** (Red) | 665 nm | 10 m | $120 \times 120$ | $120 \times 120$ | None (Native) | uint16 | 1 | 16,192 | 70 / 7.2M | 0.00097% | 0 |
| **B08** (NIR) | 842 nm | 10 m | $120 \times 120$ | $120 \times 120$ | None (Native) | uint16 | 78 | 15,616 | 80 / 7.2M | 0.00111% | 0 |
| **B11** (SWIR-1) | 1610 nm | 20 m | $60 \times 60$ | $120 \times 120$ | Bilinear (`order=1`) | uint16 | 34 | 15,125 | 53 / 1.8M | 0.00294% | 0 |
| **B12** (SWIR-2) | 2190 nm | 20 m | $60 \times 60$ | $120 \times 120$ | Bilinear (`order=1`) | uint16 | 9 | 15,063 | 75 / 1.8M | 0.00417% | 0 |

### Key Physical & Preprocessing Conclusions:
1. **No Negative Values:** All raw pixels satisfy $\text{DN} \ge 1$. Zero negative radiance or reflectance values exist.
2. **Values Above 10,000 DN:** Fewer than **0.0042%** of pixels exceed 10,000 DN across any band. Inspection reveals these correspond to genuine physical optical highlights: solar specular glint on water bodies, metallic roofs in industrial zones, and greenhouse glass panels. They are not data corruptions.
3. **Reflectance Conversion:** Preprocessing computes:
   $$\rho = \text{clip}\left(\frac{\text{DN}}{10000.0}, 0.0, 1.0\right)$$
   This normalizes all inputs into the stable range $[0.0, 1.0]$ for neural network training while preserving valid vegetation and water reflectance dynamics.
4. **Resampling Verification:** B11 and B12 are resampled from $60 \times 60$ to $120 \times 120$ via bilinear interpolation (`order=1`), ensuring continuous spatial gradients without ringing artifacts or edge oscillations.
5. **Preservation of Raw Files:** All source GeoTIFF files remain completely intact and unmodified in `data/curated_patches/`.

---

## Step 4 — Class Coverage Expansion Procedure

### Tile `T33UUP` Geographic Census
To ensure maximum reproducibility and avoid downloading the massive 59 GiB archive, we leveraged the first tile in BigEarthNet: `T33UUP` (Austria, UTM 33N). The complete tile contains 3,161 patches:
* **Train split:** 1,331 patches (Forest: 1,119, Agriculture: 817, Grassland: 797, Built-up: 595, Water: 149, **Wetland: 8**, **Shrub: 59**)
* **Validation split:** 813 patches (Forest: 614, Agriculture: 675, Grassland: 544, Built-up: 391, Water: 106, **Wetland: 16**, **Shrub: 13**)
* **Test split:** 1,017 patches (Forest: 825, Agriculture: 863, Grassland: 583, Built-up: 466, Water: 103, **Wetland: 27**, **Shrub: 42**)

### Inherent Geographic Constraints of Tile `T33UUP`
* **Wetland:** In the entire Austrian alpine region captured by `T33UUP`, there exist **only 8 Train patches and 16 Validation patches** with Wetland land cover. Achieving $\ge 20$ positive Wetland patches in Train or Validation is mathematically impossible within this tile.
* **Shrub:** There exist **only 13 Validation patches** with Shrub cover in the entire tile.
* Rather than fabricating data or violating official split boundaries, we captured **100% of all available Wetland and Shrub patches** in the tile across all three splits. In the Test set, Wetland (27) and Shrub (42) both comfortably exceed the $\ge 20$ target!

### Deterministic Selection Rules (`seed = 42`)
For each official split pool (`train`, `validation`, `test`):
1. **Priority 1 (Reuse Local Data):** Retain all 100 genuine patches already downloaded in Phase 2 (including all 5 Phase 1 inspection patches).
2. **Priority 2 (Capture Rare Classes):** Add all patches in the split containing *Wetland* or *Shrub/Transitional Vegetation*.
3. **Priority 3 (Deterministic Remainder):** Deterministically shuffle the remaining candidates using `np.random.RandomState(42)` and select sufficient patches to satisfy the split target counts (300 Train, 100 Val, 100 Test).

### Streaming Performance
* Patches needed from Zenodo: **400 patches** (100 already existed on disk).
* Streaming execution: Downloaded via `curl.exe` piped into `zstandard.decompressobj()` and `tarfile` in Python.
* Total data streamed: **387.12 MB** from byte 0 of `BigEarthNet-S2.tar.zst` (skipping 58.6 GiB of unneeded data).
* Total elapsed streaming time: **3 minutes 38 seconds** with 0 errors.

---

## Step 5 — Corrected Phase 2.1 Dataset Inventory & Statistics

The corrected dataset is saved as:
`data/curated_dataset_phase2_1.npz` (Size: **76.66 MB**)

### Corrected Split & Class Frequency Distribution (N = 500)

| Index | Target Broad Class | Train Count (N=300) | Train (%) | Val Count (N=100) | Val (%) | Test Count (N=100) | Test (%) | Total Positive Patches | Total (%) | Target Met ($\ge 20$ Val/Test) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | **Forest** | 258 | 86.0% | 84 | 84.0% | 95 | 95.0% | 437 | 87.4% | Yes ($\ge 20$ in all) |
| 1 | **Agriculture** | 142 | 47.3% | 74 | 74.0% | 52 | 52.0% | 268 | 53.6% | Yes ($\ge 20$ in all) |
| 2 | **Grassland/Pasture** | 209 | 69.7% | 75 | 75.0% | 39 | 39.0% | 323 | 64.6% | Yes ($\ge 20$ in all) |
| 3 | **Built-up/Artificial** | 130 | 43.3% | 58 | 58.0% | 31 | 31.0% | 219 | 43.8% | Yes ($\ge 20$ in all) |
| 4 | **Water** | 41 | 13.7% | 28 | 28.0% | 21 | 21.0% | 90 | 18.0% | **Yes ($\ge 20$ in all)** |
| 5 | **Wetland** | **8** | **2.7%** | **16** | **16.0%** | **27** | **27.0%** | **51** | **10.2%** | **Test $\ge 20$ Met**; Val at 100% tile max (16) |
| 6 | **Shrub/Transitional** | **59** | **19.7%** | **13** | **13.0%** | **42** | **42.0%** | **114** | **22.8%** | **Test $\ge 20$ Met**; Val at 100% tile max (13) |

### Key Improvements Over Phase 2
* **Positive Training Examples:** Wetland increased from **0 to 8**; Shrub increased from **0 to 59**.
* **Zero Zero-Positive Splits:** All 7 broad classes now contain positive instances across Train, Validation, and Test.
* **Test Set Reliability:** Every single class has $\ge 21$ positive patches in the Test set (Forest: 95, Agriculture: 52, Grassland: 39, Built-up: 31, Water: 21, Wetland: 27, Shrub: 42), guaranteeing statistically robust independent evaluation for all classes.
* **Preservation of Phase 1:** All 5 Phase 1 inspection patches remain in the test split.

---

## Step 6 — Visual Verification of Multispectral Composites

We generated two 3×3 visualization matrices displaying 9 representative genuine patches from Train, Validation, and Test:
1. `outputs/figures/phase2_1_true_colour.png`: True-Colour RGB composites using Bands B04 (Red), B03 (Green), and B02 (Blue), with 2%–98% percentile contrast enhancement.
2. `outputs/figures/phase2_1_false_colour.png`: False-Colour NIR composites using Bands B08 (NIR) $\rightarrow$ Red, B04 (Red) $\rightarrow$ Green, and B03 (Green) $\rightarrow$ Blue.

### Multi-Band Diagnostic Highlights
* **Vegetation & Biomass:** In True-Colour RGB, forest canopy appears dark olive-green, while pastures and arable crops show lighter green tones. In False-Colour NIR, dense photosynthetic foliage reflects intensely at 842 nm, rendering vigorous forest and crops as vibrant scarlet red.
* **Hydrology & Water:** Inland water bodies absorb strongly in NIR and SWIR, appearing jet black or navy in false colour, providing sharp delineation against surrounding banks.
* **Wetlands & Shrub Transition:** Wetland patches exhibit mixed moisture absorption and emergent vegetation, appearing as dark greenish-brown textures in True-Colour and characteristic mottled cyan/dark crimson in NIR false-colour. Shrub and transitional woodlands show intermediate NIR reflectance, distinctly lower than dense mature forest.

---

## Step 7 — Artifact Inventory

| Category | File Path | Format / Size | Status | Purpose |
|:---|:---|:---:|:---:|:---|
| **Dataset** | `data/curated_dataset_phase2_1.npz` | NPZ (76.66 MB) | **Created** | 500 preprocessed genuine Sentinel-2 patches with multi-hot labels |
| **Dataset (Original)** | `data/curated_dataset_phase2.npz` | NPZ (16.34 MB) | **Preserved** | Untouched Phase 2 baseline dataset (100 patches) |
| **Raw GeoTIFFs** | `data/curated_patches/` (500 subfolders) | TIFF (~63 MB) | **Expanded** | Raw 16-bit GeoTIFF bands (B2, B3, B4, B8, B11, B12) |
| **Metric Table** | `outputs/metrics/phase2_1_patch_inventory.csv` | CSV (131 KB) | **Created** | Complete metadata, labels, and band statistics for all 500 patches |
| **Metric Table** | `outputs/metrics/phase2_1_class_frequencies.csv` | CSV (474 B) | **Created** | Class counts and percentages across Train, Val, Test, and Total |
| **Metric Table** | `outputs/metrics/phase2_1_band_validation.csv` | CSV (1.2 KB) | **Created** | Radiometric, geometric, and non-finite audit across 6 bands |
| **Figure** | `outputs/figures/phase2_1_true_colour.png` | PNG (2.9 MB) | **Created** | 3×3 True-Colour (B4-B3-B2) composite matrix |
| **Figure** | `outputs/figures/phase2_1_false_colour.png` | PNG (3.3 MB) | **Created** | 3×3 False-Colour NIR (B8-B4-B3) composite matrix |
| **Code** | `src/dataset.py` | Python | **Updated** | Added `curate_phase2_1_dataset()` function |
| **Verification** | `verify_phase2_1.py` | Python | **Created** | Autonomous independent test suite (all assertions pass) |
| **Report** | `report/phase2_1_audit_report.md` | Markdown | **Created** | Comprehensive audit documentation |

---

## Step 8 — Phase 3 Readiness Assessment

The dataset audit and corrections for Phase 2.1 are complete:
* [x] Genuine BigEarthNet Sentinel-2 L2A patches utilized exclusively.
* [x] Official BigEarthNet train, validation, and test split assignments strictly preserved.
* [x] Zero data leakage across splits ($Train \cap Val = 0$, $Train \cap Test = 0$, $Val \cap Test = 0$).
* [x] 0 duplicate patch IDs.
* [x] Phase 1 inspection patches preserved strictly in the test split.
* [x] Radiometric scaling ($\text{DN} / 10000.0$) and clipping ($[0.0, 1.0]$) mathematically verified; specular highlights accounted for.
* [x] 20 m bands (B11, B12) resampled via bilinear interpolation to $120 \times 120$.
* [x] Positive training examples established for all 7 broad classes (Wetland: 8, Shrub: 59).
* [x] Test set possesses $\ge 21$ positive patches for every class (statistically sound evaluation).
* [x] Source data limitations for tile `T33UUP` fully documented without data fabrication.
* [x] Verification script `verify_phase2_1.py` runs cleanly and exits with code 0.

**Decision:** The dataset is fully validated, mathematically consistent, and **ready for Phase 3 Model Training**. Model training has not been started, pending user review and approval.
